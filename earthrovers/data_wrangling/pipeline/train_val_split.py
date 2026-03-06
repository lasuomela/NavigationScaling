"""
Cluster the rides based on their GPS locations and split into train/val sets.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import BallTree


from earthrovers.data_wrangling.pipeline.utils import get_ride_sensor_path
from earthrovers.data_wrangling.pipeline.data_loading import _load_gps_measurement
from tqdm import tqdm

EARTH_RADIUS_M = 6_371_000.0
RADIUS_M = 100.0
RADIUS_RAD = RADIUS_M / EARTH_RADIUS_M

# Spherical centroid per cluster from first points
def _spherical_centroid_latlon(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[float, float]:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    x_m, y_m, z_m = x.mean(), y.mean(), z.mean()
    lon_c = np.arctan2(y_m, x_m)
    hyp = np.hypot(x_m, y_m)
    lat_c = np.arctan2(z_m, hyp)

    # Find the point closest to the centroid
    distances = haversine_distances(
        np.column_stack((lat, lon)),
        np.array([[lat_c, lon_c]])
    )
    closest_idx = np.argmin(distances)
    lat_c = lat_deg[closest_idx]
    lon_c = lon_deg[closest_idx]
    return lat_c, lon_c, np.max(distances)*6371  # Return the maximum distance in km

def cluster_centroids_from_first_points(
    gps_meas: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns: DataFrame [cluster_id, centroid_lat, centroid_lon, n_rides]
    Centroid uses first GPS point of each ride in the cluster.
    """
    first_pts = gps_meas.groupby('ride_id', sort=False, as_index=False).first()[['ride_id', 'latitude', 'longitude']]    
    df = labels.merge(first_pts, on='ride_id', how='left', validate='one_to_one')

    # group and compute spherical centroid
    groups = df.groupby('cluster', sort=False)
    rows = []
    for cid, g in groups:
        lat_c, lon_c, intra_cluster_max_distance = _spherical_centroid_latlon(g['latitude'].to_numpy(), g['longitude'].to_numpy())
        rows.append({'cluster': cid, 'centroid_lat': lat_c, 'centroid_lon': lon_c, 'n_rides': len(g), 'intra_cluster_max_distance (km)': intra_cluster_max_distance})
    return pd.DataFrame(rows)

def load_gps_measurement(ride_path: Path):
    """
    Load the GPS data for a ride.
    """
    if isinstance(ride_path, pd.Series):
        ride_path = ride_path.iloc[0]

    gps_path = get_ride_sensor_path(ride_path, 'gps')
    if not gps_path:
        return None
    
    gps_data = _load_gps_measurement(gps_path)
    if gps_data.empty:
        return None
    
    gps_data = gps_data['gps']

    # Pick a measurement every 20 seconds
    gps_data = gps_data.iloc[::20].reset_index(drop=True)
    
    return gps_data

def min_pairwise_ride_distances_streaming(
    gps_meas: pd.DataFrame,
    radius_m: float = 100.0,
    batch_points: int = 20000, # Decrease this if you get OOM errors
    leaf_size: int = 40,
) -> pd.DataFrame:
    """
    Compute the minimum pairwise distances between all rides using a batched BallTree radius query approach.

    gps_meas: DataFrame with columns ['ride_id','lat','lon'] (degrees).
    Returns: DataFrame ['ride_a','ride_b','min_dist_m'] with unique (unordered) ride pairs.
    """

    assert {'ride_id','latitude','longitude'}.issubset(gps_meas.columns)

    # Map ride_id to compact int codes to keep memory small
    ride_codes, ride_uniques = pd.factorize(gps_meas['ride_id'], sort=False)
    ride_codes = ride_codes.astype(np.int32, copy=False)
    coords_rad = np.deg2rad(gps_meas[['latitude','longitude']].to_numpy(dtype=np.float64))

    # Build BallTree (haversine expects [lat, lon] in radians; distances are in radians)
    tree = BallTree(coords_rad, metric='haversine', leaf_size=leaf_size)
    r_rad = radius_m / EARTH_RADIUS_M

    N = coords_rad.shape[0]
    dmin = {}  # key: (a<<32)|b  (a<b), value: min distance in meters (float)

    # Helper to fold a batch's results into the global minima dict
    def fold_min(keys: np.ndarray, dists_m: np.ndarray):
        # per-batch min per key
        dfb = pd.DataFrame({'key': keys, 'd': dists_m})
        gb = dfb.groupby('key', sort=False)['d'].min()
        for k, v in gb.items():
            old = dmin.get(k)
            if old is None or v < old:
                dmin[k] = v

    # Stream queries in batches to bound peak memory
    starts = range(0, N, batch_points)
    for start in tqdm(starts, desc=f"Calculating ride pairwise distances within ({radius_m} m...)", total=(N + batch_points - 1)//batch_points):
        stop = min(start + batch_points, N)
        # query only for this slice as the "queries"; neighbors are from whole set
        ind_list, dist_list = tree.query_radius(coords_rad[start:stop], r=r_rad, return_distance=True, sort_results=True)

        # Flatten
        counts = np.fromiter((len(ix) for ix in ind_list), dtype=np.int32, count=stop-start)
        if counts.sum() == 0:
            continue
        i_local = np.repeat(np.arange(stop - start, dtype=np.int32), counts)
        j_all = np.concatenate(ind_list)
        d_all_rad = np.concatenate(dist_list)

        # Global indices of query points
        i_all = i_local + start

        # Drop self-matches
        mask = i_all != j_all
        if not mask.all():
            i_all = i_all[mask]; j_all = j_all[mask]; d_all_rad = d_all_rad[mask]

        # Keep only inter-ride pairs
        ri = ride_codes[i_all]
        rj = ride_codes[j_all]
        inter = ri != rj
        if not inter.any():
            continue
        ri = ri[inter]; rj = rj[inter]; d_all_rad = d_all_rad[inter]

        # Canonicalize unordered pair (a<b)
        a = np.minimum(ri, rj).astype(np.int32, copy=False)
        b = np.maximum(ri, rj).astype(np.int32, copy=False)

        # Pack pair to a 64-bit key: (a<<32)|b
        keys = (a.astype(np.int64) << 32) | b.astype(np.int64)
        dists_m = d_all_rad * EARTH_RADIUS_M

        # Fold into global minima
        fold_min(keys, dists_m)

    # Materialize final DataFrame
    if not dmin:
        return pd.DataFrame(columns=['ride_a','ride_b','min_dist_m'])

    keys = np.fromiter(dmin.keys(), dtype=np.int64, count=len(dmin))
    mins = np.fromiter(dmin.values(), dtype=np.float64, count=len(dmin))
    a = (keys >> 32).astype(np.int32)
    b = (keys & 0xFFFFFFFF).astype(np.int32)

    out = pd.DataFrame({
        'ride_a': ride_uniques[a],
        'ride_b': ride_uniques[b],
        'min_dist_m': mins
    })

    return out

def cluster_rides(edges: pd.DataFrame, all_rides: pd.Index, radius_m: float = 100.0) -> pd.DataFrame:
    """
    edges columns: ride_a, ride_b, min_dist_m (output of your min-distance step)
    all_rides: index/array of all ride_ids present in gps_meas
    Returns: DataFrame [ride_id, cluster_id]
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # union pairs within threshold
    if 'min_dist_m' in edges.columns:
        close = edges[edges['min_dist_m'] <= radius_m]
    else:
        close = edges  # if you've already filtered
    for a, b in close[['ride_a', 'ride_b']].itertuples(index=False):
        union(a, b)

    # ensure all rides appear (singletons)
    for r in all_rides:
        find(r)

    roots = {r: find(r) for r in all_rides}
    # stable, compact cluster IDs
    root_order = pd.Index(pd.unique(pd.Series(roots.values())))
    root_to_id = {root: i for i, root in enumerate(root_order)}
    labels = pd.DataFrame({
        'ride_id': list(all_rides),
        'cluster': [root_to_id[roots[r]] for r in all_rides]
    })
    return labels

def cluster_rides_by_location(ride_df: pd.DataFrame, fetch_location_name: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster rides by their GPS locations.
    """
    gps_stamps = ride_df.copy(deep=True)
    paths = gps_stamps['ride_path']

    # Generator that yields one DataFrame per ride (and tags it with ride_id)
    def iter_gps_frames():
        for ride_id, ride_path in paths.items():
            df = load_gps_measurement(ride_path)
            if df is None or df.empty:
                continue
            # Keep track of which ride each row came from
            yield df.assign(ride_id=ride_id)

    # Concatenate directly from the generator (no intermediate Series/list)
    gps_meas = pd.concat(
        tqdm(iter_gps_frames(), total=len(paths), desc="Loading GPS data for clustering"),
        ignore_index=True
    )

    edges = min_pairwise_ride_distances_streaming(gps_meas)
    ride_clusters = cluster_rides(edges, gps_meas['ride_id'].unique(), radius_m=100.0)
    cluster_stats = cluster_centroids_from_first_points(gps_meas, ride_clusters)

    # Get the country and city for each cluster
    if fetch_location_name:
        print("Fetching location names for clusters, this will take a while...")
        geolocator = Nominatim(user_agent="earthrovers")
        reverse_gc = RateLimiter(geolocator.reverse, min_delay_seconds=1.0)
        cluster_stats['country'] = cluster_stats.apply(
            lambda row: reverse_gc((row['centroid_lat'], row['centroid_lon']), exactly_one=True, language='en').raw['address'].get('country', 'Unknown'),
            axis=1
        )
        cluster_stats['city'] = cluster_stats.apply(
            lambda row: reverse_gc((row['centroid_lat'], row['centroid_lon']), exactly_one=True, language='en').raw['address'].get('city', 'Unknown'),
            axis=1
        )

    return ride_clusters, cluster_stats

def train_val_split(
        ride_df: pd.DataFrame,
        val_size: float = 0.05,
        val_locations: List[Tuple[float, float]] = None,
        fetch_location_name: bool = True,
    ):
    """
    Split the data into train and validation sets based on GPS locations of the rides.
    """

    ride_cache = ride_df['ride_path'].iloc[0].parent.parent.parent / 'ride_clusters.csv'
    cluster_stats_cache = ride_cache.parent / 'ride_clusters_stats.csv'

    if not ride_cache.exists() or not cluster_stats_cache.exists():
        print("Clustering rides by location...")
        ride_clusters, cluster_stats = cluster_rides_by_location(ride_df, fetch_location_name=fetch_location_name)
        ride_clusters.to_csv(ride_cache, index=False)
    else:
        print(f"Loading ride clusters from {ride_cache}")
        ride_clusters = pd.read_csv(ride_cache, dtype={'ride_id': str, 'cluster': int})
        cluster_stats = pd.read_csv(cluster_stats_cache)

    # Set the index to ride_id for easier merging
    ride_clusters.set_index('ride_id', inplace=True)
    ride_df = ride_df.join(ride_clusters, on='ride_id', how='left')

    if val_locations is not None:
        # Find any cluster centroids that are within 2 km of the validation locations
        val_locations = np.radians(val_locations)
        cluster_centroids = np.radians(cluster_stats[['centroid_lat', 'centroid_lon']].values)
        distances = haversine_distances(val_locations, cluster_centroids) * 6371  # Convert to km

        # Find clusters that are within 2 km of any validation location
        close_clusters = np.any(distances < 2, axis=0)
        train_clusters = cluster_stats[~close_clusters]['cluster'].values
        val_clusters = cluster_stats[close_clusters]['cluster'].values

        train_df = ride_df[ride_df['cluster'].isin(train_clusters)]
        val_df = ride_df[ride_df['cluster'].isin(val_clusters)] 
        
    else:
        # Assign the ride clusters to the train and validation sets based on rides per cluster
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        train_idx, val_idx = next(gss.split(ride_df, groups=ride_df['cluster']))
        train_df = ride_df.iloc[train_idx]
        val_df = ride_df.iloc[val_idx]

    # Add information about train / val split assignment into cluster stats
    train_clusters = train_df['cluster'].unique()
    val_clusters = val_df['cluster'].unique()
    cluster_stats['split'] = 'train'
    cluster_stats.loc[cluster_stats.index.isin(val_clusters), 'split'] = 'val'

    if not cluster_stats_cache.exists():
        # Save the cluster stats to a CSV file
        cluster_stats.to_csv(cluster_stats_cache, index=False)

    # Print the cluster stats
    print("Cluster stats:")
    print(cluster_stats.to_string())
    print()
    print(f"Number of clusters in train: {len(train_clusters)}")
    print(f"Number of clusters in val: {len(val_clusters)}")

    return {'train': train_df, 'val': val_df}