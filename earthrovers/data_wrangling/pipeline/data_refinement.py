
"""
Refine the raw ride data by calculating additional information such as compass yaw and navigation objectives.
"""
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from sklearn.metrics.pairwise import haversine_distances
from scipy.signal import savgol_filter, argrelextrema

def refine_raw_data(
        ride_data: pd.DataFrame,
        use_state_estimates: bool = True,
        correct_declination: bool = False,
    ):
    """
    Calculate additional information from the raw data.
    """
    if use_state_estimates:
        position_header = 'state_estimate'
    else:
        position_header = 'gps'

    assert position_header in ride_data.columns, f"Expected '{position_header}' column in ride_data"

    # Calculate the compass yaw from the magnetometer
    ride_data.loc[:, ('compass', 'yaw')] = WGS84_yaw_from_magnetometer(
        magnetometer_data=ride_data['magnetometer'],
        correct_declination=correct_declination,
        gps_data=ride_data[[(position_header, 'latitude'), (position_header, 'longitude')]],
    )

    # Segment the episode into plausible demonstrations of reaching some goal.
    ride_data = get_navigation_objective_segments(
        ride_data,
        position_header=position_header,
        plot_navigation_objectives=False,
    )

    return ride_data


def get_navigation_objective_segments(
        ride_data: pd.DataFrame,
        position_header: str = 'gps',
        plot_navigation_objectives: bool = False,
    ):
    """
    The Frodobots2k does not include information about which goal the robot is navigating to at any given time,
    and most of the time the robot navigates to some point and then returns to the starting point.

    This function segments the data into multiple navigation objectives. This is done by finding peaks
    in the relationship between distance traveled and Euclidean distance to starting point.

    If you want to visualize the segmentation result, set plot_navigation_objectives to True. For visualizing the
    actual navigation path and objectives on a map, I recommend looking at the Rerun .rrd's.
    """
    gps = np.radians(ride_data[[(position_header, 'latitude'), (position_header, 'longitude')]].dropna()) # Latitude and longitude in radians from GPS or state estimate
    gps_distances = haversine_distances(gps, gps)
    gps_distances *= 6371000 # Multiply by Earth's radius to get distance in meters

    # Euclidean distance to the starting point
    start_distances = gps_distances[:, 0]
    start_distances = pd.Series(start_distances, index=gps.index, name='distance_to_start')
    ride_data.loc[:, ('navigation_stats', 'distance_to_start')] = start_distances

    # Calculate the total distance traveled
    path_distances = np.diag(gps_distances, k=1)

    # Add a leading zero to the path distances to make the cumulative sum start at zero
    path_cumsum = np.cumsum(np.insert(path_distances, 0, 0))
    path_cumsum = pd.Series(path_cumsum, index=gps.index, name='path_distance')
    ride_data.loc[:, ('navigation_stats', 'path_distance')] = path_cumsum

    # Find segments of the ride where the robot is coherently navigating towards some 'goal point'.
    ride_data = compute_navigation_objectives_peaksegment(ride_data, gps_distances, position_header=position_header)
    
    if plot_navigation_objectives:
        plot_navigation_objectives(ride_data)

    # For each time step, limit the navigation objective to be at most certain distances away.
    ride_data = compute_intermediate_navigation_objectives_vectorized(ride_data, position_header=position_header)

    return ride_data

def compute_intermediate_navigation_objectives_vectorized(
    ride_data: pd.DataFrame,
    position_header: str = 'gps',
):
    """
    Use this if you don't want the navigation objective for any given time step to be further away than
    a certain distance.
    """
    distance_categories = [30., 150.]
    ride_data['navigation_stats'] = ride_data['navigation_stats'].ffill().bfill()

    iter_df = ride_data.dropna(subset=[('navigation_objective', 'timestamp'), ('navigation_stats', 'path_distance')])
    iter_df_sorted = iter_df.sort_index()

    objective_timestamps = iter_df_sorted[('navigation_objective', 'timestamp')].values
    path_distances = iter_df_sorted[('navigation_stats', 'path_distance')].values

    intermediate_objectives = {}
    for distance in distance_categories:
        intermediate_path_distances = path_distances + distance

        idx = np.searchsorted(path_distances, intermediate_path_distances, side='left')
        idx = np.minimum(idx, len(path_distances) - 1)

        # Ensure that each idx doesn't exceed the corresponding ('navigation_objective', 'timestamp')
        idx = iter_df_sorted.index[idx].values
        idx[idx > objective_timestamps] = objective_timestamps[idx > objective_timestamps]

        intermediate_objectives[f'navigation_objective_{distance}m', 'timestamp'] = idx
        intermediate_objectives[f'navigation_objective_{distance}m', 'lat'] = iter_df_sorted.loc[idx, (position_header, 'latitude')].values
        intermediate_objectives[f'navigation_objective_{distance}m', 'lon'] = iter_df_sorted.loc[idx, (position_header, 'longitude')].values

    intermediate_objectives_df = pd.DataFrame(
        intermediate_objectives,
        index=iter_df_sorted.index,
        columns=pd.MultiIndex.from_tuples(intermediate_objectives.keys())
    )

    # Merge the intermediate objectives back into the original ride_data
    for col in intermediate_objectives_df.columns:
        ride_data.loc[intermediate_objectives_df.index, col] = intermediate_objectives_df[col]
        # Fill NaN values        
        ride_data[col] = ride_data[col].ffill().bfill()

        assert all(ride_data[(col[0], 'timestamp')] <= ride_data[('navigation_objective', 'timestamp')])

    return ride_data

def compute_navigation_objectives_peaksegment(
        ride_data: pd.DataFrame,
        gps_distances: np.ndarray,
        filter_size: int = 51,
        filter_order: int = 2,
        inter_peak_min_distance: float = 20.0,
        position_header: str = 'gps',
    ):
    """
    Incrementally segment the navigation path into different navigation objectives based on the
    trend of distance to segment starting point.
    """
    gps = ride_data[[(position_header, 'latitude'), (position_header, 'longitude')]].dropna()
    stats = ride_data['navigation_stats'].dropna()
    peak_idx = 0
    peak_idxs = []
    while True:
        # Get distance of each path point to the previous peak
        dists = gps_distances[peak_idx:, peak_idx]
        if len(dists) < filter_size:
            break

        # Smooth the distances
        smoothed_dists = savgol_filter(dists, filter_size, filter_order)

        # Find the local maxima of the distance curve
        new_peaks = argrelextrema(
            smoothed_dists,
            np.greater,
            order=min(filter_size, len(smoothed_dists))
        )[0]
        if len(new_peaks) == 0:
            # No more peaks found, break the loop
            break

        # Get the next local maximum relative to the current peak
        new_peak = None
        previous_peak_path_len = stats.loc[gps.index[peak_idx]]['path_distance']
        for peak in new_peaks:
            new_peak_path_len = stats.loc[gps.index[peak+peak_idx]]['path_distance']
            # peak_dist = new_peak_path_len - previous_peak_path_len
            peak_dist = dists[peak]
            if peak_dist > inter_peak_min_distance:
                new_peak = peak + peak_idx
                break

        if new_peak is None:
            # No new peak found, break the loop
            break

        # If the new peak is within inter_peak_min_distance of the path end, break the loop
        dist_to_end = stats.iloc[-1]['path_distance'] - new_peak_path_len
        if dist_to_end < inter_peak_min_distance:
            break

        peak_idxs.append(new_peak)
        peak_idx = new_peak
        if new_peak >= len(gps_distances) - 1:
            break
    
    # Add the last point as a peak
    if len(peak_idxs) == 0 or peak_idxs[-1] != len(gps_distances) - 1:
        peak_idxs.append(len(gps_distances) - 1)


    # Get the navigation objectives
    objectives = []
    for peak in peak_idxs:
        # Get the timestamp for the peak
        timestamp = gps.index[peak]

        # Get the latitude and longitude for the peak
        lat = gps.iloc[peak][(position_header, 'latitude')]
        lon = gps.iloc[peak][(position_header, 'longitude')]
        objectives.append((timestamp, lat, lon))
    
    # Create a DataFrame for the navigation objectives
    objectives_df = pd.DataFrame(objectives, columns=['timestamp', 'lat', 'lon'])
    objectives_df['timestamp'] = pd.to_datetime(objectives_df['timestamp'])

    # For each row in ride_data, set the navigation objective to the 'next' objective as measured by the timestamp
    start_time = ride_data.index[0]
    for i, row in objectives_df.iterrows():
        end_time = row['timestamp']
        mask = (ride_data.index >= start_time) & (ride_data.index < end_time)
        ride_data.loc[mask, ('navigation_objective', 'timestamp')] = row['timestamp']
        ride_data.loc[mask, ('navigation_objective', 'lat')] = row['lat']
        ride_data.loc[mask, ('navigation_objective', 'lon')] = row['lon']
        start_time = end_time

    # Fill NaN's
    ride_data['navigation_objective'] = ride_data['navigation_objective'].ffill().bfill()

    return ride_data

def plot_navigation_objectives(ride_data: pd.DataFrame):
    """
    Heloper to plot the navigation objectives on the ride data.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    stats = ride_data['navigation_stats'].dropna()
    ax.plot(
        stats['path_distance'],
        stats['distance_to_start'],
        label='Path distance')

    # Plot the navigation objectives as vertical lines
    objectives = ride_data['navigation_objective'].dropna()
    # Find unique
    unique_objectives = objectives.drop_duplicates()
    for _, row in unique_objectives.iterrows():
        path_dist = stats.loc[row['timestamp']]['path_distance']
        ax.axvline(x=path_dist, color='r', linestyle='--', label='Navigation objective change')
    plt.savefig('navigation_objectives.png')
    
def WGS84_yaw_from_magnetometer(
    magnetometer_data: pd.DataFrame,
    gps_data: pd.DataFrame = None,
    correct_declination: bool = False,
):
    """
    See https://colab.research.google.com/#scrollTo=b7beaec6-5be1-4c16-9769-345f29f76d63&fileId=https%3A//huggingface.co/datasets/frodobots/FrodoBots-2K/blob/main/helpercode.ipynb
    for description of the magnetometer data.

    Note: On some of the robots in the dataset, the magnetometer seems to be installed in a random orientation,
    meaning that the yaw calculated here is not necessarily correct. In the future, could try to implement an automatic heuristic
    to try and detect magnetometer orientation, but for now this is left as is.

    Args:
    - x, y, z: Magnetometer data in raw LSB.
        In the colab spec, they say that coordinates are:
        
        Positive x = Vertical Up
        Positive y = West
        Positive z = South

        However, looking at the data, they seem more like:
        Positive x = Vertical Down
        Positive y = East
        Positive z = South

        What a weird coordinate system. Let's assume the latter for now.

    Returns:
    - WGS84 Yaw (or compass direction) in radians. 0 is North, pi/2 is East, +-pi is South, -pi/2 is West
    """
    if magnetometer_data.isnull().all().all():
        return pd.Series(index=magnetometer_data.index, name='yaw')

    # Switch from DES to NED coordinate system.
    x, y, z = -magnetometer_data['z'], magnetometer_data['y'], -magnetometer_data['x']

    # LSB Raw data to Gauss
    x /= 3000
    y /= 3000

    # Calculate the yaw
    yaw = np.arctan2(y, x)

    if correct_declination:
        first_gps = gps_data.loc[gps_data.first_valid_index()]
        declination = get_magnetic_declination(
            lat=first_gps['latitude'],
            lon=first_gps['longitude'],
            date=gps_data.first_valid_index(),
        )
        yaw += declination
        
    # Turn the yaw into a pandas series with the same index as the magnetometer data
    yaw = pd.Series(yaw, index=magnetometer_data.index, name='yaw')
    return yaw

def get_magnetic_declination(
    lat: float,
    lon: float,
    date: datetime,
    model: str ='IGRF',
):
    """
    Get the magnetic declination at a specific location and time from the NOAA API.
    Currently not in use, also not tested properly.

    Returns:
    - The magnetic declination in radians, East of True North.
    """

    print(f"Fetching magnetic declination for lat: {lat}, lon: {lon}, date: {date}, model: {model}")

    base_url = "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination"
    result_format = "json"
    key = ""  # API access key. Replace with your actual key if you plan to use this function.
    
    params = {
        "lat1": lat,
        "lon1": lon,
        "startYear": date.year,
        "startMonth": date.month,
        "startDay": date.day,
        "key": key,
        "resultFormat": result_format
    }

    if model is not None:
        params["model"] = model
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        if result_format == "json":
            response = response.json()
            degrees = response['result'][0]['declination']
            return degrees / 180 * np.pi  # Convert to radians
        return response.text  # For XML, CSV, or HTML formats
    else:
        raise Exception(f"Error fetching magnetic declination from NOAA API: {response.status_code}, {response.text}")