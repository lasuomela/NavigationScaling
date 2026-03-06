"""
A PyTorch dataset for loading trajectories saved as HuggingFace datasets.
"""
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torchvision.transforms.v2 as T
import datasets
import einops
from pathlib import Path
import tqdm
import os

datasets.disable_caching()

from earthrovers.common.utils import compute_distance, compute_direction

class FrodoBotsDataset(torch.utils.data.Dataset):
    """
    Present a collection of trajectories from FrodoBots saved
    as HuggingFace datasets as a PyTorch dataset.
    """

    def __init__(
            self,
            dataset_path: Path,
            mode: str = 'train',
            sequence_transform: T.Compose = None,
            individual_transform: T.Compose = None,
            sequence_length: int = 1,
            samples_to_load: int = -1,
            hflip_augmentation: bool = False,
            random_goal: bool = False,
            max_goal_distance: float = 100.0,
            goal_sampling_distribution: str = 'beta', # 'beta' or 'uniform'
            persist_index: bool = False,
            num_locations: int = -1,  # -1 means no limit
            max_rides_per_location: int = -1,  # -1 means no limit
            min_rides_per_location: int = -1,  # -1 means no limit
            max_hours_per_location: int = -1,  # -1 means no limit
            min_hours_per_location: int = -1,  # -1 means no limit
            pick_cluster_ids: List[int] = [-1], # If specified, only use these cluster ids. Overrides other location selection criteria.
    ):
        """
        Initialize the dataset.
        """
        self._mode = mode
        self._sequence_transform = sequence_transform
        self._individual_transform = individual_transform
        self._samples_to_load = samples_to_load
        self._dataset_path = dataset_path
        self._sequence_length = sequence_length
        self._hflip_augmentation = hflip_augmentation
        self._random_goal = random_goal
        self._max_goal_distance = max_goal_distance
        self._goal_sampling_distribution = goal_sampling_distribution
        self._persist_index = persist_index

        self._index_path = self._get_index_file_path(
            dataset_path,
            mode=mode,
            persist_index=persist_index,
            sequence_length=sequence_length,
            num_locations=num_locations,
            max_rides_per_location=max_rides_per_location,
            min_rides_per_location=min_rides_per_location,
            max_hours_per_location=max_hours_per_location,
            min_hours_per_location=min_hours_per_location,
            pick_cluster_ids=pick_cluster_ids,
        )

        self._is_rank_0=(
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        if max_goal_distance > 0:
            if max_goal_distance < 10.0 or max_goal_distance > 150.0 or max_goal_distance % 10.0 != 0:
                raise ValueError(
                    """
                    max_goal_distance must be a multiple of 10.0 within [10.0, 150.0] (inclusive) or -1 for no limit.
                    Other values require modifying the precomputed goal distances in the dataset. See data_wrangling/data_refinement.py for details.
                    """
                )
            self.navigation_objective_label = f"navigation_objective_{max_goal_distance:.1f}m"
        else:
            self.navigation_objective_label = "navigation_objective"

        columns_to_return = [
            "front_camera",
            "('control', 'linear')",
            "('control', 'angular')",
            "('gps', 'latitude')",
            "('gps', 'longitude')",
            f"('{self.navigation_objective_label}', 'lat')",
            f"('{self.navigation_objective_label}', 'lon')",
            f"('{self.navigation_objective_label}', 'rows_ahead')",
            "('compass', 'yaw')",
            "('ride_id', '')",
            "('ride_location_cluster', '')",
            "('waypoints', 'x')",
            "('waypoints', 'y')",
            "('waypoints', 'yaw')",
            "('waypoints', 'timestamps')",
        ]
        
        self._load_dataset(
            dataset_path,
            columns_to_return=columns_to_return,
        )

        # Store as numpy arrays to mitigate python copy-on-access memory usage
        self._camera_keys = np.array([key for key in columns_to_return if "camera" in key]).astype(np.bytes_)
        self._columns_to_return = np.array(columns_to_return).astype(np.bytes_)

    def __del__(self):
        """
        Clean up the index.
        """
        if not self._persist_index:
            if self._index_path.exists():
                self._index_path.unlink(missing_ok=True)

    def __len__(self) -> int:
        """
        Get the number of trajectories in the dataset.
        """
        return len(self.index)

    def get_camera_keys(self) -> List[str]:
        """
        Get the camera keys in the dataset.
        """
        return [key.decode('utf-8') for key in self._camera_keys]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a navigation sequence from the dataset.
        """
        seq_start_idx = self.index["index"][idx].as_py()
        seq_slice = slice(seq_start_idx, seq_start_idx + self._sequence_length)
        seq = self._trajectories[seq_slice]
        
        seq = self._compute_goal(seq, seq_start_idx)

        # Stack the control inputs into a single tensor
        seq['target'] = torch.stack([seq["('control', 'linear')"], seq["('control', 'angular')"],], dim=-1)
        del seq["('control', 'linear')"]
        del seq["('control', 'angular')"]

        # Rename the ride_id column to a more convenient name
        seq['ride_id'] = seq["('ride_id', '')"]
        del seq["('ride_id', '')"]

        # Apply horizontal flip augmentation as functional transform because
        # the goal direction and target angular velocity need to be flipped as well
        seq['flipped'] = torch.zeros(1, dtype=torch.bool)
        if self._hflip_augmentation:
            if np.random.rand() > 0.5:
                for camera in self.get_camera_keys():
                    seq[camera] = T.functional.hflip(seq[camera])

                seq['target'][..., 1] *= -1
                seq['goal_input'][..., 1] *= -1
                seq["('waypoints', 'y')"] *= -1
                seq["('waypoints', 'yaw')"] *= -1

                seq['flipped'] = torch.ones(1, dtype=torch.bool)

        # Stack the waypoints into a single tensor
        seq['waypoints'] = torch.stack([
            seq["('waypoints', 'x')"],
            seq["('waypoints', 'y')"],
            torch.cos(seq["('waypoints', 'yaw')"]), # Use cos, sin parameterization for yaw
            torch.sin(seq["('waypoints', 'yaw')"]),
        ], dim=-1)
        del seq["('waypoints', 'x')"]
        del seq["('waypoints', 'y')"]
        del seq["('waypoints', 'yaw')"]

        # Apply image transformations
        for camera in self._camera_keys:
            camera = camera.decode('utf-8')

            if self._individual_transform is not None:
                # Apply a transform with individual randomization to each image
                seq[camera] = torch.stack([self._individual_transform(img) for img in seq[camera]], axis=0)

            if self._sequence_transform is not None:
                seq[camera] = self._sequence_transform(seq[camera])

        return seq

    def _compute_goal(
            self,
            seq: Dict[str, torch.Tensor],
            seq_start_idx: int,
    ):
        """
        Compute the distance and direction to the goal from the current position.
        """
        # Compute the distance to the goal
        current_lat = seq["('gps', 'latitude')"]
        current_lon = seq["('gps', 'longitude')"]
        rows_ahead = seq[f"('{self.navigation_objective_label}', 'rows_ahead')"][-1]

        # Get a random goal idx in the interval [0, rows_ahead]
        # from the last sample in the sequence
        if self._random_goal:
            if rows_ahead > 0:
                if self._goal_sampling_distribution == 'beta':
                    # Pick the goal idx from a beta distribution to avoid
                    # the goal distribution being skewed too much towards close by goals
                    rows_ahead = int(round(
                        np.random.beta(5, 1, size=None) * rows_ahead.item()
                    ))
                elif self._goal_sampling_distribution == 'uniform':
                    # Pick the goal idx uniformly from the interval [0, rows_ahead]
                    rows_ahead = int(round(
                        np.random.uniform(0, 1, size=None) * rows_ahead.item()
                    ))
                else:
                    raise ValueError(
                        f"Unknown goal sampling distribution {self._goal_sampling_distribution}. "
                        "Supported distributions are 'beta' and 'uniform'."
                    )

        goal_idx = seq_start_idx + self._sequence_length - 1 + rows_ahead
        if self._samples_to_load > 0:
            goal_idx = min(goal_idx, len(self._trajectories) - 1)

        navigation_objective = self._locations[
            [goal_idx]
        ]

        # Expand the objective lat/lon to match the current lat/lon
        for obj_key in ["('gps', 'latitude')", "('gps', 'longitude')"]:
            navigation_objective[obj_key] = einops.repeat(
                navigation_objective[obj_key], "1 -> s", s=current_lat.shape[0]
            )

        goal_lat = navigation_objective["('gps', 'latitude')"]
        goal_lon = navigation_objective["('gps', 'longitude')"]

        distance = compute_distance(current_lat, current_lon, goal_lat, goal_lon)

        # Compute the direction to the goal
        current_yaw = seq["('compass', 'yaw')"]
        direction = compute_direction(current_lat, current_lon, goal_lat, goal_lon, current_yaw)

        # Set the waypoints beyond the navigation objective to the last valid waypoint
        wps_past_goal = np.array(seq["('waypoints', 'timestamps')"]) > np.array(navigation_objective["timestamp"])
        if np.any(wps_past_goal):
            idx = np.where(~wps_past_goal, np.arange(wps_past_goal.shape[1]),0)
            np.maximum.accumulate(idx,axis=1, out=idx)

            rows = np.arange(idx.shape[0])[:, None]
            seq["('waypoints', 'x')"] = seq["('waypoints', 'x')"][rows, idx]
            seq["('waypoints', 'y')"] = seq["('waypoints', 'y')"][rows, idx]
            seq["('waypoints', 'yaw')"] = seq["('waypoints', 'yaw')"][rows, idx]


        del seq["('waypoints', 'timestamps')"] # Not needed anymore
        seq['goal_input'] = torch.stack([distance, direction], dim=-1)
        seq['current_position'] = torch.stack([current_lat, current_lon], dim=-1)
        seq['current_yaw'] = current_yaw
        seq['goal_position'] = torch.stack([goal_lat, goal_lon], dim=-1)
        return seq
         
    @staticmethod
    def _load_trajectories(
            dataset_path: Path,
            mode: str = 'train',
            samples_to_load: int = -1,
            progress: bool = True,
    ) -> datasets.Dataset:
        
        if mode == 'train_with_val':
            mode = 'val'

        trajectory_paths = list(dataset_path.glob(f"{mode}/*.npz"))
        if len(trajectory_paths) == 0:
            raise FileNotFoundError(f"No dataset files found in {dataset_path}")

        if samples_to_load > 0:
            # Only load the first dataset chunk
            trajectory_paths = trajectory_paths[:1]

        if progress:
            pbar = tqdm.tqdm(
                total=len(trajectory_paths),
                desc=f"Memory-mapping {mode} dataset files",
            )

        trajectory_shards = []
        for path in trajectory_paths:
            if progress:
                pbar.update(1)
            shard = datasets.load_from_disk(path)
            trajectory_shards.append( shard )

        trajectories = datasets.concatenate_datasets(trajectory_shards)

        # Trick to reduce memory footprint in pytorch Dataloder
        trajectories._data = datasets.table.MemoryMappedTable(trajectories.data.table, '', [])
        
        if samples_to_load > 0:
            trajectories = trajectories.select(range(samples_to_load))

        return trajectories


    def _load_dataset(
            self,
            dataset_path: Path,
            columns_to_return: List[str] = None,
        ) -> datasets.Dataset:
        """
        Memory map the trajectories from disk.
        """
        trajectories = self._load_trajectories(
            dataset_path,
            mode=self._mode,
            samples_to_load=self._samples_to_load,
            progress=self._is_rank_0,
        )
        tensor_columns = trajectories.column_names
        [tensor_columns.remove(col) for col in ['timestamp', "('waypoints', 'timestamps')"]]
        trajectories.set_format(
            type='torch',
            columns=tensor_columns,
            output_all_columns=True,
        )

        self._locations = trajectories.select_columns([
            "('gps', 'latitude')",
            "('gps', 'longitude')",
            "timestamp",
        ])

        trajectories = trajectories.select_columns(columns_to_return)
        # Make sure any images are decoded
        for name, type in trajectories.features.items():
            if isinstance(type, datasets.Image):
                trajectories = trajectories.cast_column(name, datasets.Image(decode=True))

        self._trajectories = trajectories
        self.load_index()

    
    def load_index(self):
        """
        Memory map the index from disk.
        """
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"Index file {self._index_path} does not exist. "
                "Please build the index first using `FrodoBotsDataset.build_index()`."
            )

        with pa.memory_map(str(self._index_path), "r") as index_mmap:
            reader = ipc.open_file(index_mmap)
            self.index = reader.read_all()
        
    @staticmethod
    def build_index(
            dataset_path: Path,
            sequence_length: int,
            mode: str = 'train',
            persist_index: bool = False,
            num_locations: int = -1,  # -1 means no limit
            max_rides_per_location: int = -1,  # -1 means no limit
            min_rides_per_location: int = -1,  # -1 means no limit
            max_hours_per_location: int = -1,  # -1 means no limit
            min_hours_per_location: int = -1,  # -1 means no limit
            samples_to_load: int = -1, # For debugging purposes, load only a subset of the dataset
            pick_cluster_ids: List[int] = [-1], # If specified, only use these cluster ids. Overrides other location selection criteria.
        ):
        """
        Build an index to fetch chunks of the trajectories with the specified length,
        while ensuring that the samples within a chunk come from a single trajectory.
        """
        index_path = FrodoBotsDataset._get_index_file_path(
            dataset_path,
            mode=mode,
            persist_index=persist_index,
            sequence_length=sequence_length,
            num_locations=num_locations,
            max_rides_per_location=max_rides_per_location,
            min_rides_per_location=min_rides_per_location,
            max_hours_per_location=max_hours_per_location,
            min_hours_per_location=min_hours_per_location,
            pick_cluster_ids=pick_cluster_ids,
        )

        if index_path.exists():
            print(f"Index file {index_path} already exists. Skipping index creation.")
            return

        trajectories = FrodoBotsDataset._load_trajectories(
            dataset_path,
            mode=mode,
            samples_to_load=samples_to_load,
            progress=True,
        )
        # Build the index and write it to disk
        print("Calculating index...")
        FrodoBotsDataset._build_index(
            trajectories,
            index_path,
            mode,
            sequence_length,
            num_locations=num_locations,
            max_rides_per_location=max_rides_per_location,
            min_rides_per_location=min_rides_per_location,
            max_hours_per_location=max_hours_per_location,
            min_hours_per_location=min_hours_per_location,
            pick_cluster_ids=pick_cluster_ids,
        )

    @staticmethod
    def _get_index_file_path(
            dataset_path: Path,
            mode: str = 'train',
            persist_index: bool = False,
            sequence_length: int = None,
            num_locations: int = None,
            max_rides_per_location: int = None,
            min_rides_per_location: int = None,
            max_hours_per_location: int = None,
            min_hours_per_location: int = None,
            pick_cluster_ids: List[int] = [-1],
    ):
        if not persist_index:
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            index_path = Path(dataset_path) / f"{job_id}_{mode}.index.arrow"
        else:
            assert sequence_length is not None, "sequence_length must be specified when persist_index is True"
            assert max_rides_per_location is not None, "max_rides_per_cluster must be specified when persist_index is True"
            assert min_rides_per_location is not None, "min_rides_per_cluster must be specified when persist_index is True"
            assert (num_locations is not None) or (pick_cluster_ids[0] > -1), "num_locations or pick_cluster_ids must be specified when persist_index is True"
            
            index_filename = f"{mode}_seq{sequence_length}_"
            if (pick_cluster_ids[0] > -1):
                cluster_ids_str = "_".join([str(cid) for cid in pick_cluster_ids])
                index_filename += f"clusterIds{cluster_ids_str}"
            else:
                index_filename += f"numLocations{num_locations}"

            index_filename += f"_maxRides{max_rides_per_location}_minRides{min_rides_per_location}_maxLocHours{max_hours_per_location}_minLocHours{min_hours_per_location}.index.arrow"
            index_path = Path(dataset_path) / index_filename
        return index_path

    @staticmethod
    def _build_index(
            trajectories: datasets.Dataset,
            index_path: Path,
            mode: str,
            sequence_length: int,
            num_locations: int = -1,
            max_rides_per_location = -1,
            min_rides_per_location = -1,
            max_hours_per_location: int = -1,
            min_hours_per_location: int = -1,
            sampling_rate: int = 4, # Hz
            pick_cluster_ids: List[int] = [-1], # If specified, only use these cluster ids. Overrides other location selection criteria.
        ):
        """
        Build an index to fetch observation sequences with the specified length,
        while ensuring that the samples within a sequence come from a single trajectory.
        """

        assert not (
            (max_rides_per_location > 0 or min_rides_per_location > 0) \
            and (max_hours_per_location > 0 or min_hours_per_location > 0)
        ), "Specifying both max/min rides and max/min hours per location not implemented. Please choose one of the two."
            

        # Extract ride_id column
        ride_ds = trajectories.select_columns(
            [
                "('ride_id', '')",
                "('ride_location_cluster', '')",
                "('navigation_objective', 'lat')",
                "('navigation_objective', 'lon')",
            ])
        ride_ds.set_format(type='numpy')

        # Load the ride data to memory
        ride_infos = ride_ds[:]

        matches = np.ones(
            (len(ride_ds) - sequence_length + 1),
            dtype=bool,
        )

        keys_to_match = [
            "('ride_id', '')",
            "('navigation_objective', 'lat')",
            "('navigation_objective', 'lon')",
        ]
        for key in keys_to_match:
            # Get start and end segments
            seq_start = ride_infos[key][:len(ride_ds) - sequence_length + 1]
            seq_end = ride_infos[key][sequence_length - 1:]

            # Vectorized comparison
            matches *= seq_start == seq_end

        ### For each cluster, pick at most N different rides
        if (max_rides_per_location > 0) or (min_rides_per_location > 1) or (max_hours_per_location > 0) or (min_hours_per_location > 0) or (num_locations > 0) or (pick_cluster_ids[0] > -1):
            
            # Find unique clusters
            clusters = np.unique(ride_infos["('ride_location_cluster', '')"])

            # Find the rides in each cluster
            cluster_rides = {}
            cluster_framecounts = {}
            for cluster in clusters:
                ride_ids = ride_infos["('ride_id', '')"][
                    ride_infos["('ride_location_cluster', '')"] == cluster
                ]
                cluster_framecounts[cluster] = len(ride_ids)
                ride_ids = np.unique(ride_ids)
                cluster_rides[cluster] = ride_ids

            stats_path = index_path.parent / f"{mode}_rides_per_location.csv"
            with open(stats_path, 'w') as f:
                f.write("cluster,rides,frames\n")
                for cluster, ride_ids in cluster_rides.items():
                    f.write(f"{cluster},{len(ride_ids)},{cluster_framecounts[cluster]}\n")

            # Sort the clusters by descending frame count to ensure we have the most populated clusters first
            sorted_clusters = sorted(
                cluster_framecounts.keys(),
                key=lambda c: cluster_framecounts[c],
                reverse=True
            )

            if not (len(pick_cluster_ids) == 1 and pick_cluster_ids[0] == -1):
                # Only use the specified cluster ids
                sorted_clusters = [c for c in sorted_clusters if c in pick_cluster_ids]
                if len(sorted_clusters) < len(pick_cluster_ids):
                    missing_clusters = set(pick_cluster_ids) - set(sorted_clusters)
                    raise ValueError(
                        f"Some of the specified cluster ids were not found in the {mode} dataset: {missing_clusters}. Available clusters: {clusters}"
                    )

            # Pick the subset of clusters that fit the criteria
            # (max/min rides per cluster, max/min hours per cluster, num clusters)
            valid_clusters = []
            for cluster in sorted_clusters:
                ride_ids = cluster_rides[cluster]
                if min_rides_per_location > 0 and len(ride_ids) < min_rides_per_location:
                    # Skip this location if it has less than the minimum number of rides
                    continue
                elif min_hours_per_location > 0 and cluster_framecounts[cluster] < min_hours_per_location * 3600 * sampling_rate:
                    # Skip this location if it has less than the minimum number of frames
                    continue
                else:
                    valid_clusters.append(cluster)

            # Find the rides that fit the criteria
            cluster_selected_rides = {}
            for cluster in valid_clusters:
                ride_ids = cluster_rides[cluster]

                if num_locations > 0 and len(cluster_selected_rides) >= num_locations:
                    # Stop if we have enough locations
                    break

                if (max_rides_per_location > 0) and (len(ride_ids) > max_rides_per_location):
                    # Select at most max_rides_per_location from the cluster rides
                    cluster_selected_rides[cluster] = ride_ids[:max_rides_per_location]
                else:
                    # Use all rides in the cluster
                    cluster_selected_rides[cluster] = ride_ids

            if (len(cluster_selected_rides) == 0) or (num_locations > 0 and len(cluster_selected_rides) < num_locations):
                raise ValueError(
                    f"Not enough locations found with the given criteria: "
                    f"num_locations={num_locations}, "
                    f"max_rides_per_location={max_rides_per_location}, min_rides_per_location={min_rides_per_location},"
                    f"max_hours_per_location={max_hours_per_location}, min_hours_per_location={min_hours_per_location}. "
                    f"Found {len(cluster_selected_rides)} locations."
                )
            
            all_selected_rides = np.concatenate(list(cluster_selected_rides.values()))

            # Create a boolean mask indicating which indices are valid
            ride_ids = ride_infos["('ride_id', '')"][:len(matches)]
            valid_indices = np.isin(ride_ids, all_selected_rides)

            # Limit the amount of frames per location by frame counts / duration
            if max_hours_per_location > 0:
                max_frames_per_location = int(max_hours_per_location * 3600 * sampling_rate)
                # Limit the number of frames per location
                for cluster in cluster_selected_rides.keys():
                    cluster_frames = ride_infos["('ride_location_cluster', '')"][:len(matches)] == cluster
                    cluster_indices = np.where(cluster_frames & valid_indices)[0]
                    if len(cluster_indices) > max_frames_per_location:
                        # Select the max_frames_per_location first indices
                        valid_indices[cluster_indices[max_frames_per_location:]] = False
                
            # Apply the valid indices mask
            matches *= valid_indices

            # Count number of frames
            locations, framecounts = np.unique(
                ride_infos["('ride_location_cluster', '')"][:len(matches)][matches],
                return_counts=True,
            )
            location_framecounts = dict(zip(locations, framecounts))

            # Save the sampling stats
            stats_path = index_path.with_suffix('.stats.csv')
            with open(stats_path, 'w') as f:
                f.write("cluster,rides,frames\n")
                for cluster, rides in cluster_selected_rides.items():
                    f.write(f"{cluster},{len(rides)},{location_framecounts[cluster]}\n")

        # Get the indices of the sequence start observations
        indices = matches.nonzero()[0]

        # Write the indices to a file
        index = pa.table({"index": pa.array(indices, type=pa.int32())})
        with ipc.new_file(str(index_path), index.schema) as writer:
            writer.write_table(index)
        
def plot_sample(batch: Dict[str, torch.Tensor]):
    """
    Plot a sample from the dataset.
    """
    from earthrovers.train.train_utils.visualization import plot_obs_and_controls

    outputs = {'pred': batch["target"].clone()}  # Mock outputs for visualization
    # Add a bit of randomness to the outputs
    outputs['pred'] += torch.randn_like(outputs['pred'])

    viz_idxs= -1
    ride_ids = [batch["ride_id"][0]]
    flipped = batch["flipped"][viz_idxs]
    viz_images = batch["front_camera"][viz_idxs].permute(1,2,0)
    viz_goal_inputs = batch["goal_input"][viz_idxs]
    viz_targets = batch["target"][viz_idxs]
    viz_outputs = outputs['pred'][viz_idxs]
    viz_current_positions = batch["current_position"][viz_idxs]
    viz_current_yaws = batch["current_yaw"][viz_idxs]
    viz_goal_positions = batch["goal_position"][viz_idxs]
    viz_gt_waypoints = batch["waypoints"][viz_idxs]
    viz_pred_waypoints = batch["waypoints"][viz_idxs].clone()  # Assuming waypoints are the same as targets for visualization
    viz_pred_waypoints += torch.randn_like(viz_pred_waypoints)  # Add some noise for visualization

    figs = [
        plot_obs_and_controls(
            ride_ids,
            viz_images,
            viz_goal_inputs,
            viz_targets,
            viz_outputs,
            viz_current_positions,
            viz_current_yaws,
            viz_goal_positions,
            flipped,
            gt_waypoints=viz_gt_waypoints,
            pred_waypoints=viz_pred_waypoints,
        )
    ]
    # Save the figure
    for i, img in enumerate(figs):
        img.image.save(f'sample_plot_{i}.png')



if __name__ == "__main__":

    import earthrovers
    pkg_top_dir = Path(earthrovers.__file__).parent.parent

    path = pkg_top_dir / "dataset"
    persist_index = False
    seq_len = 1
    max_rides_per_location=-1
    min_rides_per_location=-1
    max_hours_per_location=0.5
    min_hours_per_location=1.0
    num_locations=4

    # Build the index for the dataset
    FrodoBotsDataset.build_index(
        path,
        sequence_length=seq_len,
        mode='train',
        persist_index=persist_index,
        max_rides_per_location=max_rides_per_location,
        min_rides_per_location=min_rides_per_location,
        max_hours_per_location=max_hours_per_location,
        min_hours_per_location=min_hours_per_location,
        num_locations= num_locations,
    )
    ds = FrodoBotsDataset(
        path,
        sequence_length=seq_len,
        max_goal_distance=30.0,
        hflip_augmentation=False,
        random_goal=True,
        persist_index=persist_index,
        num_locations=num_locations,
        max_rides_per_location=max_rides_per_location,
        min_rides_per_location=min_rides_per_location,
        max_hours_per_location=max_hours_per_location,
        min_hours_per_location=min_hours_per_location,
    )
    
    sample = ds[0]
    plot_sample(sample)