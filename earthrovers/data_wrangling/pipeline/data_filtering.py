"""
This module contains functions to filter the raw ride data from the FrodoBots dataset.
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from earthrovers.data_wrangling.pipeline.state_estimation import optimize_trajectory

def filter_raw_data(
        ride_data: pd.DataFrame,
        ride_id: str,
        video_sampling_frequency: float = 4.0,
        remove_stationary_images: bool = True,
        optimize_trajectory_poses: bool = True,
        fuse_magnetometer: bool = False,
        verbose: bool = False,
) -> List[pd.DataFrame]:
    """
    Apply different filters to the raw ride data.

    Returns a list of episodes after filtering - a single ride
    may be split into multiple episodes if there are gaps in the GPS data.
    """
    # Trim the data to segment that has video frames
    ride_data = trim_to_video_duration(ride_data)
    if ride_data is None:
        if verbose:
            print(f'No video timestamps found, skipping {ride_id}')
        return []
 
    # Subsample the video frames to video_sampling_frequency Hz
    ride_data = subsample_video_frames(ride_data, video_sampling_frequency=video_sampling_frequency)
    if ride_data is None:
        if verbose:
            print(f'No video timestamps found after subsampling, skipping {ride_id}')
        return []

    # Split the data to multiple episodes if there are gaps in the GPS data
    episodes = remove_sensor_gaps(
        ride_data,
        'gps',
        ride_id=ride_id,
        gap_duration_threshold=4.0 if optimize_trajectory_poses else 2.0,
        mode='split',
        verbose=verbose,
    )

    # Process each episode
    for i in range(len(episodes)-1, -1, -1):
        ride_data = episodes[i]

        if ride_data.empty:
            if verbose:
                print(f'Empty episode, skipping {ride_id} episode {i}')
            episodes.pop(i)
            continue

        if not ride_data[('gps', 'latitude')].notnull().any():
            if verbose:
                print(f'No GPS data found, skipping {ride_id} episode {i}')
            episodes.pop(i)
            continue

        # Trim the data to the segment that has GPS data
        ride_data = trim_to_gps_duration(ride_data)

        # Remove gaps in the magnetometer data
        ride_data = remove_sensor_gaps(
            ride_data,
            'magnetometer',
            ride_id=ride_id,
            gap_duration_threshold=video_sampling_frequency,
            mode='trim',
        )[0]

        # Trim the stationary segments before and after the navigation
        ride_data = trim_pre_post_nav_stationary(ride_data)            

        # Check if the episode duration is too short
        if episode_duration_too_short(ride_data, threshold=20.0):
            if verbose:
                print(f'Episode is too short, skipping {ride_id} episode {i}')
            episodes.pop(i)
            continue

        # Check if the episode is stationary
        if episode_is_stationary(ride_data):
            if verbose:
                print(f'Episode is stationary, skipping {ride_id} episode {i}')
            episodes.pop(i)
            continue

        # Check if the episode has enough movement
        if episode_displacement_too_small(ride_data, threshold=15.0):
            if verbose:
                print(f'Episode has too small displacement, skipping {ride_id} episode {i}')
            episodes.pop(i)
            continue

        if not episode_has_magnetometer_measurements(ride_data):
            if verbose:
                print(f'No magnetometer data found, skipping {ride_id} episode {i}')
            episodes.pop(i)
            continue

        if not ride_data[('front_camera', 'frame_id')].notnull().any():
            if verbose:
                print(f'No front camera data found, skipping {ride_id} episode {i}')
            episodes.pop(i)
            continue

        # Perform sensor fusion to get more accurate pose data
        if optimize_trajectory_poses:
            ride_data = optimize_trajectory(ride_data, fuse_magnetometer=fuse_magnetometer)

        # Remove stationary segments in the middle of the navigation
        if remove_stationary_images:
            ride_data = remove_stationary_segment_images(ride_data)

        episodes[i] = ride_data

    return episodes

def subsample_video_frames(
        ride_data: pd.DataFrame,
        video_sampling_frequency: float = 4.0, # Hz
) -> pd.DataFrame:
    """
    Subsample the video frames to the given frequency.
    """
    for camera_type in ['front_camera', 'rear_camera']:
        if camera_type not in ride_data.columns:
            continue

        # Original video fps is ~20Hz. Pick images at sampling_frequency Hz.
        image_indices = ride_data[camera_type]['frame_id'].dropna().index
        image_indices_subsampled = []
        for idx in image_indices:
            if image_indices_subsampled:
                if (idx - image_indices_subsampled[-1]).total_seconds() >= 1/video_sampling_frequency:
                    image_indices_subsampled.append(idx)
            else:
                image_indices_subsampled.append(idx)

        # Keep only the subsampled images
        subsampled_images = ride_data.loc[image_indices_subsampled, camera_type]
        ride_data.loc[:, camera_type] = np.nan
        ride_data.loc[subsampled_images.index, camera_type] = subsampled_images.values

        # Drop any rows where all fields are NaN
        ride_data = ride_data.dropna(how='all')

        if ride_data[camera_type].isnull().all().all():
            return None

    return ride_data


def episode_has_magnetometer_measurements(ride_data: pd.DataFrame) -> bool:
    """
    Check if the episode has magnetometer data.
    """
    if 'magnetometer' not in ride_data.columns:
        return False
    return ride_data['magnetometer'].notnull().any().any()

def episode_is_stationary(ride_data: pd.DataFrame, threshold: float = 0.9) -> bool:
    """
    Check if the episode is stationary based on the wheel RPM's.

    Args:
    - ride_data: The ride data for a single episode.
    - threshold: The threshold for the percentage of zero RPM's to consider the episode stationary.
    """
    # Pick the valid entries from the ride data
    rpms = ride_data['control']['rpm_1'].dropna()

    if rpms.empty:
        return True
    
    # If more than threshold % of the RPM's are zero, the episode is stationary
    if len(rpms[rpms == 0]) / len(rpms) > threshold:
        return True
    return False

def episode_duration_too_short(ride_data: pd.DataFrame, threshold: float = 20.0) -> bool:
    """
    Check if the episode duration is too short.

    Args:
    - ride_data: The ride data for a single episode.
    - threshold: Minimum episode duration in seconds.
    """
    if ride_data.empty:
        return True

    duration = ride_data.index[-1] - ride_data.index[0]
    if duration.total_seconds() < threshold:
        return True
    return False

def episode_displacement_too_small(ride_data: pd.DataFrame, threshold: float = 15.0) -> bool:
    """
    Check if the robot has moved enough during the episode.

    Args:
    - ride_data: The ride data for a single episode.
    - threshold: Minimum displacement from the starting location in meters.
    """
    if 'gps' not in ride_data.columns:
        raise ValueError('No GPS data found in the ride data')
    
    if 'state_estimate' in ride_data.columns:
        positions = ride_data['state_estimate'].dropna()
    else:
        positions = ride_data['gps'].dropna()

    # Convert the position data to radians
    positions = np.radians(positions[['latitude', 'longitude']])

    # Calculate the displacement from the starting location
    displacement = haversine_distances(positions, positions.iloc[[0]])
    displacement *= 6371000  # multiply by Earth radius to get meters

    if any(displacement >= threshold):
        return False
    return True

def trim_pre_post_nav_stationary(
        ride_data: pd.DataFrame,
        start_slack_seconds: int = 1,
        end_slack_seconds: int = 0.5,
    ) -> pd.DataFrame:
    """
    Trim the stationary segments before and after the navigation.
    """
    # Find the first and last non-stationary index based on the wheel RPM's
    rpm_1 = ride_data['control']['rpm_1']
    first_non_stationary = rpm_1[rpm_1 != 0].first_valid_index()
    last_non_stationary = rpm_1[rpm_1 != 0].last_valid_index()

    ride_start_index = first_non_stationary - pd.Timedelta(seconds=start_slack_seconds)
    ride_end_index = last_non_stationary + pd.Timedelta(seconds=end_slack_seconds)

    # Trim the ride data
    ride_data = ride_data[ride_start_index:ride_end_index]
    return ride_data

def remove_stationary_segment_images(
        ride_data: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Remove the images from the segments where no control commands were issued.
    """
    if ('front_camera' not in ride_data.columns) and ('rear_camera' not in ride_data.columns):
        return ride_data
    
    # Find the segments longer than the stationary_threshold
    # where both linear and angular commands are zero
    control = ride_data['control']

    # Find the non-stationary timestamps
    nonstat_timestamps = control[
            ((control['linear'] != 0) | (control['angular'] != 0)) & control['linear'].notna()
    ].index
    
    # Find the individual segments
    nonstat_segment_start_idxs = nonstat_timestamps[nonstat_timestamps.diff() > pd.Timedelta(seconds=2.0)]
    nonstat_segment_stop_idxs = nonstat_timestamps[nonstat_timestamps.diff(-1) < -pd.Timedelta(seconds=2.0)]

    # Add the first index as the start of the first segment
    nonstat_segment_start_idxs = nonstat_segment_start_idxs.insert(0, nonstat_timestamps[0])
    # Add the last index as the end of the last segment
    nonstat_segment_stop_idxs = nonstat_segment_stop_idxs.append(pd.Index([nonstat_timestamps[-1]]))

    # Set the camera frame_id to NaN for the data before, between, and after the non-stationary segments
    for camera_type in ['front_camera', 'rear_camera']:
        if camera_type in ride_data.columns:
            original_frame_ids = ride_data[(camera_type, 'frame_id')].copy()

            # Set the frame_id to NaN for all the timestamps
            ride_data[(camera_type, 'frame_id')] = np.nan

            # Set the frame_id to the original value for the non-stationary segments
            for start, stop in zip(nonstat_segment_start_idxs, nonstat_segment_stop_idxs):
                ride_data.loc[start:stop, (camera_type, 'frame_id')] = original_frame_ids.loc[start:stop]
    return ride_data

def trim_to_video_duration(ride_data: pd.DataFrame) -> pd.DataFrame:
    """
    Trim the ride data to the parts that have video frames.
    """
    camera_columns = ['front_camera', 'rear_camera']
    timestamps = []

    for camera in camera_columns:
        if camera in ride_data.columns:
            camera_timestamps = ride_data[camera]['frame_id'].dropna().index
            if not camera_timestamps.empty:
                timestamps.append((camera_timestamps[0], camera_timestamps[-1]))

    if not timestamps:
        return None

    start_time = max(t[0] for t in timestamps)
    end_time = min(t[1] for t in timestamps)

    # Trim the ride data
    ride_data = ride_data[start_time:end_time]
    return ride_data

def trim_to_gps_duration(ride_data: pd.DataFrame) -> pd.DataFrame:
    """
    Trim the ride data to the parts that have GPS data.
    """
    if 'gps' not in ride_data.columns:
        raise ValueError('No GPS data found in the ride data')

    start_time = ride_data['gps'].first_valid_index()
    end_time = ride_data['gps'].last_valid_index()

    # Trim the ride data
    ride_data = ride_data.loc[start_time:end_time]
    return ride_data

def detect_sensor_gaps(
        ride_data: pd.DataFrame,
        sensor_name: str,
        gap_duration_threshold: float = 2.0, # seconds
) -> pd.DataFrame:
    """
    Detect gaps in data stream of a given sensor.
    """
    valid_index = ride_data[sensor_name].dropna().index.to_series()
    is_gap_start = valid_index.diff(-1).dt.total_seconds() < -gap_duration_threshold
    is_gap_end = valid_index.diff().dt.total_seconds() > gap_duration_threshold

    # Create a dataframe with the gap start and end timestamps
    gap_indices = pd.DataFrame(index=range(is_gap_start.sum()), columns=['start', 'end'])
    gap_indices['start'] = valid_index[is_gap_start].values
    gap_indices['end'] = valid_index[is_gap_end].values

    return gap_indices

def remove_sensor_gaps(
    ride_data: pd.DataFrame,
    sensor_name: str,
    gap_duration_threshold: float = 2.0, # seconds
    mode: str = 'trim',
    ride_id: str = None,
    verbose: bool = False,
) -> List[pd.DataFrame]:
    """
    Remove ride data for segments that don't have data for the given sensor.
    Mode should be in
        'trim':
            Drop all the ride data starting from the first gap.
        'split':
            Split the ride into multiple episodes.
    """
    gap_indices = detect_sensor_gaps(ride_data, sensor_name, gap_duration_threshold)
    if gap_indices.empty:
        return [ride_data]

    if ride_id is not None:
        if verbose:
            print(f'''Gaps in {sensor_name} data found in {ride_id}:
                {gap_indices}''')

    # Split the ride data into episodes
    episodes = []
    if mode == 'trim':
        # Only pick the first segment until the first sensor gap
        episodes.append(ride_data[:gap_indices['start'].iloc[0]])

    elif mode == 'split':
        # Split the ride data into multiple episodes based on the sensor gaps
        for i in range(len(gap_indices) + 1):
            if i == 0:
                start = None
                stop = gap_indices['start'].iloc[i]
            elif i == len(gap_indices):
                start = gap_indices['end'].iloc[i-1]
                stop = None
            else:
                start = gap_indices['end'].iloc[i-1]
                stop = gap_indices['start'].iloc[i]

            episodes.append(ride_data[start:stop])
    else:
        raise ValueError(f'Unknown mode {mode}')
    
    return episodes
