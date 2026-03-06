"""
Prepare the ride data for export to a dataset for model training.
"""
from typing import List

import pandas as pd
import numpy as np

from earthrovers.data_wrangling.pipeline.state_estimation import wgs84_to_enu

def align_and_subsample_data(
        ride_data: pd.DataFrame,
        control_frequency: float = 10.0, # Hz
        control_horizon_length: int = 1, # seconds
        waypoint_prediction_horizon: int = 30, # seconds
        waypoint_prediction_interval: int = 3, # seconds
        export_columns: list = ['front_camera', 'gps', 'state_estimate', 'compass', 'control', 'navigation_objective'],
) -> pd.DataFrame:
    """
    Prepare the ride data for export to a dataset for model training.

    For each timestep we require:
    - The front camera frame
    - GPS position
    - Compass yaw
    - The next N linear and angular velocities

    GPS:
    - pad with the last known value

    Compass:
    - pad with the last known value

    Velocities:
    - Resample to 10Hz
    - For each row, pick the next N valid values

    Images:
    - Subsample at 4Hz
    """

    if 'state_estimate' not in ride_data.columns:
        # Remove state_estimate from export columns if it does not exist
        export_columns = [col for col in export_columns if col != 'state_estimate']

    # If 'navigation_objective' is in the export columns, add all the columns whose names start with 'navigation_objective'
    if 'navigation_objective' in export_columns:
        navigation_objective_columns = [
            col[0] for col in ride_data.columns if (col[0].startswith('navigation_objective'))
        ]
        export_columns += navigation_objective_columns
        # Remove duplicates
        export_columns = list(set(export_columns))

    # Fill NaNs in GPS and compass data by propagating the last known value
    ffill_columns = ['gps', 'compass', 'navigation_objective', 'state_estimate'] # State estimate should already be aligned with image timestamps
    for group in ffill_columns:
        if group in ride_data.columns:
            for col in ride_data[group].columns:
                ride_data[(group, col)] = ride_data[group][col].ffill().bfill()

    # Subsample the ride data
    noncontrol_columns = [ col for col in export_columns if col != 'control']
    image_indices_subsampled = ride_data['front_camera'].dropna().index
    ride_data_subsampled = ride_data.loc[image_indices_subsampled, noncontrol_columns]

    # For each row, calculate the number of rows forward until the navigation objective
    ride_data_subsampled = calculate_navigation_objective_rows_ahead(
        ride_data_subsampled=ride_data_subsampled,
        objective_columns=navigation_objective_columns,
    )

    # For each timestamp, compute robot future waypoints in the local frame
    ride_data_subsampled = get_future_waypoints_vectorized(
        ride_data_subsampled,
        position_header='state_estimate' if 'state_estimate' in ride_data.columns else 'gps',
        prediction_horizon=waypoint_prediction_horizon,  # seconds
        prediction_interval=waypoint_prediction_interval,  # seconds
    )

    if 'control' not in export_columns:
        return ride_data_subsampled
    
    # Create action chunks with desired control frequency and horizon length
    ride_data_subsampled = resample_control_signals(
        ride_data=ride_data,
        ride_data_subsampled=ride_data_subsampled,
        control_frequency=control_frequency,
        control_horizon_length=control_horizon_length,
    )
    return ride_data_subsampled

  
def calculate_navigation_objective_rows_ahead(
        ride_data_subsampled: pd.DataFrame,
        objective_columns: List,
) -> pd.DataFrame:
    """
    For each timestamp, calculate the number of rows forward until the row with the navigation objective gps location.

    This way, in the dataloader, we can easily access all the measurements at the navigation objective timestamp
    without having to store all the navigation objective data at each row.
    """
    for col in objective_columns:
        # Step 1: Extract and reset index
        objective_ts = ride_data_subsampled.reset_index()[
            [('timestamp', ''), (col, 'timestamp')]
        ].copy()

        # Step 2: Flatten MultiIndex columns
        objective_ts.columns = ['_'.join(filter(None, col)).strip() for col in objective_ts.columns]

        # Now columns are: ['timestamp', 'navigation_objective_timestamp']

        # Step 3: Add row number
        objective_ts['row_number'] = objective_ts.index

        # Step 4: Create reference timestamp DataFrame
        temp = objective_ts[['timestamp', 'row_number']].copy()

        # Step 6: Merge closest timestamp
        objective_ts_merged = pd.merge_asof(
            objective_ts,
            temp,
            left_on=f'{col}_timestamp',
            right_on='timestamp',
            direction='nearest',
            suffixes=('', '_matched')
        )

        # Step 7: Calculate rows ahead
        objective_ts_merged['rows_ahead'] = (
            objective_ts_merged['row_number_matched'] - objective_ts_merged['row_number']
        )
        objective_ts_merged = objective_ts_merged.set_index('timestamp')

        # Step 8: Add the rows ahead to the ride data
        ride_data_subsampled[(col, 'rows_ahead')] = objective_ts_merged['rows_ahead']
    return ride_data_subsampled

def resample_control_signals(
        ride_data: pd.DataFrame,
        ride_data_subsampled: pd.DataFrame,
        control_frequency: float = 10.0,
        control_horizon_length: int = 1,
) -> pd.DataFrame:
    """
    Resample control signals to match the timestamps of the subsampled ride data
    so that each row is assigned the future control signals over control_horizon_length seconds,
    at a frequency of control_frequency Hz.
    """
    # Create a union of all desired resample indices
    all_resample_idx = []
    for idx in ride_data_subsampled.index:
        window_idx = pd.date_range(start=idx, end=idx + pd.Timedelta(seconds=control_horizon_length),
                                freq=f'{1/control_frequency}s', inclusive='left')
        all_resample_idx.extend(window_idx)
    all_resample_idx = pd.DatetimeIndex(sorted(set(all_resample_idx)))

    # Reindex control data once and interpolate over the full range
    control_df = ride_data['control'].dropna()
    control_interp = control_df.reindex(control_df.index.union(all_resample_idx)).interpolate(method='slinear')
    control_interp = control_interp.ffill().bfill()

    # Define a helper to extract the control data window for a given timestamp
    def get_control_window(ts):
        window_idx = pd.date_range(start=ts, end=ts + pd.Timedelta(seconds=control_horizon_length),
                                freq=f'{1/control_frequency}s', inclusive='left')
        window = control_interp.loc[window_idx]
        return window['linear'].values.squeeze(), window['angular'].values.squeeze()

    # Now apply this extraction for all ride_data_subsampled indices
    linear_dfs = []
    angular_dfs = []
    for ts in ride_data_subsampled.index:
        linear_vals, angular_vals = get_control_window(ts)
        linear_dfs.append(linear_vals)
        angular_dfs.append(angular_vals)

    # Assign the lists to new columns in your ride data
    ride_data_subsampled[('control', 'linear')] = linear_dfs
    ride_data_subsampled[('control', 'angular')] = angular_dfs

    return ride_data_subsampled
    
def get_future_waypoints_vectorized(
    ride_data: pd.DataFrame,
    position_header: str = 'gps',
    prediction_horizon: int = 20,  # seconds
    prediction_interval: int = 5,  # seconds
):
    """
    For each row in the ride_data, compute future 'waypoints' relative to the current pose:
    robot future poses in the local frame of each row.

    Waypoints are in right-handed coordinates with x forward, y to left, positive yaw counter-clockwise.
    """
    if position_header not in ride_data.columns:
        raise ValueError(f"Expected '{position_header}' column in ride_data")

    orientation_header = 'compass' if position_header == 'gps' else position_header

    required_columns = [
        (position_header, 'latitude'),
        (position_header, 'longitude'),
        (orientation_header, 'yaw'),
        ('navigation_objective', 'timestamp'),
    ]

    current_pose = ride_data[required_columns].dropna()

    if current_pose.empty:
        raise ValueError(f"No valid data found in '{position_header}' column of ride_data")

    current_pose, _ = wgs84_to_enu(
        current_pose,
        position_header=position_header,
        orientation_header=orientation_header,
    )

    # Precompute future timestamps
    intervals = np.arange(prediction_interval, prediction_horizon + 1, prediction_interval)

    timestamps = current_pose.index.values[:, None] + np.array(pd.to_timedelta(intervals, unit='s'))

    # Flatten timestamps for efficient matching
    flat_timestamps = timestamps.ravel()
    matched_indices = current_pose.index.get_indexer(
        flat_timestamps, method='nearest', tolerance=pd.Timedelta(seconds=prediction_interval)
    )

    # Negative indices indicate no match found
    valid_matches = matched_indices >= 0 

    # Fetch matched rows efficiently
    matched_poses = current_pose.iloc[matched_indices[valid_matches]].copy()

    # Reshape matched poses to original shape
    num_rows, num_intervals = timestamps.shape
    waypoint_data = np.full((num_rows, num_intervals, 3), np.nan)

    # Coordinates for relative calculation
    e = current_pose[('enu', 'e')].values
    n = current_pose[('enu', 'n')].values
    yaw = current_pose[('enu', 'yaw')].values

    matched_e = matched_poses[('enu', 'e')].values
    matched_n = matched_poses[('enu', 'n')].values
    matched_yaw = matched_poses[('enu', 'yaw')].values

    rel_e = matched_e - np.repeat(e, num_intervals)[valid_matches]
    rel_n = matched_n - np.repeat(n, num_intervals)[valid_matches]
    rel_yaw = matched_yaw - np.repeat(yaw, num_intervals)[valid_matches]

    # Extract the reference yaw for the same matched indices
    ref_yaw = np.repeat(yaw, num_intervals)[valid_matches]

    # Rotate into the local frame of current_pose
    rel_x =  rel_e * np.cos(ref_yaw) + rel_n * np.sin(ref_yaw)
    rel_y = -rel_e * np.sin(ref_yaw) + rel_n * np.cos(ref_yaw)

    # Prepare full array
    waypoint_array = np.stack([rel_x, rel_y, rel_yaw], axis=-1)

    # Initialize the target array
    waypoint_data = np.full((num_rows, num_intervals, 3), np.nan)

    # Assign only valid matched positions
    waypoint_data.reshape(-1, 3)[valid_matches] = waypoint_array

    # Forward-fill NaN values in waypoint_data
    # Assuming waypoint_data has shape (rows, timestamps, features)
    mask = np.isnan(waypoint_data).any(axis=2)

    # Identify the indices of valid (non-NaN) values
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)

    # Forward-fill index positions
    np.maximum.accumulate(idx, axis=1, out=idx)

    # Use indices to gather forward-filled values
    rows = np.arange(waypoint_data.shape[0])[:, None]
    waypoint_data = waypoint_data[rows, idx]

    # Objective timestamp masking
    objective_timestamps = current_pose[('navigation_objective', 'timestamp')].values[:, None]
    mask_objective = timestamps > objective_timestamps

    # Set waypoints beyond objective timestamps
    for i in range(num_rows):
        objective_idx = np.where(~mask_objective[i])[0]
        if objective_idx.size:
            # Duplicate the last valid waypoint for all future timestamps
            # after the last objective timestamp
            last_valid = objective_idx[-1]
            waypoint_data[i, mask_objective[i]] = waypoint_data[i, last_valid]
        else:
            # If no valid waypoints, set all future waypoints to zero
            waypoint_data[i, mask_objective[i]] = 0

    waypoints_df = pd.DataFrame(
        {
            'timestamp': current_pose.index,
            'timestamps': list(timestamps),
            'x': waypoint_data[:, :, 0].tolist(),
            'y': waypoint_data[:, :, 1].tolist(),
            'yaw': waypoint_data[:, :, 2].tolist(),
        }
    ).set_index('timestamp')

    waypoints_df.columns = pd.MultiIndex.from_product(
        [['waypoints'], waypoints_df.columns]
    )

    return ride_data.join(waypoints_df, on='timestamp')