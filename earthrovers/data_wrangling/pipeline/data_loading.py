"""
Utilities to load the raw FrodoBots dataset files from disk.
"""
import json
import pandas as pd
from pathlib import Path

from earthrovers.data_wrangling.pipeline.utils import get_ride_sensor_path, read_csv

def load_cmd_data(
        ride_path: Path,
        ride_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Load the control data from the ride path and merge it with the ride data.
    """
    control_path = get_ride_sensor_path(ride_path, 'cmd')        
    control_data = read_csv(control_path)

    # Convert timestamps
    control_data['timestamp'] = pd.to_datetime(control_data['timestamp'], unit='s')  

    # Compute absolute velocities once
    control_data['abs_angular'] = control_data['angular'].abs()
    control_data['abs_linear'] = control_data['linear'].abs()

    # Get the index of the rows with max angular velocity
    max_angular_idx = control_data.groupby('timestamp')['abs_angular'].idxmax()

    # Create a subset with those max angular rows
    control_data = control_data.loc[max_angular_idx]

    # For ties, use max linear velocity
    duplicate_timestamps = control_data.index.duplicated(keep=False)
    if duplicate_timestamps.any():
        tie_group = control_data[duplicate_timestamps]
        max_linear_idx = tie_group.groupby(tie_group.index)['abs_linear'].idxmax()
        control_data = pd.concat([
            control_data[~duplicate_timestamps],
            control_data.loc[max_linear_idx]
        ])

    # Set the timestamp as the index
    control_data.set_index('timestamp', inplace=True)

    # Drop auxiliary columns
    control_data = control_data.drop(columns=['abs_angular', 'abs_linear'])

    # Rename columns
    control_data.columns = pd.MultiIndex.from_product([['control'], control_data.columns])

    # Check for duplicate timestamps
    if control_data.index.duplicated().any():
        raise ValueError(f'Duplicate timestamps in control data for {ride_path}')
    
    # Merge with ride data
    ride_df = ride_df.join(control_data, how='outer')
    return ride_df

def _load_gps_measurement(gps_path: Path) -> pd.DataFrame:
    gps_data = read_csv(gps_path)

    # Remove any rows with invalid GPS data
    gps_data = gps_data[gps_data['latitude'].abs() <= 180]
    gps_data = gps_data[gps_data['longitude'].abs() <= 180]

    # The GPS timestamp is milliseconds since epoch, convert to seconds
    gps_data['timestamp'] = pd.to_datetime(gps_data['timestamp'] / 1000, unit='s')
    gps_data.set_index('timestamp', inplace=True)
    gps_data.columns = pd.MultiIndex.from_product([['gps'], gps_data.columns])

    # Check for duplicate timestamps
    duplicate_timestamps = gps_data.index.duplicated()
    if duplicate_timestamps.any():
        # Remove duplicates
        gps_data = gps_data[~duplicate_timestamps]
    return gps_data

def load_gps_data(
        ride_path: Path,
        ride_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the GPS data from the ride path and merge it with the ride data.
    """
    gps_path = get_ride_sensor_path(ride_path, 'gps')
    if not gps_path:
        print(f'No GPS data found in {ride_path}')
        return ride_df

    gps_data = _load_gps_measurement(gps_path)

    # Merge the GPS data with the ride data
    ride_df = ride_df.join(gps_data, how='outer')
    return ride_df

def load_accelerometer_data(
        ride_path: Path,
        ride_df: pd.DataFrame,
) -> pd.DataFrame:
    
    imu_path = get_ride_sensor_path(ride_path, 'imu')
    raw_data = read_csv(imu_path)
    
    accel_data_outer = pd.DataFrame(columns=['data','timestamp'])
    for accel_row, ts in zip(raw_data['accelerometer'], raw_data['timestamp']):
        accel_data_ = pd.DataFrame(columns=['x','y','z', 'timestamp_robot'])
        for accel in json.loads(accel_row):
            accel_data_.loc[len(accel_data_)] = [float(a) for a in accel]
        accel_data_outer.loc[len(accel_data_outer)] = [accel_data_, ts]
    
    accel_data_outer['timestamp'] = pd.to_datetime(accel_data_outer['timestamp'], unit='ms')
    accel_data_outer.set_index('timestamp', inplace=True)
    accel_data_outer.columns = pd.MultiIndex.from_product([['accelerometer'], accel_data_outer.columns])

    # Check for duplicate timestamps
    if accel_data_outer.index.duplicated().any():
        raise ValueError(f'Duplicate timestamps in accelerometer data for {ride_path}')

    # Merge the compass data with the ride data
    ride_df = ride_df.join(accel_data_outer, how='outer')
    return ride_df

def load_gyroscope_data(
        ride_path: Path,
        ride_df: pd.DataFrame,
) -> pd.DataFrame:
    
    imu_path = get_ride_sensor_path(ride_path, 'imu')
    raw_data = read_csv(imu_path)

    gyro_data_outer = pd.DataFrame(columns=['data','timestamp'])
    for gyro_row, ts in zip(raw_data['gyroscope'], raw_data['timestamp']):
        gyro_data_ = pd.DataFrame(columns=['x','y','z', 'timestamp_robot'])
        for gyro in json.loads(gyro_row):
            gyro_data_.loc[len(gyro_data_)] = [float(a) for a in gyro] # @todo supersample and add white noise
        gyro_data_outer.loc[len(gyro_data_outer)] = [gyro_data_, ts]

    gyro_data_outer['timestamp'] = pd.to_datetime(gyro_data_outer['timestamp'], unit='ms')
    gyro_data_outer.set_index('timestamp', inplace=True)
    gyro_data_outer.columns = pd.MultiIndex.from_product([['gyroscope'], gyro_data_outer.columns])

    # Check for duplicate timestamps
    if gyro_data_outer.index.duplicated().any():
        raise ValueError(f'Duplicate timestamps in gyroscope data for {ride_path}')

    ride_df = ride_df.join(gyro_data_outer, how='outer')
    return ride_df

def load_magnetometer_data(
        ride_path: Path,
        ride_df: pd.DataFrame,
) -> pd.DataFrame:
    
    mag_path = get_ride_sensor_path(ride_path, 'imu')
    raw_data = read_csv(mag_path, usecols=['compass', 'timestamp'])

    compass_data_list = []
    for compass_row, server_ts in zip(raw_data['compass'], raw_data['timestamp']):
        for compass in json.loads(compass_row):
            compass_data_list.append([float(c) for c in compass] + [server_ts])
    compass_data = pd.DataFrame(compass_data_list, columns=['x', 'y', 'z', 'timestamp', 'timestamp_server'])

    compass_data['timestamp'] = pd.to_datetime(compass_data['timestamp'], unit='s')
    compass_data.set_index('timestamp', inplace=True)
    compass_data.columns = pd.MultiIndex.from_product([['magnetometer'], compass_data.columns])

    # Check for duplicate timestamps
    duplicate_timestamps = compass_data.index.duplicated()
    if duplicate_timestamps.any():
        # Remove duplicates
        compass_data = compass_data[~duplicate_timestamps]
    
    # Merge the compass data with the ride data
    ride_df = ride_df.join(compass_data, how='outer')
    return ride_df

def load_camera_timestamps(
        ride_path: Path,
        ride_df: pd.DataFrame,
        camera_type: str,
        verbose: bool = False,
) -> pd.DataFrame:
    
    timestamps_path = get_ride_sensor_path(ride_path, f'{camera_type}_camera')
    if not timestamps_path:
        if verbose:
            print(f'No {camera_type} camera timestamps found in {ride_path}')
        return ride_df
    
    # Read camera timestamps
    timestamps = read_csv(timestamps_path)
    timestamps['timestamp'] = pd.to_datetime(timestamps['timestamp'], unit='s')
    timestamps.set_index('timestamp', inplace=True)
    timestamps.columns = pd.MultiIndex.from_product([[f'{camera_type}_camera'], timestamps.columns])

    # Check for duplicate timestamps
    if timestamps.index.duplicated().any():
        # The frame idx to timestamp mapping doesn't seem reliable
        return ride_df

    # Merge the timestamps with the ride data
    ride_df = ride_df.join(timestamps, how='outer')
    return ride_df

def load_raw_data(
        ride_path: Path,
        data_types: list,
        verbose: bool = False,
) -> pd.DataFrame:
    """
    Load a single raw FrodoBots ride into a pandas DataFrame.
    """
    # Load the ride data
    ride_data = pd.DataFrame(
        index=pd.to_datetime([]),
        columns=pd.MultiIndex.from_product([['ride'], []])
    )

    if 'cmd' in data_types:
        ride_data = load_cmd_data(ride_path, ride_data)

    if 'gps' in data_types:
        ride_data = load_gps_data(ride_path, ride_data)

    if 'magnetometer' in data_types:
        ride_data = load_magnetometer_data(ride_path, ride_data)

    if 'accelerometer' in data_types:
        ride_data = load_accelerometer_data(ride_path, ride_data)

    if 'gyroscope' in data_types:
        ride_data = load_gyroscope_data(ride_path, ride_data)

    if 'front_camera' in data_types:
        ride_data = load_camera_timestamps(ride_path, ride_data, 'front', verbose=verbose)

    if 'rear_camera' in data_types:
        ride_data = load_camera_timestamps(ride_path, ride_data, 'rear', verbose=verbose)

    # Sort the ride data by timestamp
    ride_data.sort_index(inplace=True)

    return ride_data