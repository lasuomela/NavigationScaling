"""
Use GTSAM to optimize the robot trajectory based on GPS, wheel odometry, and magnetometer data.
"""
from typing import Tuple

import gtsam
import pyproj
import pandas as pd
import numpy as np

from gtsam.symbol_shorthand import X # Pose key

from earthrovers.data_wrangling.pipeline.data_refinement import WGS84_yaw_from_magnetometer

def optimize_trajectory(ride_data: pd.DataFrame, fuse_magnetometer: bool = False) -> pd.DataFrame:
    """
    Optimize the trajectory of the rover based on ride data.
    
    Parameters:
    ride_data (pd.DataFrame): DataFrame containing time stamped ride data
    with columns ('gps', ['latitude', 'longitude']), ('control', ['rpm_1', 'rpm_2', 'rpm_3', 'rpm_4']),
    and optionally ('magnetometer', ['x', 'y', 'z']).
    
    Returns:
    pd.DataFrame: DataFrame with the pose estimates in columns ('state_estimate', ['latitude', 'longitude', 'yaw'])
        where 'yaw' is the compass heading in radians, e.g. 0 is North, pi/2 is East, pi is South, and -pi/2 is West.
    """
    # Create a temporary DataFrame to hold the state estimates
    state_estimate = ride_data.copy()

    # Transform GPS coordinates to a cartesian system (ENU)
    state_estimate, transformer = wgs84_to_enu(state_estimate)

    # Compute wheel odometry estimates from RPM data
    state_estimate = compute_wheel_odometry(state_estimate)

    # Compute yaw estimates from magnetometer data
    if fuse_magnetometer:
        state_estimate = enu_yaw_from_magnetometer(state_estimate)

    # Apply nonlinear factor graph smoother to the ride data to get better pose estimates
    state_estimate = apply_smoother(state_estimate, fuse_magnetometer=fuse_magnetometer)

    # Convert ENU coordinates back to WGS84 (latitude, longitude, altitude)
    state_estimate = enu_to_wgs84(state_estimate, transformer)

    # Add the state estimates to the original ride data with a 'state_estimate' header in multi-index
    state_estimate = state_estimate.rename(columns={'wgs84': 'state_estimate'})[['state_estimate']]

    # Merge the state estimates with the original ride data, making sure no timestams are duplicated
    ride_data = ride_data.join(state_estimate, how='outer')

    return ride_data

def apply_smoother(
        ride_data: pd.DataFrame,
        fuse_magnetometer = False,
        interpolation_method: str = 'smoother', # none | smoother
        plot_result: bool = False,
) -> pd.DataFrame:
    """
    Apply a GTSAM smoother to the ride data to get better pose estimates.

    Args:
        ride_data (pd.DataFrame): DataFrame containing time stamped sensor measurements
        fuse_magnetometer (bool): Whether to fuse yaw estimates from the magnetometer
        interpolation_method (str): Method to use for interpolation of readout variables. Options are 'none' or 'smoother'.
        plot_result (bool): Whether to produce a debug plot of the trajectory after optimization.
    """

    if interpolation_method == 'smoother':
        add_readout_variables = True
    else:
        add_readout_variables = False

    # Initialize
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Define noise models - These are empirically tuned and may not be optimal
    gps_sigmas = np.array([1.0, 1.0, 0.0]) * 2 # GPS noise in meters
    magnetometer_yaw_sigmas = np.array([np.pi/3, np.pi/3, np.pi/3]) *2 # Magnetometer noise in radians
    odom_sigmas = np.array([0., 0., 0.1, 0.05, 0.05, 0.05]) * 4 # Odometry noise in radians and meters. This is scaled by the time step.

    gps_noise_model = gtsam.noiseModel.Diagonal.Sigmas(gps_sigmas)
    magnetometer_yaw_noise_model = gtsam.noiseModel.Diagonal.Sigmas(magnetometer_yaw_sigmas)

    i=0
    previous_pose_key = None
    previous_odometry_ts = None
    previous_odometry_pose = None
    gps_measurement = None
    magnetometer_yaw_measurement = None

    # Create a dictionary to hold the pose keys and their corresponding timestamps
    pose_key_to_ts = {}
    measurement_out_keys_to_ts = {}

    if ride_data['front_camera'].dropna().empty:
        raise ValueError("No front camera data found in the ride data. Please ensure the ride data contains front camera measurements.")

    # Check for duplicate timestamps and raise an error if found
    if ride_data.index.duplicated().any():
        # Print the duplicate rows
        duplicate_rows = ride_data[ride_data.index.duplicated(keep=False)]
        raise ValueError(f"Duplicate timestamps found in the ride data:\n{duplicate_rows}")

    for ts, row in ride_data.iterrows():

        # Add readout variables at the front camera timestamps
        if add_readout_variables and (not row['front_camera'].isna().all()):
            pose_key = X(i)
            measurement_out_keys_to_ts[pose_key] = ts  # Store the timestamp for the pose key
            i += 1

            # Add initial estimate for the pose
            if not initial_estimate.exists(pose_key):
                initial_estimate.insert(
                    pose_key,
                    gtsam.Pose3(
                        magnetometer_yaw_measurement if magnetometer_yaw_measurement is not None else gtsam.Rot3(),
                        gps_measurement if gps_measurement is not None else gtsam.Point3()
                    )
                )

            # Add an odomotery factor connecting to the previous variable
            previous_pose_key, previous_odometry_ts, previous_odometry_pose = add_odometry_factor(
                graph,
                row,
                previous_odometry_pose,
                previous_pose_key,
                pose_key,
                odom_sigmas,
                ts,
                previous_odometry_ts
            )

        # Add a GPS factor if GPS data is available
        added_gps = False
        if not any(row['enu'].isna()):
            added_gps = True
            pose_key = X(i)  # Create a pose key for the current time step
            pose_key_to_ts[pose_key] = ts  # Store the timestamp for the pose key
            i += 1

            # GPS measurement in ENU coordinates
            gps_measurement = gtsam.Point3(
                row['enu'].values
            )

            # Add GPS factor to the graph
            gps_factor = gtsam.GPSFactor(
                pose_key,
                gps_measurement,
                gps_noise_model
            )
            graph.add(gps_factor)

            # Add initial estimate for the pose
            initial_estimate.insert(
                pose_key,
                gtsam.Pose3(
                    magnetometer_yaw_measurement if magnetometer_yaw_measurement is not None else gtsam.Rot3(),
                    gtsam.Point3(gps_measurement)
                )
            )

            # Add an odometry factor
            previous_pose_key, previous_odometry_ts, previous_odometry_pose = add_odometry_factor(
                graph,
                row,
                previous_odometry_pose,
                previous_pose_key,
                pose_key,
                odom_sigmas,
                ts,
                previous_odometry_ts
            )

        # Add a magnetometer yaw factor if magnetometer data is available
        if fuse_magnetometer and (not np.isnan(row['enu_yaw'].values.item())):
            if added_gps:
                pose_key = X(i-1)  # Use the same pose key as the GPS measurement
            else:
                pose_key = X(i)  # Create a pose key for the current time step
                pose_key_to_ts[pose_key] = ts  # Store the timestamp for the pose key
                i += 1

            # Magnetometer yaw measurement in radians
            magnetometer_yaw_measurement = row['enu_yaw'].values.item()
            magnetometer_yaw_measurement = gtsam.Rot3.Yaw(magnetometer_yaw_measurement)
            yaw_prior = gtsam.PoseRotationPrior3D(
                pose_key,
                magnetometer_yaw_measurement,
                magnetometer_yaw_noise_model,
            )
            graph.add(yaw_prior)

            # Add initial estimate for the pose
            if not initial_estimate.exists(pose_key):
                initial_estimate.insert(
                    pose_key,
                    gtsam.Pose3(
                        magnetometer_yaw_measurement,
                        gps_measurement if gps_measurement is not None else gtsam.Point3()
                    )
                )

            # Add an odometry factor
            previous_pose_key, previous_odometry_ts, previous_odometry_pose = add_odometry_factor(
                graph,
                row,
                previous_odometry_pose,
                previous_pose_key,
                pose_key,
                odom_sigmas,
                ts,
                previous_odometry_ts
            )

    # Optimize the graph
    # NOTE: LevenbergMarquardtOptimizer seems to give better results than GaussNewtonOptimizer
    # However, it seems to fail silently when the optimization fails, and just returns the initial estimate.
    # This is often caused by the magnetometer being installed in wrong orientation on the robot,
    # leading to large initial errors in yaw.
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    # Plot the trajectory
    if plot_result:
        plot_trajectory(result)

    if add_readout_variables:
        out_vars = measurement_out_keys_to_ts
    else:
        out_vars = pose_key_to_ts

    # Extract the pose estimates from the result
    data = []
    for pose_key, ts in out_vars.items():
        pose = result.atPose3(pose_key)
        data.append({
            'timestamp': ts,
            'e': pose.translation()[0],
            'n': pose.translation()[1],
            'u': pose.translation()[2],
            'yaw': pose.rotation().yaw()
        })
    estimated_state = pd.DataFrame(data).set_index('timestamp')

    # Make multi-index with 'state_estimate' level
    estimated_state.columns = pd.MultiIndex.from_product([['enu'], estimated_state.columns])

    # Replace the original 'enu' columns with the estimated state
    ride_data = ride_data.drop(columns='enu', level=0)
    ride_data = ride_data.join(estimated_state, how='outer')

    return ride_data

def add_odometry_factor(
        graph: gtsam.NonlinearFactorGraph,
        measurement: pd.Series,
        previous_odometry_pose: gtsam.Pose3,
        previous_pose_key: int,
        current_pose_key: int,
        odom_sigmas: np.ndarray,
        current_ts: pd.Timestamp,
        previous_ts: pd.Timestamp,
    ) -> None:
        
        theta = measurement['odom_theta'].values.item()
        x = measurement['odom_x'].values.item()
        y = measurement['odom_y'].values.item()
        current_odometry_pose = gtsam.Pose3(
            gtsam.Rot3.Yaw(theta),
            gtsam.Point3(x, y, 0.0)
        )

        # Add a between factor to connect to the previous pose
        if previous_pose_key != None:

            # Get the relative pose between current and previous measurement
            rel_pose = previous_odometry_pose.transformPoseTo(current_odometry_pose)

            # Create a between factor from the wheel odometry
            duration = (current_ts - previous_ts).total_seconds()
            odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(odom_sigmas * duration) # Scale by duration
            between_factor = gtsam.BetweenFactorPose3(previous_pose_key, current_pose_key, rel_pose, odometry_noise)
            graph.add(between_factor)
            
        previous_odometry_pose = current_odometry_pose
        previous_odometry_ts = current_ts
        previous_pose_key = current_pose_key

        return previous_pose_key, previous_odometry_ts, previous_odometry_pose

def enu_yaw_from_magnetometer(ride_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the
    """
    ned_yaw = WGS84_yaw_from_magnetometer(
        ride_data['magnetometer'],
        None, # No reference location since we don't correct for declination
        False, # Let's not correct declination for now
    )

    ride_data['enu_yaw'] = wgs84_yaw_to_enu(ned_yaw)
    return ride_data

def wgs84_yaw_to_enu(
        yaw: pd.DataFrame | np.ndarray,
) -> pd.DataFrame | np.ndarray:
    """
    Converts WGS84 yaw (positive clockwise from North) to ENU yaw (positive counter-clockwise from East).
    
    Parameters:
    yaw (pd.DataFrame | np.ndarray): WGS84 yaw values in radians.
    
    Returns:
    pd.DataFrame | np.ndarray: ENU yaw values in radians.
    """
    # Convert WGS84 yaw to ENU yaw
    enu_yaw = -(yaw - np.pi/2)  # Convert to clockwise from North
    enu_yaw = (enu_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi)
    
    return enu_yaw

def enu_to_wgs84(
        ride_data: pd.DataFrame,
        transformer: pyproj.Transformer,
    ) -> pd.DataFrame:
    """
    Converts local ENU coordinates to WGS84 (lat, lon, alt) with respect to a reference origin.
    Parameters:
    ride_data (pd.DataFrame): DataFrame containing ENU coordinates in columns ['e', 'n', 'u'] and yaw in ['yaw'].
    transformer (pyproj.Transformer): Transformer object that was used to convert WGS84 to ENU.
    """
    enu = ride_data['enu']
    # Convert ENU coordinates to WGS84
    lon, lat, alt = transformer.transform(
        enu['e'].values,
        enu['n'].values,
        enu['u'].values,
        direction=pyproj.enums.TransformDirection.INVERSE
    )
    # ENU yaw is positive counter-clockwise from East. Convert it to WGS84 yaw which is positive clockwise from North.
    wgs84_yaw = -(enu['yaw'] - np.pi/2)  # Convert to clockwise from North
    wgs84_yaw = (wgs84_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi)

    ride_data[('wgs84', 'latitude')] = lat
    ride_data[('wgs84', 'longitude')] = lon
    ride_data[('wgs84', 'altitude')] = alt
    ride_data[('wgs84', 'yaw')] = wgs84_yaw
    return ride_data
    


def wgs84_to_enu(ride_data: pd.DataFrame,
                 position_header: str = 'gps',
                 orientation_header: str = None,
) -> Tuple[pd.DataFrame, pyproj.Transformer]:
    """
    Converts WGS84 (lat, lon, alt) to local ENU coordinates with respect to a reference origin.
    """
    origin_lat = ride_data[(position_header, 'latitude')].dropna().iloc[0]
    origin_lon = ride_data[(position_header, 'longitude')].dropna().iloc[0]
    origin_alt = 0  # Assuming altitude is zero for simplicity

    # Define the transformation pipeline (WGS84 -> ECEF -> ENU)
    pipeline = f"""
    +proj=pipeline
    +step +proj=cart
    +step +proj=topocentric +lon_0={origin_lon} +lat_0={origin_lat} +z={origin_alt}
    """
    geographic_to_enu = pyproj.Transformer.from_pipeline(pipeline)

    # Convert the coordinates
    e, n, u = geographic_to_enu.transform(
        ride_data[(position_header, 'longitude')].values,
        ride_data[(position_header, 'latitude')].values,
        np.zeros_like(ride_data[(position_header, 'latitude')].values)  # Assuming altitude is zero for simplicity
    )
    ride_data[('enu', 'e')] = e
    ride_data[('enu', 'n')] = n
    ride_data[('enu', 'u')] = u

    if orientation_header is not None:
        # Convert yaw from WGS84 to ENU
        ride_data[('enu', 'yaw')] = wgs84_yaw_to_enu(
            ride_data[(orientation_header, 'yaw')].values
        )

    return ride_data, geographic_to_enu

def compute_wheel_odometry(
        ride_data: pd.DataFrame,
        wheel_radius=0.065,
        track_width=0.206,
        max_yaw_rate=3/5*np.pi,
    ) -> pd.DataFrame:
    """
    Computes wheel odometry estimates from RPM data.
    
    Parameters:
    ride_data (pd.DataFrame): DataFrame containing control data with columns ('control', ['rpm_1', 'rpm_2', 'rpm_3', 'rpm_4']).
    
    Returns:
    pd.DataFrame: The input DataFrame with wheel odometry estimates added in the columns:
        ['v', 'w', 'dx', 'dy', 'dtheta', 'odom_x', 'odom_y', 'odom_theta'].
    """

    wheel_rpms = ride_data['control'].drop(['linear', 'angular'], axis=1)

    # Rename the columns for clarity
    wheel_rpms.rename(columns={
        'rpm_1': 'left_front',
        'rpm_2': 'right_front',
        'rpm_3': 'left_rear',
        'rpm_4': 'right_rear'
    }, inplace=True)

    # Calculate the robot x-axis velocity (v) and yaw rate (w) from the wheel RPMs
    v = wheel_radius * np.pi / 60 * wheel_rpms.mean(axis=1)
    w = wheel_radius * np.pi / (60 * track_width) * (
        (wheel_rpms['right_front'] + wheel_rpms['right_rear'])/2 -
        (wheel_rpms['left_front'] + wheel_rpms['left_rear'])/2
    ) # Positive yaw rate is counter-clockwise

    # Clip the angular velocity to the maximum yaw rate
    w = np.clip(w, -max_yaw_rate, max_yaw_rate)

    # Add linear and angular velocities to the ride_data DataFrame
    ride_data['v'] = v.astype(float)
    ride_data['w'] = w.astype(float)

    # Interpolate NaN values in linear and angular velocities - equal to assuming constant acceleration
    mask = ride_data.index.to_series().diff() < pd.Timedelta(seconds=0.2)
    v_interp = ride_data['v'].interpolate(method='slinear')[mask]
    w_interp = ride_data['w'].interpolate(method='slinear')[mask]
    ride_data['v'] = ride_data['v'].combine_first(v_interp)
    ride_data['w'] = ride_data['w'].combine_first(w_interp)

    # Fill any remaining NaN values with 0
    ride_data['v'] = ride_data['v'].fillna(0)
    ride_data['w'] = ride_data['w'].fillna(0)

    # Calculate the duration of each velocity
    ride_data['dt'] = -ride_data.index.to_series().diff(periods=-1).dt.total_seconds()
    ride_data['dt'] = ride_data['dt'].fillna(0)  # Fill NaN values with 0 for the last row    
    ride_data['dt'] = ride_data['dt'].clip(lower=0, upper=0.2) # Clip dt to a maximum of 1.0 seconds to avoid large jumps

    # Calculate dx, dy, and dtheta induced by each v, w (in robot frame)
    ride_data['dx'] = ride_data['v'] * np.cos(ride_data['w']) * ride_data['dt']
    ride_data['dy'] = ride_data['v'] * np.sin(ride_data['w']) * ride_data['dt']
    ride_data['dtheta'] = (ride_data['w'] * ride_data['dt'])

    # Calculate cumulative change in orientation with respect to the initial orientation (odometry frame)
    ride_data['odom_theta'] = ride_data['dtheta'].cumsum()
    ride_data['odom_theta'] = ride_data['odom_theta'] % (2 * np.pi)  # Normalize to [0, 2*pi)
    # ride_data['odom_theta'] = (ride_data['odom_theta'] + np.pi) % (2 * np.pi) - np.pi # Go to the range [-pi, pi)

    # Rotate each dx, dy by the odometry frame orientation at that point to get position deltas in the odometry frame
    ride_data['odom_dx'] = ride_data['dx'] * np.cos(ride_data['odom_theta']) - ride_data['dy'] * np.sin(ride_data['odom_theta'])
    ride_data['odom_dy'] = ride_data['dx'] * np.sin(ride_data['odom_theta']) + ride_data['dy'] * np.cos(ride_data['odom_theta'])

    # Calculate the cumulative position in the odometry frame
    ride_data['odom_x'] = ride_data['odom_dx'].cumsum()
    ride_data['odom_y'] = ride_data['odom_dy'].cumsum()

    # Add a leading zero for the initial pose and shift the values by one to align with the previous position
    ride_data['odom_theta'] = ride_data['odom_theta'].shift(1).fillna(0)
    ride_data['odom_x'] = ride_data['odom_x'].shift(1).fillna(0)
    ride_data['odom_y'] = ride_data['odom_y'].shift(1).fillna(0)
    return ride_data


def plot_trajectory(result: gtsam.Values):
    import matplotlib.pyplot as plt
    import gtsam.utils.plot as gtsam_plot

    # Subsample for plotting
    plot_values = gtsam.Values()
    for i, pose_key in enumerate(result.keys()):
        if i % 5 == 0:
            # Get the pose
            pose = result.atPose3(pose_key)
            # Update the result with the subsampled pose
            plot_values.insert(pose_key, pose)
            # if plot_values.size() > 0:
            #     break

    # Plot the results
    gtsam_plot.plot_trajectory(0, plot_values, scale =1.0)

    # Set the z axis to zero for a 2D plot
    ax = plt.gca()
    ax.set_zlim(0, 1)

    # Set x and y axis to be equal
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Set the limits to be equal
    ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    ax.set_aspect('equal', adjustable='box')

    # Set the viewpoint to look down on the XY plane, with positive y up, positive x to the right, and negative z towards the viewer
    ax.view_init(elev=90, azim=-90)

    # Save the plot
    plt.savefig('gtsam_trajectory.png')
    # plt.close()
