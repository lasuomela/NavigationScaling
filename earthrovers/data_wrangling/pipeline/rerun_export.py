"""
Export processed ride data to Rerun.io format for visualization.
"""
import pandas as pd
import cv2
import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import ExitStack

from earthrovers.data_wrangling.pipeline.utils import datetime_to_nanos, get_video_path

def log_cmds(ride_data: pd.DataFrame):
    """
    Log the commands to rerun.io.
    """
    # Get the control columns
    control_data = ride_data['control'].dropna()
    for col_name, col_data in control_data.items():
        # Log the first control of the control horizon
        for t, cmd in col_data.items():
            rr.set_time_nanos("time", datetime_to_nanos(t))
            # If the command is a numpy array, log the first element
            cmd = cmd[0] if isinstance(cmd, np.ndarray) else cmd
            rr.log(col_name, rr.Scalar(cmd))

def log_state_estimates(ride_data: pd.DataFrame):
    """
    Log the sensor fusion state estimates to rerun.io.
    """
    assert 'state_estimate' in ride_data.columns, 'No state estimate data found in the ride data.'
    state_estimates = ride_data['state_estimate'].dropna()

    # Log the positions
    rr.log('state_estimate/path', rr.GeoLineStrings(
        lat_lon=state_estimates[['latitude', 'longitude']].values,
        colors=[255, 0, 0],  # Red color for the path
    ), static=True)

    rr.send_columns(
        "state_estimate/position",
        indexes=[rr.TimeNanosColumn("time", datetime_to_nanos(state_estimates.index))],
        columns=rr.GeoPoints.columns(positions=state_estimates[['latitude', 'longitude']]),
    )

    # Log the compass yaw as an arrow
    arrow_length = 0.0001  # Length of the arrow in degrees
    arrow_starts = state_estimates[['latitude', 'longitude']]
    arrow_ends = arrow_starts.copy()
    arrow_ends['latitude'] += arrow_length * np.cos(state_estimates['yaw'])
    arrow_ends['longitude'] += arrow_length * np.sin(state_estimates['yaw'])

    for ((t, arrow_start), (_, arrow_end)) in zip(arrow_starts.iterrows(), arrow_ends.iterrows()):
        rr.set_time_nanos("time", datetime_to_nanos(t))

        rr.log('state_estimate/compass', rr.GeoLineStrings(
            lat_lon=[
                arrow_start.values,
                arrow_end.values
            ],
            colors=[255, 255, 0],
        ))

def log_gps_compass(ride_data: pd.DataFrame):
    """
    Log the raw GPS and compass data to rerun.io.
    """
    assert 'gps' in ride_data.columns, 'No GPS data found in the ride data.'
    assert 'compass' in ride_data.columns, 'No compass yaw data found in the ride data.'

    gps_data = ride_data['gps'].dropna()
    compass_data = ride_data['compass'].dropna()

    # Log the GPS data
    rr.log('gps/path', rr.GeoLineStrings(lat_lon=gps_data.values, colors=[0,0,255]), static=True)
    rr.send_columns(
        "gps/position",
        indexes=[rr.TimeNanosColumn("time", datetime_to_nanos(gps_data.index))],
        columns=rr.GeoPoints.columns(positions=gps_data),
    )

    # Log the compass yaw
    for t, yaw in compass_data['yaw'].items():

        # Only start logging compass data after the first GPS measurement
        if t < gps_data.first_valid_index():
            continue

        rr.set_time_nanos("time", datetime_to_nanos(t))

        # Get the latest GPS measurement at the current timestamp
        gps_data_filtered = gps_data.loc[:t]
        if gps_data_filtered.empty:
            break
        latest_gps = gps_data_filtered.iloc[-1]

        # Make an arrow with length of 0.0001 degrees
        arrow_lat = latest_gps['latitude'] + 0.0001 * np.cos(yaw)
        arrow_lon = latest_gps['longitude'] + 0.0001 * np.sin(yaw)
        rr.log('compass', rr.GeoLineStrings(
            lat_lon=[
                [latest_gps['latitude'], latest_gps['longitude']],
                [arrow_lat, arrow_lon]
                ],
            colors=[0,255,0],
        ))

def init_camera(
    ride_data: pd.DataFrame,
    ride_path: Path,
    camera: str,
    stack: ExitStack = None,
    tempdir: Path = None,
):
    """
    Open the video file for the specified camera and return the VideoCapture object and frame IDs.
    """
    video_path = get_video_path(ride_path, camera, stack=stack, tempdir=tempdir)
    if video_path is None:
        return

    # Log the camera images
    vcap = cv2.VideoCapture(str(video_path))
    if not vcap.isOpened():
        raise Exception(f'Failed to open video {video_path}. Check if the file exists and is readable, and that your OpenCV was built with FFmpeg support.')

    video_frame_ids = ride_data[(camera, 'frame_id')].dropna()    
    return vcap, video_frame_ids
    
def log_camera_rrd(
    visualization_fps: int,
    visualization_image_width: int,
    vcap,
    video_frame_ids,
):
    """
    Log the camera images to rerun.io at the specified visualization_fps and image width.
    """
    latest_frame_time = pd.Timestamp("1970-01-01")
    visualization_interval = pd.Timedelta(f'{1/visualization_fps}s')
    grabbed_frame_idx = -1
    for t, frame_id in video_frame_ids.items():

        # Capture frames at the specified visualization_fps
        if (t-latest_frame_time) >= visualization_interval:

            # Seek to the frame with the specified frame_id
            while grabbed_frame_idx < frame_id:
                grabbed_frame_idx += 1
                vcap.grab()

            ret, frame = vcap.retrieve()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the image to width of visualization_image_width pixels
            frame = cv2.resize(frame, (
                visualization_image_width,
                int(frame.shape[0] * visualization_image_width / frame.shape[1]),
            ))

            rr.set_time_nanos("time", datetime_to_nanos(t))
            # Log the image as JPEG to save space
            rr.log("image", rr.Image(frame).compress(jpeg_quality=95))

            latest_frame_time = t

def log_navigation_objectives(ride_data: pd.DataFrame):
    """
    Log the navigation objective GPS locations to rerun.io.
    """
    if 'navigation_objective' not in ride_data.columns:
        print('No navigation objectives found in the ride data.')
        return
    
    navigation_data = ride_data['navigation_objective'].dropna()

    # Get the unique navigation objective rows
    navigation_data = navigation_data.drop_duplicates()

    # Generate colors using a colormap
    colormap = plt.cm.spring
    num_objectives = len(navigation_data)
    colors = [colormap(i / max(num_objectives - 1, 1)) for i in range(num_objectives)]

    for i, (index, objective) in enumerate(navigation_data.iterrows()):
        objective_coords = [objective['lat'], objective['lon']]
        color = [int(c * 255) for c in colors[i][:3]]  # Convert RGB to 0-255 scale
        rr.log(f'navigation/objective_{i}', rr.GeoPoints(lat_lon=objective_coords, colors=color), static=True)

def export_to_rerunio(
        ride_data: pd.DataFrame,
        ride_path,
        ride_id: str,
        output_path: Path,
):
    """
    Export the ride data to Rerun.io .rrd format.
    """

    # Create a new folder for the ride
    ride_output_path = output_path
    ride_output_path.mkdir(parents=True, exist_ok=True)

    # Export the ride data to Rerun.io format for visualization
    rr.init(ride_id)
    rr.save(ride_output_path / f'{ride_id}.rrd')

    # Log the commands
    log_cmds(ride_data)

    # Log the GPS and compass data
    log_gps_compass(ride_data)

    # Log the state estimates
    if 'state_estimate' in ride_data.columns:
        log_state_estimates(ride_data)

    # Log the navigation objectives
    log_navigation_objectives(ride_data)

    with ExitStack() as stack:

        # Log the front camera images
        vcap, video_frame_ids = init_camera(
            ride_data=ride_data,
            ride_path=ride_path,
            camera='front_camera',
            stack=stack,
            tempdir=Path('/dev/shm/'),
        )
        
        log_camera_rrd(
            visualization_fps=5,
            visualization_image_width=320,
            vcap=vcap,
            video_frame_ids=video_frame_ids,
        )
