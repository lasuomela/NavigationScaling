from typing import List, Tuple

import pandas as pd
import zipfile
from datetime import datetime
from pathlib import Path
import fnmatch
import tqdm
from contextlib import ExitStack
import tempfile
import shutil

def datetime_to_nanos(dt: datetime):
    """
    Convert a datetime object to nanoseconds since epoch.
    """
    return (dt - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ns")

class ZipPath(zipfile.Path):
    """
    A subclass of zipfile.Path that implements 'glob' method.
    """
    def glob(self, pattern):
        """
        Return a generator that yields all the paths matching the pattern.

        VERY SLOW as has to iterate all paths in root.namelist()
        """
        for path in self.iterdir():
            if fnmatch.fnmatchcase(path.name, pattern):
                yield path

    @property
    def stem(self):
        """
        Return the stem of the zip file. This was only added in Python 3.11 so add it here.
        """
        return self.name.split('.')[0]

def fast_reconstruct_zip_path_list(zip_tuple_paths: List[Tuple[str, Tuple[str, str], int]]):
    """
    Reconstruct a list of ZipPath's from the list of tuples.
    ZipPath/zipfile.Path init is slow, building child paths through '/' is fast.
    """
    base_files = {}
    zip_obj_paths = []
    for ride_id, (root, at), cluster in zip_tuple_paths:
        if root not in base_files:
            base_files[root] = ZipPath(root)
        # Create ZipPath objects for each ride path
        zip_obj_paths.append((ride_id, base_files[root] / at, cluster))
    return zip_obj_paths
    
def split_zip_path(ride_path: ZipPath | str):
    """
    Split a zip file path into 'root' and 'at' components of a ZipPath object. 
    """
    if isinstance(ride_path, ZipPath):
        root = ride_path.root.filename
        at = ride_path.at
    elif isinstance(ride_path, str):
        assert '.zip/' in ride_path, f'Expected a zip file path, got {ride_path}'
        components = ride_path.split('.zip/')
        root = components[0] + '.zip'
        at = components[1]
    else:
        raise TypeError(f'Expected ZipPath or str, got {type(ride_path)}')
    
    return root, at

def split_zip_path_list( ride_list: List[Tuple[str, Path, int]]):
    """
    Split a list of ZipPath tuples into a list of tuples (ride_id, (filename, at), cluster_idx).
    This is used to reconstruct the ZipPath objects later.
    """
    ride_list = [
        (ride_id, 
        split_zip_path(ride_path),  # Split the ZipPath into root and at components
        cluster
        ) for ride_id, ride_path, cluster in ride_list]
    return ride_list
    

def get_ride_sensor_path(ride_path: Path, file_type: str):

    ride_id = ride_path.stem.split('_')[1]

    if file_type == 'gps':
        filename = f'gps_data_{ride_id}.csv'
    elif file_type == 'cmd':
        filename = f'control_data_{ride_id}.csv'
    elif file_type == 'imu':
        filename = f'imu_data_{ride_id}.csv'
    elif file_type == 'front_camera':
        filename = f'front_camera_timestamps_{ride_id}.csv'
    elif file_type == 'rear_camera':
        filename = f'rear_camera_timestamps_{ride_id}.csv'
    else:
        raise ValueError(f'Unknown file type: {file_type}')
    
    file_path = ride_path / filename
    if not file_path.is_file():
        return None
    return file_path

def read_csv(path: Path | zipfile.Path, **kwargs):
    """
    Extend the pandas.read_csv function to handle zip files.
    """
    if isinstance(path, zipfile.Path):
        # Read CSV from zip file
        with path.open('r') as f:
            df = pd.read_csv(f, **kwargs)
    else:
        # Read CSV from regular file
        df = pd.read_csv(path, **kwargs)
    return df

def list_rides(
        dataset_path: Path,
)    -> Tuple[pd.DataFrame, bool]:
    """
    List all the rides in the dataset directory.
    """

    # List all the rides in the dataset
    ride_paths = {}
    dir_pattern = 'output_rides_*'
    rides_dirs = list(dataset_path.glob(dir_pattern))

    # Check if the rides are in zip archives or expanded directories
    first_dir_type = rides_dirs[0].suffix
    for ride_dir in rides_dirs:
        current_dir_type = ride_dir.suffix
        assert current_dir_type == first_dir_type, \
            f'The dataset directory should contain either all zip archives or all expanded directories, ' \
            f'but found both {first_dir_type} and {current_dir_type}.'
    zipped = first_dir_type == '.zip'

    for ride_dir in tqdm.tqdm(rides_dirs, desc='Listing rides'):
        if zipped:
            ride_dir = ZipPath(ride_dir, at=f"{ride_dir.stem}/")
        dir_ride_paths = ride_dir.glob('ride_*')
        ride_paths.update({ride_path.stem: ride_path for ride_path in dir_ride_paths})

    ride_paths = pd.DataFrame(ride_paths.items(), columns=['ride_id', 'ride_path'])
    ride_paths.index = ride_paths.ride_id
    ride_paths = ride_paths.drop(columns='ride_id')
    return ride_paths, zipped

def get_video_path(
        ride_path: Path,
        camera: str,
        stack: ExitStack = None,
        tempdir: Path = Path('/dev/shm/')
    ):
    """
    Get the video .m3u8 file path for the specified camera from the ride directory.
    If the ride directory is a zip file, extract the video to a temporary directory.
    """
    recordings_path = ride_path / 'recordings'
    if not recordings_path.exists():
        print(f'No video recordings found in {ride_path}')
        return None

    if camera == 'front_camera':
        suffix = '*uid_s_1000__uid_e_video.m3u8'
    elif camera == 'rear_camera':
        suffix = '*uid_s_1001__uid_e_video.m3u8'
    else:
        raise ValueError(f'Unknown camera: {camera}')
    
    video_path = list(recordings_path.glob(suffix))

    assert video_path, f'No video recordings found for {camera} camera in {ride_path}'
    video_path = video_path[0]

    if isinstance(video_path, zipfile.Path):
        tmp_dir = stack.enter_context(
            tempfile.TemporaryDirectory(dir=tempdir)
        )
        tmp_dir = Path(tmp_dir)
        video_path = extract_zip_video(video_path, tmp_dir)

    return video_path

def extract_zip_video(
        zip_m3u8_path: zipfile.Path,
        dst_dir: Path,
    ):
    video_path = dst_dir / zip_m3u8_path.name

    # Extract the .m3u8 file from the zip archive to a directory
    with zip_m3u8_path.open(mode='rb') as src_bytes, \
        video_path.open('wb') as dst_file:
        shutil.copyfileobj(
            src_bytes,
            dst_file
        )

    ts_paths = get_camera_ts_filepaths(zip_m3u8_path)
    for ts_file in ts_paths:
        with ts_file.open(mode='rb') as src_bytes, \
            (dst_dir / ts_file.name).open('wb') as dst_file:
            shutil.copyfileobj(
                src_bytes,
                dst_file
            )
    return video_path

def get_camera_ts_filepaths(m3u8_path: Path):
    """
    Get the list of video .ts file paths from the m3u8 file.
    """
    base_path = m3u8_path.parent
    with m3u8_path.open('r') as f:
        lines = f.readlines()

    ts_files = []
    for line in lines:
        if line.startswith('#'):
            continue
        # Remove any trailing whitespace
        line = line.strip()
        # Check if the line is a valid .ts file path
        if line.endswith('.ts'):
            ts_files.append(base_path / line)
    
    return ts_files