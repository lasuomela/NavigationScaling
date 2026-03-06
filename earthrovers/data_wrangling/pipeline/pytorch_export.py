"""
Dataset writer that creates a Huggingface dataset from ride episodes and writes it to disk in chunks.
"""
import datasets
import pandas as pd
import numpy as np
from PIL import Image
import subprocess
import tempfile

from pathlib import Path
from contextlib import ExitStack

datasets.disable_caching()

from earthrovers.data_wrangling.pipeline.utils import get_video_path, ZipPath

class DatasetWriter():
    """
    Dataset writer class that accumulates ride episodes and writes them to disk in chunks as Huggingface datasets.
    Suitable for multi-worker usage.
    """
    def __init__(
            self,
            output_path: Path,
            split: str,
            worker_idx: int = 0,
            num_workers: int = 1,
            chunk_size: str = "1GB",
            image_encoding_format: str = "jpeg",
            jpeg_quality: int = 2,
            image_write_width: int = 224,
            image_write_height: int = -1, # Keep the aspect ratio
        ):
        self.output_path = output_path / split
        self.chunk_write_size = datasets.py_utils.convert_file_size_to_int(chunk_size)
        self.image_encoding_format = image_encoding_format
        self.num_workers = num_workers
        self.image_write_width = image_write_width
        self.image_write_height = image_write_height
        self.jpeg_quality = jpeg_quality

        # Create the output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.current_chunk = []
        self.current_chunk_idx = worker_idx
        self.current_chunk_size = 0


    def close(self):
        if len(self.current_chunk) > 0:
            self.write_chunk()
        else:
            print("Dataset writer closed without writing any data.")

    def add_episode(
            self,
            ride_path: Path,
            ride_id: str,
            episode: pd.DataFrame,
            ride_location_cluster: int,
            use_state_estimates: bool = False,
        ):
        """
        Add a ride episode to the current chunk. If the chunk size exceeds the limit, write it to disk.
        """

        # If using state estimates, replace 'gps' and 'compass' columns with data from 'state_estimates
        if use_state_estimates:
            assert 'state_estimate' in episode.columns.get_level_values(0), \
                "Episode does not contain 'state_estimate' data. Please check the episode data."

            episode[('gps', 'latitude')] = episode[('state_estimate', 'latitude')]
            episode[('gps', 'longitude')] = episode[('state_estimate', 'longitude')]
            episode[('compass', 'yaw')] = episode[('state_estimate', 'yaw')]
            
            # Remove the 'state_estimate' columns
            episode = episode.drop(columns='state_estimate', level=0)

        # Add the episode identifier to the dataset
        episode['ride_id'] = ride_id
        episode['ride_location_cluster'] = ride_location_cluster

        # Create a Huggingface dataset from the episode DataFrame
        episode_ds = datasets.Dataset.from_pandas(episode)

        # Find the camera columns
        is_camera = episode.columns.get_level_values(0).str.contains("camera")
        cameras = episode.columns[is_camera].get_level_values(0)

        image_datasets = []
        for camera in cameras:

            # Check if the camera has any nan values
            if episode[(camera, 'frame_id')].isnull().any():
                raise ValueError(f"Camera {camera} has missing frame_ids. Have the episode measurements been subsampled and aligned?")

            # Read the images for the camera at the specified frame indices
            frame_idxs = episode[(camera, 'frame_id')].astype(int)
            ds = self._read_images_ffmpeg(
                ride_path,
                camera,
                frame_idxs,
                encode_format=self.image_encoding_format,
                jpeg_quality=self.jpeg_quality,
            )
            if ds is None:
                print(f"Skipping ride {ride_id} due to corrupt video stream.")
                return
            image_datasets.append(ds)

        # Merge the image datasets with the episode dataset
        episode_ds = datasets.concatenate_datasets([episode_ds] + image_datasets, axis=1)

        # Add the episode dataset to the current chunk
        self.current_chunk.append(episode_ds)
        self.current_chunk_size += episode_ds._estimate_nbytes()

        # If the chunk size exceeds the limit, write it to disk
        if self.current_chunk_size >= self.chunk_write_size:
            self.write_chunk()

    def write_chunk(self):
        chunk = datasets.concatenate_datasets(self.current_chunk)
        filename = f"chunk-{self.current_chunk_idx:04d}.npz"
        npz_path = self.output_path / filename
        chunk.save_to_disk(
            npz_path,
            num_shards=1,
        )

        self.current_chunk_idx += self.num_workers
        self.current_chunk = []
        self.current_chunk_size = 0
        
    def _read_images_ffmpeg(
            self,
            ride_path: Path | ZipPath,
            camera: str,
            frame_idxs: pd.Series,
            encode_format: str = "jpeg",
            jpeg_quality: int = 2,
        ):
        """
        Read frames from a video and write the into a Huggingface dataset. Two output modes:
        - "raw": Extracts raw RGB frames and converts them to PIL Images.
        - "jpeg": Extracts JPEG-encoded frames, and directly turn the bytes into a dataset.
        
        Args:
            ride_path (Path): Path to the ride directory.
            camera (str): Camera identifier.
            frame_idxs (pd.Series): Pandas Series with frame indices (the values) for each time step.
            encode_format (str): Either "raw" for raw RGB frames or "jpeg" for JPEG-encoded frames.
            jpeg_quality (int): (Optional) Quality parameter for JPEG encoding (lower is better quality).
            
        Returns:
            datasets.Dataset: Dataset with the extracted images.
        """
        # Create a list of the desired frame indices (assumed to be in ascending order)
        frame_indices = list(frame_idxs.values)
        
        # Build the ffmpeg filter expression to select only the desired frames.
        # It looks like: select='eq(n\,FRAME1)+eq(n\,FRAME2)+...'
        filter_expr = "+".join(f"eq(n\\,{idx})" for idx in frame_indices)

        # Add resizing and format conversion to the filter expression.
        filter_str = f"select='{filter_expr}',scale={self.image_write_width}:{self.image_write_height},format=rgb24"

        with ExitStack() as stack:

            # Get the video file path
            video_path = get_video_path(
                ride_path,
                camera,
                stack=stack,
                tempdir=Path('/dev/shm/'),
            )

            # Create a temporary file for the filter string
            filter_file = stack.enter_context(
                tempfile.NamedTemporaryFile(mode="w", suffix=".txt")
            )
            filter_file.write(filter_str)
            filter_file.flush()

            if encode_format == "raw":
                # Use raw video output so that ffmpeg writes uncompressed RGB data.
                cmd = [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-filter_complex_script", filter_file.name,
                    "-vsync", "0",         # ensure exactly the selected frames are output
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "pipe:1"
                ]
            elif encode_format == "jpeg":
                # Use the mjpeg to produce JPEG-encoded frames.
                # You can control JPEG quality using the "-q:v" parameter (lower means better quality).
                cmd = [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-filter_complex_script", filter_file.name,
                    "-vsync", "0",         # output exactly the selected frames
                    "-vframes", str(len(frame_indices)),
                    "-q:v", str(jpeg_quality),
                    "-f", "image2pipe",
                    "-vcodec", "mjpeg",
                    "pipe:1"
                ]
            else:
                raise ValueError("Unsupported encode_format. Choose 'raw' or 'jpeg'.")
        
            # Run ffmpeg and capture the output data from stdout.
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                raise ValueError(f"ffmpeg error: {proc.stderr.decode()}")
            data = proc.stdout

        images = []
        if encode_format == "raw":
            # In raw mode, each frame is image_write_width X image_write_height pixels with 3 channels (RGB)
            channels = 3
            frame_size = channels * self.image_write_width * self.image_write_height

            # Process each frame chunk.
            for i in range(0, len(data), frame_size):
                frame_bytes = data[i:i+frame_size]
                # Convert to a numpy array and reshape.
                frame_array = np.frombuffer(
                    frame_bytes,
                    dtype=np.uint8,
                    ).reshape(
                        (self.image_write_height, self.image_write_width, channels)
                )
                image = Image.fromarray(frame_array, 'RGB')
                image.format = self.image_encoding_format
                images.append(image)

            ds = datasets.Dataset.from_dict({
                camera: images,
            })
                
        elif encode_format == "jpeg":
            # In JPEG mode, we need to split the concatenated JPEG byte stream into individual images.
            # JPEG images start with 0xFFD8 and end with 0xFFD9.
            signature = b'\xff\xd8'
            end_marker = b'\xff\xd9'
            pos = 0
            frame_buffers = []
            while True:
                start = data.find(signature, pos)
                if start == -1:
                    break
                end = data.find(end_marker, start)
                if end == -1:
                    break
                # Include the end marker.
                end += len(end_marker)
                frame_buffers.append(data[start:end])
                pos = end
            
            if len(frame_buffers) != len(frame_indices):
                # In this case, the video data is probably corrupted
                return None

            # Convert the list of byte buffers directly into a dataset without decoding.
            ds = datasets.Dataset.from_dict({
                camera: frame_buffers,
            }).cast_column(camera, datasets.Image(decode=False))

        else:
            raise ValueError(f"Unsupported encode_format: {encode_format}")
        
        return ds
