"""
Default configuration for the pipeline to filter the raw FrodoBots datasets
and export them as .rrd for Rerun visualization or as a HuggingFace dataset
for PyTorch training.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from omegaconf import MISSING

import earthrovers
pkg_top_dir = Path(earthrovers.__file__).parent.parent

@dataclass
class Config:
    # Paths
    dataset_path: Path = MISSING                          # Path to the raw FrodoBots dataset, has to be set by the user.
    rerun_output_path: Path = pkg_top_dir / "rerun"       # Output folder to save the visualization .rrd's
    dataset_output_path: Path = pkg_top_dir / "dataset"   # Output folder to save the processed dataset.

    # Miscellaneous options
    ride_id: Optional[str] = None       # Optional: If specified, only process this ride ID.
    num_rides: int = -1                 # Limit the number of rides to process. -1 for all rides.
    random_rides: bool = False          # Select the 'num_rides' randomly instead of sequentially.
    num_workers: int = 20                # Number of parallel workers to use. 0 to run everything in main process.
    fetch_location_names: bool = False  # Whether to fetch human-readable location names for the ride clusters.
    verbose: bool = False               # Whether to print information about the filtering process.

    # Pipeline parts to run
    sensor_fusion: bool = True          # Try to improve the state estimates using sensor fusion with GTSAM.
    filter_data: bool = True            # Filter out rides with missing sensor measurements.
    refine_data: bool = True            # Process the data to determine navigation segments, calculate yaw, etc.
    align_and_subsample: bool = True    # Align all measurements to video timestamps subsampled to 4 Hz.

    # Export options
    rerun_export: bool = False      # Export the processed rides as .rrd files for Rerun visualization.
    pytorch_export: bool = True    # Export the processed rides as a Huggingface Datasets artifact for training with PyTorch.
    hf_chunk_size: str = "10GB"     # Chunk size for writing the Dataset to disk. Script peak RAM usage is num_workers * chunk size.

    # Names of the sensors to load
    data_types_to_load: List[str] = field(default_factory=lambda: 
        [
            'gps',
            'magnetometer',
            'front_camera',
            'cmd',
        ]
    )
    
    # GPS coordinates (lat, lon) of locations to assign to validation set
    val_locations: List[List[float]] = field(default_factory=lambda:
        [
            [-0.110523, 34.751274],     # Jamhuri Gardens, Kisumu, Kenya
            [-24.690128, 25.879688],    # Orange digital center, Gaborone, Botswana
            [19.570339, -98.814499],    # Tepetlaoxtoc de Hidalgo, Mexico
            [30.482447, 114.302673],    # Wuhan, China
            [-21.985817, 27.836973],    # Selebi-Phikwe, Botswana
            [-20.17403, 57.500015],     # Port Louis, Mauritius
        ]
    )

    # Measurement processing parameters. Note: values outside these defaults have not been tested.
    #
    image_width: int = 224                      # Width to which camera images are resized.
    image_height: int = 224                     # Height to which camera images are resized.
    subsample_frequency_hz: float = 4.0         # Frequency to which data is subsampled (in Hz).
    action_chunk_frequency_hz: float = 10.0     # Frequency to chunk actions/commands (in Hz).
    action_chunk_length_s: float = 1.0          # Length of each action/command chunk (in seconds).
