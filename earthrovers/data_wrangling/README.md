# Data preparation
This module contains scripts to prepare and preprocess the FrodoBots datasets for training with PyTorch, and for visualizing the data with [Rerun.io](https://rerun.io/).

## Downloading the dataset
The datasets can be downloaded using the download_FBD.py script found in the `earthrovers/data_wrangling/scripts/` directory. It supports multithreaded downloads and resuming interrupted downloads.

To use the script, first download the `.csv` file containing the dataset URLs, and place it in `earthrovers/data_wrangling/assets/`. Then run the script as follows:

```python
python download_FBD.py --save_dir /path/to/save/dataset --csv_file /path/to/csv --parallel_downloads 20
```

The `.csv` file for the FrodoBots2K dataset is freely available on [Huggingface](https://huggingface.co/datasets/frodobots/FrodoBots-2K). If you are interested in the larger FrodoBots8K dataset used in this paper, please contact `hello@frodobots.com`. The recently released [Berkeley-Frodobots7K](https://huggingface.co/datasets/frodobots/Berkeley-FrodoBots-7K) datasets are NOT raw datasets, and not compatible with this data wrangling pipeline.

## Data wrangling pipeline
`process_raw_dataset.py` provides a pipeline to process the raw data into Huggingface Dataset format suitable for training with PyTorch. The rides are clustered into distincs geographical locations by the GPS. To process the whole dataset, run the following command:

```python
python process_raw_dataset.py \
    --dataset_path=/path/to/raw/dataset \
    --dataset_output_path=/path/to/save/processed/dataset \
    --num_workers=20 \
    --hf_chunk_size=10GB
```

See `earthrovers/data_wrangling/assets/config.py` for the full list of configurable parameters. Adjust `num_workers` and `hf_chunk_size` according to your machine - peak RAM consumption is `num_workers * hf_chunk_size`. Using 20 workers on a single node, we were able to process the entire dataset in less than 5 hours.

## Visualizing the data with Rerun
You can also visualize the processed rides using Rerun.io. E.g.

```python
python process_raw_dataset.py \
    dataset_path=/path/to/raw/dataset \
    ride_id=ride_20735_ee171f_20240311030537 \
    rerun_export=True \
    pytorch_export=False \
    num_workers=0
```

will write the specified ride as an .rrd file that can be opened with Rerun into `NavigationScaling/rerun`. Instead of specifying a single ride ID, you can also set `num_rides=20` with optional `random_rides=True` to export multiple rides.

## Dataset format
The processed dataset is stored as a Huggingface Dataset. The underlying data files are stored in a column-first tabular format based on Apache Arrow, which allows efficient loading and processing of large datasets. Each row in the dataset corresponds to sensor measurements for a single timestamp from a single ride, following the high-level schema below:

| ride_id | timestamp | sensor_0 | ... | sensor_N |
|---------|-----------|----------|-----|----------|
| 0       | 0         |          |     |          |
| ...     |           |          |     |          |
| 0       | N         |          |     |          |
| 1       | 0         |          |     |          |
| ...     |           |          |     |          |
| 1       | N         |          |     |          |

You can fetch data from a processed dataset like this:

```python
from datasets import load_from_disk
from pathlib import Path

dataset_path = Path("/path/to/processed/dataset")
dataset = load_from_disk(dataset_path / "train" / "chunk-0000.npz")

# Get the first row
row = dataset[0]
print(row.keys())  # List available columns

# Get timestamps of the first 10 rows as numpy array
timestamps = dataset.select_columns("timestamp").with_format(type='numpy')
print(timestamps[:10])
```

For actual usage with PyTorch, e.g. indexing to fetch observation sequences that don't cross ride boundaries, see `earthrovers/train/data_loader/`.

<details><summary><b><font size="+1">Full dataset schema</font></b></summary><br/>

The columns available in each row of the processed dataset are as follows:

```json
  "features": {
    # Current position GPS latitude
    "('gps', 'latitude')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # Current position GPS longitude
    "('gps', 'longitude')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # Current compass yaw in radians
    "('compass', 'yaw')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # Linear forward velocity command issued by player, normalized to [0, 1]
    "('control', 'linear')": {
      "feature": {
        "dtype": "float64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    # Angular velocity command issued by player, normalized to [-1, 1]
    "('control', 'angular')": {
      "feature": {
        "dtype": "float64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    # Unique identifier for the ride
    "('ride_id', '')": {
      "dtype": "string",
      "_type": "Value"
    },
    # Cluster ID corresponding to the geographical location of the ride
    "('ride_location_cluster', '')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # Timestamp of the current row
    "timestamp": {
      "dtype": "timestamp[ns]",
      "_type": "Value"
    },
    # JPEG-encoded image bytes from the front camera
    "front_camera": {
      "decode": false,
      "_type": "Image"
    },
    # Running idx of the image frame within the original ride video
    "('front_camera', 'frame_id')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # Current navigation objective timestamp
    "('navigation_objective', 'timestamp')": {
      "dtype": "timestamp[ns]",
      "_type": "Value"
    },
    # Current navigation objective GPS latitude
    "('navigation_objective', 'lat')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # Current navigation objective GPS longitude
    "('navigation_objective', 'lon')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # Timestamp of a navigation objective <30m ahead
    "('navigation_objective_30.0m', 'timestamp')": {
      "dtype": "timestamp[ns]",
      "_type": "Value"
    },
    # Timestamp of a navigation objective <150m ahead
    "('navigation_objective_150.0m', 'timestamp')": {
      "dtype": "timestamp[ns]",
      "_type": "Value"
    },
    # GPS latitude of a navigation objective <150m ahead
    "('navigation_objective_150.0m', 'lat')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # GPS longitude of a navigation objective <150m ahead
    "('navigation_objective_150.0m', 'lon')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # GPS latitude of a navigation objective <30m ahead
    "('navigation_objective_30.0m', 'lat')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # GPS longitude of a navigation objective <30m ahead
    "('navigation_objective_30.0m', 'lon')": {
      "dtype": "float64",
      "_type": "Value"
    },
    # The number of rows ahead from current row to the row with the navigation objective timestamp
    "('navigation_objective', 'rows_ahead')": {
      "dtype": "int64",
      "_type": "Value"
    },
    # The number of rows ahead from current row to a row with the navigation objective <30m ahead
    "('navigation_objective_30.0m', 'rows_ahead')": {
      "dtype": "int64",
      "_type": "Value"
    },
    # The number of rows ahead from current row to a row with the navigation objective <150m ahead
    "('navigation_objective_150.0m', 'rows_ahead')": {
      "dtype": "int64",
      "_type": "Value"
    },
    # Timestamps of future 'waypoint' locations
    "('waypoints', 'timestamps')": {
      "feature": {
        "dtype": "timestamp[ns]",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    # X coordinates (meters, relative to current pose) of future 'waypoint' locations
    "('waypoints', 'x')": {
      "feature": {
        "dtype": "float64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    # Y coordinates (meters, relative to current pose) of future 'waypoint' locations
    "('waypoints', 'y')": {
      "feature": {
        "dtype": "float64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
    # Yaw angles (radians, relative to current pose) of future 'waypoint' locations
    "('waypoints', 'yaw')": {
      "feature": {
        "dtype": "float64",
        "_type": "Value"
      },
      "_type": "Sequence"
    },
  }
  ```
  </details>
