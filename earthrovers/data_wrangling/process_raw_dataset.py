"""
Process the raw FrodoBots dataset by loading, filtering, refining, and exporting the data.
Supports both single-process and multi-process execution.
"""
from typing import List, Tuple
from queue import Queue

import argparse
import traceback
import multiprocessing as mp
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from omegaconf import OmegaConf, open_dict, DictConfig

from earthrovers.data_wrangling.assets.config import Config
from earthrovers.data_wrangling.pipeline.data_loading import load_raw_data
from earthrovers.data_wrangling.pipeline.data_filtering import filter_raw_data
from earthrovers.data_wrangling.pipeline.data_refinement import refine_raw_data
from earthrovers.data_wrangling.pipeline.data_export import align_and_subsample_data
from earthrovers.data_wrangling.pipeline.pytorch_export import DatasetWriter
from earthrovers.data_wrangling.pipeline.rerun_export import export_to_rerunio
from earthrovers.data_wrangling.pipeline.train_val_split import train_val_split
from earthrovers.data_wrangling.pipeline.utils import list_rides, ZipPath, fast_reconstruct_zip_path_list, split_zip_path_list

def process_rides_batch(
        worker_idx: int,
        args: argparse.Namespace,
        ride_batch: List[Tuple[str, Path]],
        split: str,
        pbar: tqdm = None,
        progress_queue: Queue = None,
        error_queue: Queue = None,
    ):
    """Processes a batch of rides assigned to a single worker."""
    dataset_writer = None
    if progress_queue is not None:
        parent = mp.parent_process()
    else:
        parent = None
    try:
        if args.pytorch_export:
            dataset_writer = DatasetWriter(
                output_path=Path(args.dataset_output_path),
                split=split,
                worker_idx=worker_idx,
                num_workers=args.num_workers,
                image_encoding_format='jpeg',
                chunk_size=args.hf_chunk_size,
                image_write_height=args.image_height,
                image_write_width=args.image_width,
            )

        if args.zipped and not isinstance(ride_batch[0][1], ZipPath):
            # Reconstruct the ZipPath from the tuple (ride_id, (filename, at))
            ride_batch = fast_reconstruct_zip_path_list(ride_batch)

        for ride_id, ride_path, ride_location_cluster in ride_batch:
            if parent is not None:
                if not parent.is_alive():
                    break

            ride_data = load_raw_data(ride_path, args.data_types_to_load, verbose=args.verbose)

            if args.filter_data:
                episodes = filter_raw_data(
                    ride_data,
                    ride_id,
                    video_sampling_frequency=args.subsample_frequency_hz,
                    remove_stationary_images=True,
                    optimize_trajectory_poses=args.sensor_fusion,
                    fuse_magnetometer=True,
                    verbose=args.verbose,
                )
            else:
                episodes = [ride_data]  # If no filtering, treat the whole ride as a single episode

            if args.refine_data:
                for i in range(len(episodes)):
                    episodes[i] = refine_raw_data(
                        episodes[i],
                        correct_declination=False,
                        use_state_estimates=args.sensor_fusion,
                    )

            if args.align_and_subsample:
                for i in range(len(episodes)):
                    episodes[i] = align_and_subsample_data(
                        ride_data = episodes[i],
                        control_frequency=args.action_chunk_frequency_hz,
                        control_horizon_length=args.action_chunk_length_s,
                    )

            if args.pytorch_export and dataset_writer:
                for i in range(len(episodes)):
                    dataset_writer.add_episode(
                        ride_path,
                        ride_id,
                        episodes[i],
                        ride_location_cluster=ride_location_cluster,
                        use_state_estimates=args.sensor_fusion,
                    )

            if args.rerun_export:
                for i in range(len(episodes)):
                    export_to_rerunio(episodes[i], ride_path, ride_id + f'_episode_{i}', args.rerun_output_path)

            if progress_queue is not None:
                progress_queue.put(1)  # Signal progress update if multiprocessing
            
            if pbar is not None:
                pbar.update(1) # Update progress bar if running in single process

        if dataset_writer is not None:
            dataset_writer.close()  # Close dataset writer at the end of worker's batch

    except Exception as e:
        if error_queue is not None:
            # Put exception and stacktrace into the error queue to be raised in the main process
            error_queue.put((e, traceback.format_exc()))
        else:
            raise e
        
def process_rides(
        args: DictConfig,
        ride_paths: pd.DataFrame,
        split: str,
) -> None:
    """
    Processes rides either in single-process or multi-process mode based on args.num_workers.
    """

    # Convert DataFrame to list of tuples (ride_id, ride_path, cluster)
    ride_list = list(zip(ride_paths.index, ride_paths['ride_path'], ride_paths['cluster']))
    num_workers = args.num_workers if args.num_workers >= 0 else mp.cpu_count()

    # If the ride_paths are ZipPaths, convert into Tuple(Path, at)
    # and reconstruct in worker process to avoid pickling issues
    if args.zipped:
        ride_list = split_zip_path_list(ride_list)

    # Main processing loop
    with tqdm(total=len(ride_list), desc=f"Processing {split} rides") as pbar:

        # Single-process mode
        if args.num_workers == 0:
            print("Running in single-process mode...")
            process_rides_batch(0, args, ride_list, split, pbar=pbar)

        # Multi-process mode
        else:
            print(f"Running with {num_workers} parallel workers...")

            # Split rides into approximately equal batches for each worker
            batches = [ride_list[i::num_workers] for i in range(num_workers)]

            ctx = mp.get_context("spawn")
            with ctx.Manager() as manager:
                # Process communication queues
                progress_queue = manager.Queue()
                error_queue = manager.Queue()

                with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                    futures = [
                        executor.submit(
                            process_rides_batch,
                            i,
                            args,
                            batch,
                            split,
                            progress_queue=progress_queue,
                            error_queue=error_queue,
                        )
                    for i, batch in enumerate(batches)]

                    # Monitor updates from each worker
                    while any(f.running() for f in futures):
                        try:
                            pbar.update(progress_queue.get(timeout=1))
                        except mp.queues.Empty:
                            pass

                        # Print worker stacktrace if any error occurred
                        if not error_queue.empty():
                            e, stacktrace = error_queue.get_nowait()
                            executor.shutdown(wait=False, cancel_futures=True)
                            print(stacktrace)
                            raise e
                        
                    if not error_queue.empty():
                        e, stacktrace = error_queue.get_nowait()
                        print(stacktrace)
                        raise e


def main(args):

    # Get available rides
    with open_dict(args):
        ride_paths, args.zipped = list_rides(args.dataset_path)
    assert not ride_paths.empty, f'No rides found in {args.dataset_path}'

    # Split into train and val if creating a pytorch dataset
    if (args.ride_id is None) and args.pytorch_export:
        splits = train_val_split(ride_paths, val_locations=args.val_locations, fetch_location_name=args.fetch_location_names)
        total_rides = len(splits['train'])+ len(splits['val'])
        print(f"Split {total_rides} rides to {len(splits['train'])/total_rides:.2f} train and {len(splits['val'])/total_rides:.2f} val")
    else:
        # Add empty cluster idx column for consistency
        ride_paths['cluster'] = None
        splits = {'all': ride_paths}
        
    # Process all splits
    for split, rides in splits.items():
        print(f'Processing {split} split')
        ride_paths = rides[['ride_path', 'cluster']]

        # If a specific ride_id is given, pick only that ride
        if args.ride_id is not None:
            assert args.ride_id in ride_paths.index, f'Ride {args.ride_id} not found in {args.dataset_path}. Available rides: {ride_paths.index}'
            ride_paths = ride_paths.loc[[args.ride_id]]

        else:
            assert args.num_rides <= len(ride_paths), f'num_rides must be less or equal to the number of rides in the dataset'

            if args.random_rides:
                ride_paths = ride_paths.sample(n=args.num_rides)
            else:
                ride_paths = ride_paths.head(args.num_rides)

        # Process the selected rides
        process_rides(args, ride_paths, split)


if __name__ == '__main__':
    cfg = OmegaConf.structured(Config)
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli)
    main(cfg)
