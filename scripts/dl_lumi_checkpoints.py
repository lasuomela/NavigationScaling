"""
Code to download the checkpoints for a specified deployment model configuration.
Useful for pulling trained chekpoints from a cluster to a local machine for deployment.
"""

import argparse
import os
import yaml
from pathlib import Path


def download_checkpoints(model_config, src_model_dir, output_dir):
    """
    Downloads the checkpoints for a specified deployment model configuration.

    Args:
        model_config (str): The model configuration to download checkpoints for.
        output_dir (str): The directory to save the downloaded checkpoints.
    """
    src_model_dir = Path(src_model_dir)
    output_dir = Path(output_dir)

    # Load the configuration file
    with open(model_config, "r") as file:
        config = yaml.safe_load(file)

    for model, attrs in config.items():
        checkpoint_path = attrs['ckpt_path']
        wnb_code = checkpoint_path.split("/")[0]
        ckpt_id = checkpoint_path.split("/")[1]
        full_checkpoint_path = src_model_dir / wnb_code / 'checkpoints' / ckpt_id
        full_output_dir = output_dir / wnb_code
        full_output_dir.mkdir(parents=True, exist_ok=True)

        if (full_output_dir / ckpt_id).exists():
            print(f"Checkpoint for {model} already exists at {full_output_dir / ckpt_id}, skipping download.")
            continue

        # Command to copy the checkpoint from the source to the destination
        print(f"Downloading checkpoint for {model} from {full_checkpoint_path} to {full_output_dir}")
        os.system(f"scp {full_checkpoint_path} {full_output_dir}")

if __name__ == "__main__":

    import earthrovers
    pkg_top_dir = Path(earthrovers.__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Download Lightning checkpoints for a specified deployment model configuration, such as the one in earthrovers/deployment/src/earthrovers_deployment/config/local_config.yaml"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="The model configuration to download checkpoints for.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default= pkg_top_dir / "earthrovers/deployment/src/earthrovers_deployment/model_weights",
        help="The directory to save the downloaded checkpoints.",
    )
    parser.add_argument(
        "--src_model_dir",
        type=str,
        help="The source directory of the model checkpoints (on a remote machine).",
    )
    
    args = parser.parse_args()
    download_checkpoints(args.model_config, args.src_model_dir, args.output_dir)