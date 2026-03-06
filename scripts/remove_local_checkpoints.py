
import argparse
import shutil
import yaml
from pathlib import Path

def remove_checkpoints(model_config, checkpoint_base_dir):

    model_config = Path(model_config)
    if not model_config.exists():
        print(f"Model configuration file {model_config} does not exist.")
        return

    checkpoint_base_dir = Path(checkpoint_base_dir)
    if not checkpoint_base_dir.exists():
        print(f"Checkpoint directory {checkpoint_base_dir} does not exist.")
        return
    
    # Load the config file
    with open(model_config, 'r') as f:
        config = yaml.safe_load(f)

    for model, attrs in config.items():
        checkpoint_path = attrs['ckpt_path']
        checkpoint_id = Path(checkpoint_path).parent
        checkpoint_dir = ((checkpoint_base_dir / checkpoint_id))
        
        if checkpoint_dir.exists() and checkpoint_dir.is_dir():
            print(f"Removing checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)


if __name__ == "__main__":

    import earthrovers
    pkg_top_dir = Path(earthrovers.__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Remove local checkpoints listed in a deployment policy config file."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="The model configuration that specifies the checkpoints to remove.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default= pkg_top_dir / "earthrovers/deployment/src/earthrovers_deployment/model_weights",
        help="The directory to containing the checkpoints to remove.",
    )
    
    args = parser.parse_args()
    remove_checkpoints(args.model_config, args.checkpoint_dir)