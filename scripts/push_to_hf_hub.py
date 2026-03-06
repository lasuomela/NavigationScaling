#!/usr/bin/env python
"""
Load a Lightning checkpoint, extract the underlying PyTorch model,
and push it to the Hugging Face Hub.
"""
from pathlib import Path
import yaml
import argparse

from earthrovers.train.models.lightning_wrappers import LightningMLPModel
from earthrovers.train.models.lightning_wrappers import LightningViNTModel
from earthrovers.train.models.lightning_wrappers import LightningNoMADModel
from earthrovers.train.models.lightning_wrappers import LightningDiffusionUnetModel

TYPE_MAP = {
    "mlp_policy": LightningMLPModel,
    "vint_policy": LightningViNTModel,
    "nomad_policy": LightningNoMADModel,
    "diffusion_policy": LightningDiffusionUnetModel,
}

def main(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config[args.model_name]

    # Load the Lightning checkpoint
    lightning_model = TYPE_MAP[model_config['model_type']].load_from_checkpoint(
        args.ckpt_dir / model_config["ckpt_path"],
        weights_only=False,
    )

    # Extract the underlying PyTorch model
    pytorch_model = lightning_model.model

    model_hf_path = args.repo_name + "/" + args.model_hf_name

    # Push the model to the Hugging Face Hub
    pytorch_model.push_to_hub(
        model_hf_path
    )

    if args.export_config_path is not None:
        export_config = dict(model_config)
        export_config.pop("ckpt_path", None)
        export_config["model_hf_name"] = model_hf_path

        existing_export_config = {}
        if args.export_config_path.exists():
            with open(args.export_config_path, "r") as f:
                existing_export_config = yaml.safe_load(f) or {}

        existing_export_config[args.model_hf_name] = export_config

        with open(args.export_config_path, "w") as f:
            yaml.dump(existing_export_config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a model to the Hugging Face Hub.")
    parser.add_argument("--ckpt_dir", type=Path, required=True, help="Path to the directory containing the Lightning checkpoints.")
    parser.add_argument("--config_path", type=Path, required=True, help="Path to the YAML configuration file for the model.")
    parser.add_argument("--export_config_path", type=Path, required=False, help="Path to save the export configuration file.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model in the configuration file to load.")
    parser.add_argument("--repo_name", type=str, help="Name of the Hugging Face repository to push to.")
    parser.add_argument("--model_hf_name", type=str, required=True, help="Name of the model to use in the Hugging Face Hub.")
    args = parser.parse_args()

    main(args)
