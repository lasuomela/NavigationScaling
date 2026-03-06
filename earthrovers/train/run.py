import os
import hydra
from omegaconf import DictConfig, OmegaConf

from earthrovers.train.train_utils.train import train

@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the training process.

    Args:
        cfg (DictConfig): Configuration object containing training parameters.
    """
    # Set up the training process
    train(cfg)

if __name__ == "__main__":
    main()