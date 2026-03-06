"""
Class for training a policy using PyTorch Lightning.
"""
import torch
import lightning as L
import math
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, TQDMProgressBar

from earthrovers.train.data_loader.data_module import FrodoBotsDataModule
from earthrovers.train.train_utils.utils import LogValImgsCallback, LogTrainImgsCallback
from earthrovers.common.constants import IMAGE_NORMALIZATION_VALUES
from earthrovers.common.utils import get_encoder_img_preprocess_type
from earthrovers.train.config.registry import registry
import earthrovers.train.models.lightning_wrappers # This import is necessary to register the models


# Get the top directory of the project
import earthrovers
PROJECT_DIR = Path(earthrovers.__file__).parent.parent

def train(config: DictConfig):
    torch.set_float32_matmul_precision('high')

    img_preprocess_type = get_encoder_img_preprocess_type(
        config.earthrovers.model.encoder_type
    )
    IMG_MEAN = IMAGE_NORMALIZATION_VALUES[img_preprocess_type]['mean']
    IMG_STD = IMAGE_NORMALIZATION_VALUES[img_preprocess_type]['std']

    # Create a Wandb logger
    logger = WandbLogger(
        project=config.earthrovers.trainer.wandb.project,
        entity=config.earthrovers.trainer.wandb.entity,
        group=config.earthrovers.trainer.wandb.group,
        name=config.earthrovers.trainer.wandb.run_name,
        save_dir=PROJECT_DIR / 'logs',
    )

    # Define the training callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor=config.earthrovers.trainer.checkpoint_metric,
            save_top_k=config.earthrovers.trainer.num_checkpoints,
            mode=config.earthrovers.trainer.checkpoint_metric_mode,
            save_last=True,
            filename='model-{epoch:02d}-{step}-{val_loss:.4f}',
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
        ModelSummary(max_depth=2),
        TQDMProgressBar(refresh_rate=1, leave=True),
    ]
    if config.earthrovers.trainer.log_train_images:
        callbacks += [
        LogTrainImgsCallback(
            IMG_MEAN,
            IMG_STD,
            log_trigger='step',
            num_batches_to_log=100,
        )]
    if config.earthrovers.trainer.log_val_images:
        callbacks += [
        LogValImgsCallback(
            IMG_MEAN,
            IMG_STD,
            log_trigger='epoch',
        )
        ]

    # Create a PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=config.earthrovers.trainer.num_epochs,
        num_nodes=config.earthrovers.trainer.num_nodes,
        devices=config.earthrovers.trainer.num_gpus,
        accelerator=config.earthrovers.trainer.accelerator,
        logger=logger,
        callbacks=callbacks,
        strategy=config.earthrovers.trainer.strategy,
        val_check_interval=1 / math.sqrt(config.earthrovers.model.sequence_length),
        gradient_clip_val=config.earthrovers.trainer.gradient_clip_val,
        use_distributed_sampler=False, # Do this manually in the dataloader
    )

    if trainer.is_global_zero:
        print('Training with the config:')
        print(OmegaConf.to_yaml(config, resolve=True))
    else:
        # Avoid loading the pretrained weights on non-zero nodes
        # since ddp will distribute the model to all nodes
        if not config.earthrovers.model.freeze_encoder:
            config.earthrovers.model.pretrained_encoder = False

    # Create the lightning data module
    datamodule = FrodoBotsDataModule(
        dataloader_config=config.earthrovers.dataloader,
        trainer_config=config.earthrovers.trainer,
        img_mean=IMG_MEAN,
        img_std=IMG_STD,
    )

    # Get the model class
    Model = registry.get_model(
        config.earthrovers.model,
    )
    # Instantiate the model
    model = Model(config.earthrovers.model)

    # Train the model
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == '__main__':
    train()