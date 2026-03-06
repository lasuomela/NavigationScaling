#!/bin/bash

# Launch training script on a local machine
DATASET_PATH=$1

WANDB_MODE=disabled python -m earthrovers.train.run \
        --config-name=experiments/generalization/full.yaml \
        earthrovers.dataloader.dataset_path=$DATASET_PATH \
        earthrovers.dataloader.num_workers=1 \
        earthrovers.dataloader.batch_size=1 \
        earthrovers.model.freeze_encoder=True \
