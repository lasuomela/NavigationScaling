"""
A LightningDataModule for imitation learning with demonstrations from the FrodoBots datasets.
"""
from typing import List, Dict
from omegaconf import DictConfig

from pathlib import Path

import torch
import lightning as L
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from earthrovers.train.data_loader.frodobots_dataset import FrodoBotsDataset
from earthrovers.train.data_loader.distributed_sampler import DistributedSubsetSampler

class FrodoBotsDataModule(L.LightningDataModule):

    def __init__(
            self,
            dataloader_config: DictConfig,
            trainer_config: DictConfig,
            img_mean: List[float],
            img_std: List[float],
        ):
        super().__init__()
        self.config = dataloader_config
        self.trainer_config = trainer_config
        self.rgb_normalize_mean = img_mean
        self.rgb_normalize_std = img_std
        self.gpu_augmentation = dataloader_config.gpu_augmentation
        self.is_distributed = (trainer_config.num_nodes > 1) or (trainer_config.num_gpus > 1)

        self._setup_transforms()

        # Only run prepare_data once on rank 0
        self.prepare_data_per_node = False


    def prepare_data(self):
        """
        Build the dataset index.
        """
        # Build the val index
        print("Building val index")
        FrodoBotsDataset.build_index(
            dataset_path=self.config.dataset_path,
            sequence_length=self.config.sequence_length,
            mode='val',
            persist_index=self.config.persist_index,
            num_locations=self.config.val_num_locations,
            max_rides_per_location=self.config.val_max_rides_per_location,
            min_rides_per_location=self.config.val_min_rides_per_location,
            max_hours_per_location=self.config.val_max_hours_per_location,
            min_hours_per_location=self.config.val_min_hours_per_location,
            samples_to_load=self.config.samples_to_load,
            pick_cluster_ids=self.config.val_cluster_ids,
        )

        # Build the train index
        if self.config.train_num_locations != 0:
            print("Building train index")
            FrodoBotsDataset.build_index(
                dataset_path=self.config.dataset_path,
                sequence_length=self.config.sequence_length,
                mode='train',
                persist_index=self.config.persist_index,
                num_locations=self.config.train_num_locations,
                max_rides_per_location=self.config.train_max_rides_per_location,
                min_rides_per_location=self.config.train_min_rides_per_location,
                max_hours_per_location=self.config.train_max_hours_per_location,
                min_hours_per_location=self.config.train_min_hours_per_location,
                samples_to_load=self.config.samples_to_load,
            )
        else:
            if not self.config.train_with_val_set:
                raise ValueError(
                    "train_num_locations is 0 and train_with_val_set is False. "
                    "This would result in an empty train set."
                )
            print("Skipping building train index (train_num_locations=0)")

        if self.config.train_with_val_set:
            # Include the locations from the val set also in the train set
            print("Building train index (for training with validation set locations)")
            FrodoBotsDataset.build_index(
                dataset_path=self.config.dataset_path,
                sequence_length=self.config.sequence_length,
                mode='train_with_val',
                persist_index=self.config.persist_index,
                num_locations=self.config.valtrain_num_locations, # Use all locations. If pick_cluster_ids is set, only those will be used.
                max_rides_per_location=self.config.valtrain_max_rides_per_location,
                min_rides_per_location=self.config.valtrain_min_rides_per_location,
                max_hours_per_location=self.config.valtrain_max_hours_per_location,
                min_hours_per_location=self.config.valtrain_min_hours_per_location,
                samples_to_load=self.config.samples_to_load,
                pick_cluster_ids=self.config.valtrain_cluster_ids,
            )

    def _setup_transforms(self):
        assert self.config.image_size[0] == self.config.image_size[1], \
            "Only square images are supported"
        
        self.val_individual_transform = None
        self.train_individual_transform = T.Compose([
            # T.RandomRotation(degrees=7),
            T.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.1,
            ),
        ])

        train_seq_transform = []
        val_seq_transform = []

        if self.config.image_aspect_ratio_method == "crop":
            train_seq_transform += [T.CenterCrop(self.config.image_size[0])]
            val_seq_transform += [T.CenterCrop(self.config.image_size[0])]
        elif self.config.image_aspect_ratio_method == "resize":
            train_seq_transform += [T.Resize(self.config.image_size)]
            val_seq_transform += [T.Resize(self.config.image_size)]
        else:
            raise ValueError(
                f"Unsupported image aspect ratio method: "
                f"{self.config.image_aspect_ratio_method}"
            )
        
        train_seq_transform += [
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.rgb_normalize_mean, std=self.rgb_normalize_std),
        ]
        val_seq_transform += [
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=self.rgb_normalize_mean, std=self.rgb_normalize_std),
        ]

        train_seq_transform = T.Compose(train_seq_transform)
        val_seq_transform = T.Compose(val_seq_transform)

        if self.gpu_augmentation:
            self.train_seq_transform = None
            self.val_seq_transform = None
            self.gpu_train_transform = train_seq_transform
            self.gpu_val_transform = val_seq_transform
        else:
            self.train_seq_transform = train_seq_transform
            self.val_seq_transform = val_seq_transform
            self.gpu_train_transform = None
            self.gpu_val_transform = None

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Define train and val datasets
            if self.config.train_num_locations != 0:
                self.train_dataset = FrodoBotsDataset(
                    self.config.dataset_path,
                    individual_transform=self.train_individual_transform,
                    sequence_transform=self.train_seq_transform,
                    mode="train",
                    samples_to_load=self.config.samples_to_load,
                    sequence_length=self.config.sequence_length,
                    hflip_augmentation=self.config.hflip_augmentation,
                    random_goal=self.config.random_goal,
                    max_goal_distance=self.config.max_goal_distance,
                    goal_sampling_distribution=self.config.goal_sampling_distribution,
                    persist_index=self.config.persist_index,
                    num_locations=self.config.train_num_locations,
                    max_rides_per_location=self.config.train_max_rides_per_location,
                    min_rides_per_location=self.config.train_min_rides_per_location,
                    max_hours_per_location=self.config.train_max_hours_per_location,
                    min_hours_per_location=self.config.train_min_hours_per_location,
                )
                self._camera_keys = self.train_dataset.get_camera_keys()

            if self.config.train_with_val_set:
                # Load also the val locations into the train set
                train_val_dataset = FrodoBotsDataset(
                    self.config.dataset_path,
                    individual_transform=self.train_individual_transform,
                    sequence_transform=self.train_seq_transform,
                    mode="train_with_val",
                    samples_to_load=self.config.samples_to_load,
                    sequence_length=self.config.sequence_length,
                    hflip_augmentation=self.config.hflip_augmentation,
                    random_goal=self.config.random_goal,
                    max_goal_distance=self.config.max_goal_distance,
                    goal_sampling_distribution=self.config.goal_sampling_distribution,
                    persist_index=self.config.persist_index,
                    num_locations=self.config.valtrain_num_locations, # Use all locations. If pick_cluster_ids is set, only those will be used.
                    max_rides_per_location=self.config.valtrain_max_rides_per_location,
                    min_rides_per_location=self.config.valtrain_min_rides_per_location,
                    max_hours_per_location=self.config.valtrain_max_hours_per_location,
                    min_hours_per_location=self.config.valtrain_min_hours_per_location,
                    pick_cluster_ids=self.config.valtrain_cluster_ids,
                )
                if not hasattr(self, '_camera_keys'):
                    self._camera_keys = train_val_dataset.get_camera_keys()

                if self.config.train_num_locations == 0:
                    self.train_dataset = train_val_dataset
                    rank_zero_info(f"Using only val set locations for training "
                          f"(train dataset size: {len(self.train_dataset)})")
                else:
                    rank_zero_info(f"Adding {len(train_val_dataset)} samples from val set locations to the train dataset "
                        f"(original train dataset size: {len(self.train_dataset)})")
                    # Combine the two datasets
                    self.train_dataset = torch.utils.data.ConcatDataset(
                        [self.train_dataset, train_val_dataset]
                    )
            
            self.val_dataset = FrodoBotsDataset(
                self.config.dataset_path,
                individual_transform=self.val_individual_transform,
                sequence_transform=self.val_seq_transform,
                mode="val",
                samples_to_load=self.config.samples_to_load,
                sequence_length=self.config.sequence_length,
                random_goal=self.config.random_goal,
                max_goal_distance=self.config.max_goal_distance,
                goal_sampling_distribution=self.config.goal_sampling_distribution,
                persist_index=self.config.persist_index,
                num_locations=self.config.val_num_locations,
                max_rides_per_location=self.config.val_max_rides_per_location,
                min_rides_per_location=self.config.val_min_rides_per_location,
                max_hours_per_location=self.config.val_max_hours_per_location,
                min_hours_per_location=self.config.val_min_hours_per_location,
                pick_cluster_ids=self.config.val_cluster_ids,
            )
            
        if stage == "validate":
            self.val_dataset = FrodoBotsDataset(
                self.config.dataset_path,
                individual_transform=self.val_individual_transform,
                sequence_transform=self.val_seq_transform,
                mode="val",
                samples_to_load=self.config.samples_to_load,
                sequence_length=self.config.sequence_length,
                random_goal=self.config.random_goal,
                max_goal_distance=self.config.max_goal_distance,
                goal_sampling_distribution=self.config.goal_sampling_distribution,
                persist_index=self.config.persist_index,
                num_locations=self.config.val_num_locations,
                max_rides_per_location=self.config.val_max_rides_per_location,
                min_rides_per_location=self.config.val_min_rides_per_location,
                max_hours_per_location=self.config.val_max_hours_per_location,
                min_hours_per_location=self.config.val_min_hours_per_location,
                pick_cluster_ids=self.config.val_cluster_ids,
            )
            if len(self.val_dataset) < self.config.batch_size:
                raise ValueError(
                    "Not enough transitions to form a single batch: "
                    f"self.config.batch_size={self.config.batch_size} > "
                    f"len(val_dataset)={len(self.val_dataset)}",
                )         
        if stage == "test":
            raise NotImplementedError("Test stage not implemented")

        if stage == "predict":
            raise NotImplementedError("Predict stage not implemented")
        
    def train_dataloader(self):
        kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "drop_last": True,
            "pin_memory": True,
            "multiprocessing_context": "fork",
        }
        if self.is_distributed:
            kwargs['sampler'] = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                drop_last=True,
            )
        else:
            kwargs['shuffle'] = True

        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self):
        kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "drop_last": True,
            "pin_memory": True,
            "multiprocessing_context": "fork",
        }
        if self.is_distributed:
            kwargs['sampler'] = DistributedSubsetSampler(
                self.val_dataset,
                shuffle=False,
                drop_last=True,
                sample_rate= 1 / self.config.sequence_length,
            )
        else:
            kwargs['shuffle'] = False
        
        return DataLoader(self.val_dataset, **kwargs)
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Apply GPU-side augmentation to the RGB observations if enabled.
        """

        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        if self.gpu_augmentation and isinstance(batch, dict):
            for key in self._camera_keys:
                if self.trainer.training:
                # Apply gpu-side transforms
                    batch[key] = self.gpu_train_transform(batch[key])
                else:
                    batch[key] = self.gpu_val_transform(batch[key])

        return batch

if __name__ == "__main__":

    from earthrovers.train.config.default_structured_configs import DataloaderConfig
    from earthrovers.train.config.default_structured_configs import TrainerConfig

    import earthrovers
    pkg_top_dir = Path(earthrovers.__file__).parent.parent

    dataset_path = pkg_top_dir / "dataset"

    data_module = FrodoBotsDataModule(
        dataloader_config=DataloaderConfig(
            dataset_path=dataset_path,
            batch_size=1,
            num_workers=4,
            samples_to_load=-1,
            sequence_length=5,
            gpu_augmentation=True,
            persist_index=True,
            train_with_val_set=True,
            valtrain_cluster_ids=[0],
        ),
        trainer_config=TrainerConfig(
        ),
        img_mean=[0.5, 0.5, 0.5],
        img_std=[0.5, 0.5, 0.5],
    )
    data_module.prepare_data()
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()

    import tqdm
    for batch in tqdm.tqdm(train_loader):
        print(batch.keys())
        break
