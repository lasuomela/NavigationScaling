from typing import Tuple, List

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import II
from pathlib import Path

@dataclass
class WandbConfig:
    """Configuration for Weights and Biases."""
    project: str = "earthrovers"
    entity: str = ""
    run_name: str = ""
    group: str = ""

@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""
    num_nodes: int = 1
    num_gpus: int = 1
    accelerator: str = "gpu"
    num_epochs: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine" # (cosine | "")
    gradient_clip_val: float = 0.5
    strategy: str = "ddp" # Distributed Data Parallel

    num_checkpoints: int = 2 # Number of top checkpoints to keep
    checkpoint_metric: str = "val_loss"
    checkpoint_metric_mode: str = "min"

    wandb: WandbConfig = WandbConfig()
    log_train_images: bool = True
    log_val_images: bool = True

@dataclass
class DataloaderConfig:
    """Configuration for the Dataset."""
    dataset_path: Path = Path("")
    image_size: Tuple = (224, 224)
    image_aspect_ratio_method: str = "resize" # (resize | crop)
    batch_size: int = 64
    num_workers: int = 4
    samples_to_load: int = -1 # Can be used to limit the dataset size for debugging
    sequence_length: int = II("earthrovers.model.sequence_length") # Number of frames in each sequence
    gpu_augmentation: bool = True # Use GPU augmentation
    hflip_augmentation: bool = True # Use horizontal flip augmentation
    random_goal: bool = True # Use random goal augmentation
    max_goal_distance: float = 150.0 # Maximum distance to the goal. Currently needs to be in [10, 150] with interval of 10. See data_wrangling/data_refinement.py for how to change this.
    goal_sampling_distribution: str = "beta" # beta | uniform
    persist_index: bool = True # Whether to persist the index between runs
    
    train_num_locations: int = -1 # Number of locations from which to sample for training
    train_max_rides_per_location: int = -1 # Maximum number of rides per location for training
    train_min_rides_per_location: int = -1 # Minimum number of rides per location for training
    train_max_hours_per_location: float = -1 # Maximum number of hours per location for training
    train_min_hours_per_location: float = -1 # Minimum number of hours per location for training

    val_num_locations: int = -1 # Number of locations from which to sample for validation
    val_cluster_ids: List[int] = field(default_factory=lambda: [-1]) # The val clusters to use. Default: use all validation clusters. This will override val_num_locations if set to specific clusters.
    val_max_rides_per_location: int = -1 # Maximum number of rides per location for validation
    val_min_rides_per_location: int = -1 # Minimum number of rides per location for validation
    val_max_hours_per_location: float = -1 # Maximum number of hours per location for validation
    val_min_hours_per_location: float = -1 # Minimum number of hours per location for validation

    train_with_val_set: bool = False # Whether to include the validation set locations also in the training set
    valtrain_cluster_ids: List[int] = field(default_factory=lambda: [-1]) # The val clusters to train on. Default: use all validation clusters.
    valtrain_num_locations: int = -1 # Number of locations from which to sample for training with validation set
    valtrain_max_rides_per_location: int = -1 # Maximum number of rides per location for training with validation set
    valtrain_min_rides_per_location: int = -1 # Minimum number of rides per location for training with validation set
    valtrain_max_hours_per_location: float = -1 # Maximum number of hours per location for training with validation set
    valtrain_min_hours_per_location: float = -1 # Minimum number of hours per location for training with validation set


@dataclass
class BaseModelConfig:
    """Base configuration for models."""
    type: str = ""
    encoder_type: str = ""
    pretrained_encoder: bool = True
    freeze_encoder: bool = False

    loss_type: str = "scaled_mse" # Loss function to use

    goal_input_dim: int = 2 # Parametrize goal as distance and direction to the goal
    num_action_parameters: int = 2 # Predict linear and angular velocity
    num_prediction_steps: int = 10 # How many actions to predict

    # Get these from the final parsed config
    input_size: Tuple = II("earthrovers.dataloader.image_size")
    learning_rate: float = II("earthrovers.trainer.learning_rate")
    lr_scheduler: str = II("earthrovers.trainer.lr_scheduler")
    num_epochs: int = II("earthrovers.trainer.num_epochs")

@dataclass
class MLPModelConfig(BaseModelConfig):
    """Configuration for a MLP model."""
    type: str = "mlp"
    encoder_type: str = "theaiinstitute/theia-base-patch16-224-cddsv"
    sequence_length: int = 1
    goal_output_dim: int = 64
    num_flatten_channels: int = 6
    embed_goal: bool = False
    state_fusion_type: str = "mlp" # (mlp | patch_transformer)
    action_decoder_type: str = "mlp" # (mlp | transformer)

@dataclass
class TransformerModelConfig(BaseModelConfig):
    """Configuration for a Transformer model."""
    type: str = "deepseek"
    encoder_type: str = "theaiinstitute/theia-small-patch16-224-cddsv"
    sequence_length: int = 1

@dataclass
class TransformerV2ModelConfig(BaseModelConfig):
    """Configuration for a TransformerV2 model."""
    type: str = "deepseek_big"
    encoder_type: str = "theaiinstitute/theia-base-patch16-224-cddsv"
    sequence_length: int = 100
    goal_output_dim: int = 64
    num_flatten_channels: int = 6
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    intermediate_size: int = 2048
    qk_rope_head_dim: int = 32
    qk_nope_head_dim: int = 64
    v_head_dim: int = 64
    kv_lora_rank: int = 256
    q_lora_rank: int = 512
    predict_waypoints: bool = False
    num_waypoints: int = 10
    waypoint_loss_alpha: float = 0.05
    late_goal_fusion: bool = False

@dataclass
class DiffusionUnetModelConfig(MLPModelConfig):
    """Configuration for a Diffusion UNet model."""
    type: str = "diffusion_unet"
    loss_type: str = "mse"
    diffusion_unet_step_embed_dim: int = 128
    diffusion_unet_down_dims: Tuple[int, ...] = (64, 128, 256)
    diffusion_unet_kernel_size: int = 5
    diffusion_unet_num_groups: int = 8
    num_train_diffusion_steps: int = 100
    num_viz_actions: int = 6
    num_viz_diffusion_steps: int = 10
    parametrization: str = "eps" # (eps | cfm)
    use_goal: bool = True

@dataclass
class ViNTModelConfig(BaseModelConfig):
    """Configuration for a ViNT-like model."""
    type: str = "vint"
    loss_type: str = "scaled_mse"
    encoder_type: str = "theaiinstitute/theia-base-patch16-224-cddsv"
    sequence_length: int = 6
    num_flatten_channels: int = 6

@dataclass
class NoMADModelConfig(BaseModelConfig):
    """Configuration for a NoMAD-like model."""
    type: str = "nomad"
    loss_type: str = "mse"
    encoder_type: str = "theaiinstitute/theia-base-patch16-224-cddsv"
    sequence_length: int = 6
    num_flatten_channels: int = 6
    num_train_diffusion_steps: int = 100
    num_viz_actions: int = 6
    num_viz_diffusion_steps: int = 10
    parametrization: str = "eps" # (eps)

@dataclass
class DiffusionSequenceUnetModelConfig(DiffusionUnetModelConfig, TransformerV2ModelConfig):
    """Configuration for a Diffusion Sequence UNet model."""
    type: str = "diffusion_sequence_unet"

@dataclass
class DiffusionTransformerModelConfig(TransformerV2ModelConfig):
    """Configuration for a Diffusion Transformer model."""
    type: str = "diffusion_sequence_transformer"
    loss_type: str = "mse"
    dit_nblocks: int = 6
    dit_feedforward_dim: int = 2048
    dit_nheads: int = 8
    num_train_diffusion_steps: int = 100
    num_viz_actions: int = 6
    num_viz_diffusion_steps: int = 10
    parametrization: str = "eps"


"""
Store the configs in the ConfigStore
"""
cs = ConfigStore.instance()

cs.store(
    package="earthrovers.logger",
    group="logger",
    name="wandb",
    node=WandbConfig,
)
cs.store(
    package="earthrovers.trainer",
    group="trainer",
    name="base",
    node=TrainerConfig,
)
cs.store(
    package="earthrovers.dataloader",
    group="dataloader",
    name="base",
    node=DataloaderConfig,
)
cs.store(
    package="earthrovers.model",
    group="model",
    name="mlp",
    node=MLPModelConfig,
)
cs.store(
    package="earthrovers.model",
    group="model",
    name="deepseek",
    node=TransformerModelConfig,
)
cs.store(
    package="earthrovers.model",
    group="model",
    name="deepseek_big",
    node=TransformerV2ModelConfig,
)
cs.store(
    package="earthrovers.model",
    group="model",
    name="diffusion_unet",
    node=DiffusionUnetModelConfig,
)
cs.store(
    package="earthrovers.model",
    group="model",
    name="diffusion_sequence_unet",
    node=DiffusionSequenceUnetModelConfig,
)
cs.store(
    package="earthrovers.model",
    group="model",
    name="diffusion_sequence_transformer",
    node=DiffusionTransformerModelConfig,
)
cs.store(
    package="earthrovers.model",
    group="model",
    name="vint",
    node=ViNTModelConfig,
)
cs.store(
    package="earthrovers.model",
    group="model",
    name="nomad",
    node=NoMADModelConfig,
)