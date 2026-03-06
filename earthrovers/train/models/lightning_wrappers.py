"""
Wrap pure pytorch models in pytorch-lightning modules for training.
"""
from typing import Dict
from omegaconf import DictConfig

import torch
import lightning as L
from copy import deepcopy

from earthrovers.train.config.registry import registry
from earthrovers.common.models.mlp_model import MLPModel
from earthrovers.common.models.transformer_model_v2 import TransformerModelV2
from earthrovers.common.models.diffusion_unet_model import DiffusionUnetModel
from earthrovers.common.models.diffusion_transformer_model import DiffusionTransformerModel
from earthrovers.common.models.vint_like import ViNTLikeModel
from earthrovers.common.models.nomad_like import NoMADLikeModel
from earthrovers.train.models.losses import ScaledMSELoss

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher


class BaseWrapper(L.LightningModule):
    """
    Base class for all Lightning wrappers.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = None

        if config.loss_type == "scaled_mse":
            self.train_loss = ScaledMSELoss()
            self.val_loss = ScaledMSELoss()
        elif config.loss_type == "mse":
            self.train_loss = torch.nn.MSELoss()
            self.val_loss = torch.nn.MSELoss()
        elif config.loss_type == "l1":
            self.train_loss = torch.nn.L1Loss()
            self.val_loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
        
        if hasattr(config, 'predict_waypoints') and config.predict_waypoints:
            self.waypoint_loss = torch.nn.MSELoss()
            self.waypoint_loss_alpha = config.waypoint_loss_alpha

        self.example_input_array = (
            torch.zeros((1, config.sequence_length, 3, *config.input_size)),
            torch.zeros((1, config.sequence_length, config.goal_input_dim)),
        )

    @property
    def img_preprocess_type(self) -> str:
        return self.model.img_encoder.img_preprocess_type

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer =  torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-7, T_max=self.config.num_epochs)
            return [optimizer], [scheduler]
        elif self.config.lr_scheduler == '':
            return optimizer
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.config.lr_scheduler}")

@registry.register_model(name="mlp")
class LightningMLPModel(BaseWrapper):
    """
    Lightning wrapper for a simple MLP model.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__(config=config)
        self.model = MLPModel(
            encoder_type=config.encoder_type,
            input_size=(3,)+config.input_size,
            pretrained_encoder=config.pretrained_encoder,
            freeze_encoder=config.freeze_encoder,
            num_flatten_channels=config.num_flatten_channels,
            num_actions_parameters=config.num_action_parameters,
            num_prediction_steps=config.num_prediction_steps,
            goal_input_dim=config.goal_input_dim,
            goal_output_dim=config.goal_output_dim,
            goal_embed=config.embed_goal if hasattr(config, 'embed_goal') else False,
            state_fusion_type=config.state_fusion_type if hasattr(config, 'state_fusion_type') else 'mlp',
            action_decoder_type=config.action_decoder_type if hasattr(config, 'action_decoder_type') else 'mlp',
        )

    def forward(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.model(x, goal)['out']

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(batch["front_camera"], batch['goal_input'])
        actions = out['out']
        loss = self.train_loss(actions, batch["target"])
        self.log(
            f"train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["goal_input"].shape[0],
        )
        return {'loss': loss, 'actions': actions}
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(batch["front_camera"], batch['goal_input'])
        actions = out['out']
        loss = self.val_loss(actions, batch["target"])
        self.log(
            f"val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["goal_input"].shape[0],
        )

        return {'loss': loss, 'actions': actions}

class LightningDiffusionModel(BaseWrapper):
    """
    The base for all diffusion models.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__(config=config)

        if config.parametrization == 'cfm':
            # Conditional Flow Matching
            self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
            self.diffusion_schedule = None

        elif config.parametrization == 'eps':
            # Diffusion with epsilon prediction
            self.diffusion_schedule = DDIMScheduler(
                num_train_timesteps=config.num_train_diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
            self.flow_matcher = None
        else:
            raise ValueError(f"Unknown parametrization: {config.parametrization}")

    def forward(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.model.predict(
            x,
            goal,
            parametrization=self.config.parametrization,
            diffusion_schedule=deepcopy(self.diffusion_schedule),
            num_actions=self.config.num_viz_actions,
            num_steps=self.config.num_viz_diffusion_steps,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(
            batch["front_camera"],
            batch['goal_input'],
            batch["target"],
            diffusion_schedule=self.diffusion_schedule,
            flow_matcher=self.flow_matcher,
        )
        if self.config.loss_type == "scaled_mse":
            loss = self.train_loss(out['prediction'], out["target"], target_cmds=batch['target'])
        else:
            loss = self.train_loss(out['prediction'], out["target"])

        self.log(
            f"train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["goal_input"].shape[0],
        )
        out = {'loss': loss}

        # Create some actions for visualization
        if 'visualize_prediction' in batch and batch['visualize_prediction']:
            with torch.inference_mode():
                prediction = self.model.predict(
                    batch["front_camera"],
                    batch['goal_input'],
                    parametrization=self.config.parametrization,
                    diffusion_schedule=deepcopy(self.diffusion_schedule),
                    num_actions=self.config.num_viz_actions,
                    num_steps=self.config.num_viz_diffusion_steps,
                )
                out['actions'] = prediction['out']

        return out
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(
            batch["front_camera"],
            batch['goal_input'],
            batch["target"],
            diffusion_schedule=self.diffusion_schedule,
            flow_matcher=self.flow_matcher,
        )
        if self.config.loss_type == "scaled_mse":
            loss = self.val_loss(out['prediction'], out["target"], target_cmds=batch['target'])
        else:
            loss = self.val_loss(out['prediction'], out["target"])

        self.log(
            f"val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["goal_input"].shape[0],
        )
        out = {'loss': loss}

        # Create some actions for visualization
        if 'visualize_prediction' in batch and batch['visualize_prediction']:
            with torch.inference_mode():
                prediction = self.model.predict(
                    batch["front_camera"],
                    batch['goal_input'],
                    parametrization=self.config.parametrization,
                    diffusion_schedule=deepcopy(self.diffusion_schedule),
                    num_actions=self.config.num_viz_actions,
                    num_steps=self.config.num_viz_diffusion_steps,
                )
                out['actions'] = prediction['out']

        return out
    

@registry.register_model(name="diffusion_unet")
class LightningDiffusionUnetModel(LightningDiffusionModel):
    """
    Lightning wrapper for a simple diffusion U-Net model.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__(config=config)

        self.model = DiffusionUnetModel(
            encoder_type=config.encoder_type,
            input_size=(3,)+config.input_size,
            pretrained_encoder=config.pretrained_encoder,
            freeze_encoder=config.freeze_encoder,
            num_actions_parameters=config.num_action_parameters,
            num_prediction_steps=config.num_prediction_steps,
            goal_input_dim=config.goal_input_dim,
            goal_output_dim=config.goal_output_dim,
            use_goal=config.use_goal if hasattr(config, 'use_goal') else True,
            num_flatten_channels=config.num_flatten_channels,
            diffusion_unet_step_embed_dim=config.diffusion_unet_step_embed_dim,
            diffusion_unet_down_dims=config.diffusion_unet_down_dims,
            diffusion_unet_kernel_size=config.diffusion_unet_kernel_size,
            diffusion_unet_num_groups=config.diffusion_unet_num_groups,
        )

@registry.register_model(name="vint")
class LightningViNTModel(BaseWrapper):
    """
    Lightning wrapper for a ViNT-like model.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__(config=config)

        self.model = ViNTLikeModel(
            encoder_type=config.encoder_type,
            input_size=(3,)+config.input_size,
            pretrained_encoder=config.pretrained_encoder,
            freeze_encoder=config.freeze_encoder,
            sequence_length=config.sequence_length,
            num_actions_parameters=config.num_action_parameters,
            num_prediction_steps=config.num_prediction_steps,
            goal_input_dim=config.goal_input_dim,
            num_flatten_channels=config.num_flatten_channels,
        )

    def forward(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.model(x, goal)['out']

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(batch["front_camera"], batch['goal_input'])
        actions = out['out']
        loss = self.train_loss(actions, batch["target"][:, [-1]])
        self.log(
            f"train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["goal_input"].shape[0],
        )
        return {'loss': loss, 'actions': actions}
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(batch["front_camera"], batch['goal_input'])
        actions = out['out']
        loss = self.val_loss(actions, batch["target"][:, [-1]])
        self.log(
            f"val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["goal_input"].shape[0],
        )

        return {'loss': loss, 'actions': actions}

@registry.register_model(name="nomad")
class LightningNoMADModel(LightningDiffusionModel):
    """
    Lightning wrapper for a NoMAD-like model.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__(config=config)

        self.model = NoMADLikeModel(
            encoder_type=config.encoder_type,
            input_size=(3,)+config.input_size,
            pretrained_encoder=config.pretrained_encoder,
            freeze_encoder=config.freeze_encoder,
            sequence_length=config.sequence_length,
            num_flatten_channels=config.num_flatten_channels,
            num_actions_parameters=config.num_action_parameters,
            num_prediction_steps=config.num_prediction_steps,
            goal_input_dim=config.goal_input_dim,
        )

#
# The models below have not been modified for recent changes and might not work out of the box.
#

@registry.register_model(name="deepseek")
class LightningTransformerModel(BaseWrapper):
    """
    The base class for a policy that processes long observation sequences with a
    Deepseek-like transformer architecture.
    """

    def forward(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.model(x, goal)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(batch["front_camera"], batch['goal_input'])
        actions = out['out']
        action_loss = self.train_loss(actions, batch["target"])
        loss = action_loss

        if 'waypoints' in out:
            waypoint_loss = self.waypoint_loss(out['waypoints'], batch["waypoints"])
            loss = action_loss + self.waypoint_loss_alpha * waypoint_loss
            self.log(
                f"train_waypoint_loss",
                waypoint_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batch["goal_input"].shape[0],
            )

        self.log(
            f"train_loss",
            action_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["goal_input"].shape[0],
        )
        ret = {'loss': loss, 'actions': actions}
        if 'waypoints' in out:
            ret['waypoints'] = out['waypoints']
        return ret
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(batch["front_camera"], batch['goal_input'])
        actions = out['out']
        action_loss = self.val_loss(actions, batch["target"])
        loss = action_loss
        if 'waypoints' in out:
            waypoint_loss = self.waypoint_loss(out['waypoints'], batch["waypoints"])
            loss = action_loss + self.waypoint_loss_alpha * waypoint_loss
            self.log(
                f"val_waypoint_loss",
                waypoint_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batch["goal_input"].shape[0],
            )

        self.log(
            f"val_loss",
            action_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch["goal_input"].shape[0],
        )

        ret = {'loss': loss, 'actions': actions}
        if 'waypoints' in out:
            ret['waypoints'] = out['waypoints']
        return ret
        
@registry.register_model(name="deepseek_big")
class LightningTransformerModelV2(LightningTransformerModel):
    """
    The class that is actually used for Deepseek-like training.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__(config=config)
        self.model = TransformerModelV2(
            encoder_type=config.encoder_type,
            input_size=(3,)+config.input_size,
            pretrained_encoder=config.pretrained_encoder,
            freeze_encoder=config.freeze_encoder,
            late_goal_fusion=config.late_goal_fusion,
            num_actions_parameters=config.num_action_parameters,
            num_prediction_steps=config.num_prediction_steps,
            goal_input_dim=config.goal_input_dim,
            goal_output_dim=config.goal_output_dim,
            sequence_length=config.sequence_length,
            predict_waypoints=config.predict_waypoints,
            num_waypoints=config.num_waypoints,
            num_flatten_channels=config.num_flatten_channels,
            ds_num_hidden_layers=config.num_hidden_layers,
            ds_num_attention_heads=config.num_attention_heads,
            ds_intermediate_size=config.intermediate_size,
            ds_qk_rope_head_dim=config.qk_rope_head_dim,
            ds_qk_nope_head_dim=config.qk_nope_head_dim,
            ds_v_head_dim=config.v_head_dim,
            ds_kv_lora_rank=config.kv_lora_rank,
            ds_q_lora_rank=config.q_lora_rank,
        )

@registry.register_model(name="diffusion_sequence_transformer")
class LightningDiffusionTransformerModel(LightningDiffusionModel):
    """
    Lightning wrapper for a transformer-based policy that takes observation sequences as input.
    Action head is a diffusion model with a DiT-like transformer as noise decoder.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__(config=config)

        self.model = DiffusionTransformerModel(
            encoder_type=config.encoder_type,
            input_size=(3,)+config.input_size,
            pretrained_encoder=config.pretrained_encoder,
            freeze_encoder=config.freeze_encoder,
            late_goal_fusion=config.late_goal_fusion,
            num_actions_parameters=config.num_action_parameters,
            num_prediction_steps=config.num_prediction_steps,
            goal_input_dim=config.goal_input_dim,
            goal_output_dim=config.goal_output_dim,
            sequence_length=config.sequence_length,
            num_flatten_channels=config.num_flatten_channels,
            ds_num_hidden_layers=config.num_hidden_layers,
            ds_num_attention_heads=config.num_attention_heads,
            ds_intermediate_size=config.intermediate_size,
            ds_qk_rope_head_dim=config.qk_rope_head_dim,
            ds_qk_nope_head_dim=config.qk_nope_head_dim,
            ds_v_head_dim=config.v_head_dim,
            ds_kv_lora_rank=config.kv_lora_rank,
            ds_q_lora_rank=config.q_lora_rank,
            noise_decoder_type='dit',
            dit_nblocks=config.dit_nblocks,
            dit_nheads=config.dit_nheads,
            dit_feedforward_dim=config.dit_feedforward_dim,
        )

@registry.register_model(name="diffusion_sequence_unet")
class LightningDiffusionSequenceUnetModel(LightningDiffusionModel):
    """
    Lightning wrapper for a transformer-based policy that takes observation sequences as input.
    Action head is a diffusion model with a U-Net noise decoder.
    """
    def __init__(
            self,
            config: DictConfig = None,
        ):
        super().__init__(config=config)

        self.model = DiffusionTransformerModel(
            encoder_type=config.encoder_type,
            input_size=(3,)+config.input_size,
            pretrained_encoder=config.pretrained_encoder,
            freeze_encoder=config.freeze_encoder,
            late_goal_fusion=config.late_goal_fusion,
            num_actions_parameters=config.num_action_parameters,
            num_prediction_steps=config.num_prediction_steps,
            goal_input_dim=config.goal_input_dim,
            goal_output_dim=config.goal_output_dim,
            sequence_length=config.sequence_length,
            num_flatten_channels=config.num_flatten_channels,
            ds_num_hidden_layers=config.num_hidden_layers,
            ds_num_attention_heads=config.num_attention_heads,
            ds_intermediate_size=config.intermediate_size,
            ds_qk_rope_head_dim=config.qk_rope_head_dim,
            ds_qk_nope_head_dim=config.qk_nope_head_dim,
            ds_v_head_dim=config.v_head_dim,
            ds_kv_lora_rank=config.kv_lora_rank,
            ds_q_lora_rank=config.q_lora_rank,
            noise_decoder_type='unet',
            diffusion_unet_step_embed_dim=config.diffusion_unet_step_embed_dim,
            diffusion_unet_down_dims=config.diffusion_unet_down_dims,
            diffusion_unet_kernel_size=config.diffusion_unet_kernel_size,
            diffusion_unet_num_groups=config.diffusion_unet_num_groups,
        )