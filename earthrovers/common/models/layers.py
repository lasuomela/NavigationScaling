"""
Basic building blocks for the different models.
"""
from typing import List

import torch
import torch.nn as nn
import einops
from transformers import AutoModel

class ActionDecoder(nn.Module):
    """
    A transformer action head that decodes observation tokens into an action chunk.
    """
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_actions: int,
        num_action_params: int,
        num_layers: int = 3,
        num_heads: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Action queries (start as zeros, or could be learned if desired)
        self.action_tokens = nn.Parameter(
            torch.zeros(1, num_actions, model_dim),
            requires_grad=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.proj = nn.Linear(input_dim, model_dim)
        self.fc = PredictionHead(model_dim, [hidden_dim, hidden_dim, num_action_params])
        self.positional_encoding = LearnedPositionalEncoding(model_dim, num_actions)

    def _causal_mask(self, size: int, device):
        # mask shape: [size, size], True = blocked
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()

    def forward(self, obs_tokens):
        """
        Args:
            obs_tokens: [B, E] | [B, S, E]  (encoder memory, e.g. processed observations)
        Returns:
            actions: [B, A, action_dim]
        """
        B = obs_tokens.size(0)
        if obs_tokens.dim() == 2:
            obs_tokens = obs_tokens.unsqueeze(1)  # [B, 1, E]
        obs_tokens = self.proj(obs_tokens)  # [B, S, model_dim]

        # Prepare action queries
        action_tokens = einops.repeat(self.action_tokens, "1 A E -> B A E", B=B)
        action_tokens = self.positional_encoding(action_tokens)

        # Causal mask for autoregressive decoding
        causal_mask = self._causal_mask(action_tokens.size(1), obs_tokens.device)

        # Transformer decoder: action_tokens query obs_tokens
        decoded = self.decoder(
            tgt=action_tokens,
            memory=obs_tokens,
            tgt_mask=causal_mask,
        )

        # Predict actions
        actions = self.fc(decoded)  # [B, A, action_dim]
        return actions

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        """
        Learned positional encoding.
        """
        super().__init__()

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, seq_len, d_model),
        )

    def forward(self, x):
        x = x + self.position_embeddings
        return x

class GoalEncoder(nn.Module):
    """
    Encodes the goal information (e.g. distance and angle to goal) into a high-dimensional embedding.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: List[int] = [64, 128],
    ):
        super(GoalEncoder, self).__init__()
        fc = []
        for i_dim, o_dim in zip(
            [input_dim] + hidden_layer_dims,
            hidden_layer_dims + [output_dim]
        ):
            fc.append(nn.Linear(i_dim, o_dim))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.2))

        fc.append(nn.LayerNorm(output_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)

class StateEncoder(nn.Module):
    """
    Encode the image and goal embeddings into a single state representation.
    Can use either an MLP or a patch-based transformer for fusion.
    """
    def __init__(
        self,
        obs_input_dim: int,
        goal_input_dim: int,
        output_dim: int,
        fusion_type: str = "mlp",
    ):
        super(StateEncoder, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == "mlp":
            self.fusion = nn.Sequential(
                nn.Linear(obs_input_dim + goal_input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif fusion_type == "patch_transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=obs_input_dim,
                nhead=4,
                dim_feedforward=obs_input_dim,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.goal_proj = nn.Linear(goal_input_dim, obs_input_dim)
            self.goal_norm = nn.LayerNorm(obs_input_dim)
            self.goal_position = LearnedPositionalEncoding(obs_input_dim, 1)
            self.fusion = nn.TransformerEncoder(
                encoder_layer,
                num_layers=4,
            )

    def forward(self, obs, goal):
        if self.fusion_type == "mlp":
            x = torch.cat([obs, goal], dim=-1)
            return self.fusion(x)
        elif self.fusion_type == "patch_transformer":
            goal = self.goal_proj(goal)
            goal = self.goal_norm(goal)
            goal = self.goal_position(goal.unsqueeze(1))  # (B, 1, E)
            x = torch.cat([goal, obs], dim=1)  # (B, 1+P, E)
            x = self.fusion(x)
            return x[:, 0]  # Return the goal token


class DINO(nn.Module):
    """
    Wrapper for the DINO encoders.
    """
    img_preprocess_type = "IMAGENET_DEFAULT"
    def __init__(
            self,
            encoder_type: str = 'facebook/dinov2-with-registers-small',
            input_size: List[int] = [3, 224, 224],
            freeze: bool = True,
        ):
        super(DINO, self).__init__()
        model = AutoModel.from_pretrained(encoder_type)        
        self.model = model
        self.num_register_tokens = model.config.num_register_tokens

        if hasattr(self.model, 'embeddings'):
            if hasattr(self.model.embeddings, 'mask_token'):
                # Freeze the mask token if it exists
                self.model.embeddings.mask_token.requires_grad = False

        if freeze:
            self._freeze()

        self.out_featuremap_H, self.out_featuremap_W = self._get_output_patch_size(input_size)
        self.out_featuremap_C = model.config.hidden_size

    def forward(self, x):
        x = self.model(x)
        patch_tokens = x.last_hidden_state[:, 1+self.num_register_tokens:]
        return patch_tokens
    
    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def output_dim(self):
        return self.out_featuremap_C

    @property
    def output_featuremap_size(self):
        return self.out_featuremap_H, self.out_featuremap_W

    def _get_output_patch_size(self, input_size):
        """
        Get the output patch size of the model.
        """
        patch_size = self.model.config.patch_size
        return input_size[1] // patch_size, input_size[2] // patch_size

class Flatten(nn.Module):
    def __init__(
            self,
            input_dim,
            input_size,
            compression_channels=2,
            meanpool_output_dim=512,
            flatten_type="compress",
        ):
        """
        Compress a 2D feature map into a 1D tensor.

        Args:
            input_dim: Dimension of the input tensor
            input_size: Spatial size (H, W) of the input tensor
            compression_channels: Channels per spatial location in the output
            flatten_type: Type of compression layer to use (compress | meanpool)
        """

        super(Flatten, self).__init__()

        self.input_size = input_size # Spatial size of the input / output

        if flatten_type == "compress":
            self.compression = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1,compression_channels),
                nn.ReLU(True),
                nn.Flatten(),
            )
            self.output_dim = compression_channels * input_size[0] * input_size[1]

        elif flatten_type == "meanpool":
            self.compression = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(input_dim, meanpool_output_dim),
            )
            self.output_dim = meanpool_output_dim
        else:
            raise ValueError(
                f"Invalid flatten_type '{flatten_type}'. Must be one of ['compress', 'meanpool']"
            )

    def forward(self, x):
        """
        Args:
            x: [B, P, E]
        Returns:
            x: 
        """
        # Reshape the input tensor to 2D for convolution
        x = einops.rearrange(
            x,
            "b (h w) c -> b c h w",
            h=self.input_size[0],
            w=self.input_size[1],
        )

        # [B, E, H, W] -> [B, self.output_dim]
        x = self.compression(x)
        return x

class PredictionHead(torch.nn.Module):
    """
    FC layers for the prediction head.
    """
    def __init__(
            self,
            input_dim,
            output_layer_dims=[256, 128, 64, 20],
            dropout=0.2,
        ):
        super(PredictionHead, self).__init__()

        fc = []
        for i, (input_dim, output_dim) in enumerate(
            zip(
                [input_dim] + output_layer_dims[:-1],
                output_layer_dims
            )
            ):

            fc.append( nn.Dropout(dropout))
            fc.append( nn.Linear(input_dim, output_dim))
            if i < len(output_layer_dims) - 2:
                fc.append(nn.ReLU())

        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)
    
class OutputShaper(torch.nn.Module):
    """
    Converts model output into an action chunk.
    """

    def __init__(
            self,
            num_action_params,
            len_trajectory_pred,
        ):
        super(OutputShaper, self).__init__()
        self.num_action_params = num_action_params
        self.len_trajectory_pred = len_trajectory_pred

    def forward(
            self,
            action_pred,
        ):
        """
        Convert a flat model output into len_trajectory_pred actions with dim num_action_params.
        Args:
            action_pred (torch.Tensor): [B, action_pred_horizon * num_action_params]
                The action prediction from the model.

        Returns:
            torch.Tensor: [B, action_pred_horizon, num_action_params]
                The action prediction reshaped into a sequence of actions.
        """

        action_pred = einops.rearrange(
                action_pred,
                "... (t p) -> ... t p",
                t=self.len_trajectory_pred,
                p=self.num_action_params,
        )
        return action_pred
    
class WaypointShaper(torch.nn.Module):
    """
    Converts model output into a sequence of waypoints.
    """

    def __init__(
            self,
            num_waypoint_params,
            len_trajectory_pred,
        ):
        super(WaypointShaper, self).__init__()
        self.num_waypoint_params = num_waypoint_params
        self.len_trajectory_pred = len_trajectory_pred
        assert num_waypoint_params == 4, "num_waypoint_params must be 4 for [x, y, cos(theta), sin(theta)] parametrization"

    def forward(
            self,
            waypoint_pred,
        ):
        """
        Convert a flat model output into len_trajectory_pred waypoints with dim num_waypoint_params.
        Args:
            waypoint_pred (torch.Tensor): [..., waypoint_pred_horizon * num_waypoint_params]
                The waypoint prediction from the model.

        Returns:
            torch.Tensor: [..., waypoint_pred_horizon, num_waypoint_params]
                The waypoint prediction reshaped into a sequence of waypoints.
        """

        waypoint_pred = einops.rearrange(
                waypoint_pred,
                "... (t p) -> ... t p",
                t=self.len_trajectory_pred,
                p=self.num_waypoint_params,
        )
        # Assume the parametrization is [x, y, cos(theta), sin(theta)]
        # Convert waypoint position deltas to absolute waypoints
        waypoint_positions = torch.cumsum(waypoint_pred[..., :2], dim=-2)

        # Normalize the rotation parameters to a valid rotation
        waypoint_orientations = torch.nn.functional.normalize(waypoint_pred[..., 2:], dim=-1)

        # Combine positions and orientations
        waypoint_pred = torch.cat([waypoint_positions, waypoint_orientations], dim=-1)

        return waypoint_pred
    

class FourierEmbedding(nn.Module):
    """
    Periodic embedding for angles (radians).
    Ensures wrap-around continuity: -pi+eps ≈ pi-eps.
    """
    def __init__(self, num_freqs=8, base=2.0):
        super().__init__()
        freqs = (base ** torch.arange(num_freqs)).float()
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, theta):
        """
        Args:
            theta: (B, 1) angle in radians in [-pi, pi]
        Returns:
            (B, 2 * num_freqs) embedding
        """
        theta = theta * self.freqs  # (B, F)
        return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)  # (B, 2F)

class GoalEmbedder(nn.Module):
    """
    Embeds a goal represented as (distance, angle) into a high-dimensional space using Fourier features for the angle.
    Not used in the current model.
    """
    def __init__(
        self,
        max_goal_distance: float = 0.150,  # in kilometers
        num_freqs: int = 8,
        base: float = 2.0
    ):
        super(GoalEmbedder, self).__init__()
        self.max_goal_distance = max_goal_distance
        self.embedder = FourierEmbedding(num_freqs=num_freqs, base=base)
        self.num_freqs = num_freqs

    @property
    def output_dim(self):
        return 4 * self.num_freqs  # 2x for distance + 2x for angle

    def forward(self, goal):
        """
        Args:
            goal: (B, 2) where goal[:, 0] is distance in kilometers and goal[:, 1] is in [-1, 1] representing angle in radians
        """
        goal_distance = torch.clamp(goal[:, 0:1] / self.max_goal_distance, 0, 1)  # Normalize distance to [0, 1]
        goal_distance_embedded = self.embedder(goal_distance * torch.pi)  # (B, 16), scale to [0, pi] for embedding
        goal_angle = goal[:, 1:2] * torch.pi  # Scale angle to [-pi, pi]
        goal_angle_embedded = self.embedder(goal_angle)  # (B, 16)
        goal_embedded = torch.cat([goal_distance_embedded, goal_angle_embedded], dim=-1)  # (B, 32)
        return goal_embedded