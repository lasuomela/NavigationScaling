"""
A ViNT-like architecture, adapted for PointGoals and using pretrained transformer vision encoders.
"""
import torch
import torch.nn as nn
import einops
import math

from earthrovers.common.models.layers import DINO, GoalEncoder, Flatten, PredictionHead, OutputShaper
from earthrovers.common.models.theia_encoder import TheiaEncoder

from huggingface_hub import PyTorchModelHubMixin

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        """
        The basic positional encoding from ViNT.
        https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py

        Args:
            d_model: Dimension of the model
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

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

class SeqEncoder(nn.Module):
    def __init__(
            self,
            seq_len,
            input_dim,
            num_layers=4,
            nhead=4,
            ff_dim_factor=4,
            output_type="cls",
            use_cls_token=True,
            pos_enc_type="learned",
        ):
        super(SeqEncoder, self).__init__()

        if output_type not in ["cls", "sequence"]:
            raise ValueError(
                f"Invalid output_type '{output_type}'"
            )
        if output_type == "cls" and not use_cls_token:
            raise ValueError(
                "output_type 'cls' requires use_cls_token=True"
            )
        if pos_enc_type not in ["learned", "sinusoidal"]:
            raise ValueError(
                f"Invalid pos_enc_type '{pos_enc_type}'"
            )

        self.output_type = output_type
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

        # Length of the input is observation sequence length + goal (+ cls_token)
        pos_enc_seq_len = seq_len + 1
        if use_cls_token:
            pos_enc_seq_len += 1
        
        if pos_enc_type == "learned":
            self.position_embeddings = LearnedPositionalEncoding(
                d_model=input_dim,
                seq_len=pos_enc_seq_len,
            )
        elif pos_enc_type == "sinusoidal":
            self.position_embeddings = SinusoidalPositionalEncoding(
                d_model=input_dim,
                seq_len=pos_enc_seq_len,
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim*ff_dim_factor,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        output_dim = input_dim
        if output_type == "sequence":
            output_dim *= seq_len + 1 # Seq_len + goal tokens
        self.output_dim = output_dim

    @property
    def encoding_dim(self):
        return self.output_dim

    def forward(
            self,
            obs_encoding,
            goal_encoding,
        ):
        """
        Args:
            obs_encoding: [B, S, E]
            goal_encoding: [B, E]
        Returns:
            x: [B, E]
        """
        
        x = [obs_encoding, goal_encoding.unsqueeze(1)]

        if self.use_cls_token:
            cls_token = einops.repeat(
                self.cls_token,
                "1 1 c -> b 1 c",
                b=obs_encoding.shape[0],
            )
            x.append(cls_token)

        # Concatenate the observation, goal, and cls tokens along the sequence dimension
        x = torch.cat(x, dim=1)

        x = self.position_embeddings(x)

        # [B, S+(1|2), E] -> [B, S+(1|2), E]
        x = self.encoder(x)

        if self.output_type == "cls":
            # Pick the cls token
            # [B, S+1, E] -> [B, E]
            x = x[:, -1]
        elif self.output_type == "sequence":
            # ViNT style flattens the entire sequence
            # Drop the cls token
            if self.use_cls_token:
                x = x[:, :-1]
            x = einops.rearrange(
                x,
                "b s e -> b (s e)",
            )
        return x

class ViNTLikeModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            encoder_type='facebook/dinov2-with-registers-small',
            input_size=(3, 224, 224),
            pretrained_encoder=True,
            freeze_encoder=True,
            sequence_length=6,
            num_flatten_channels=6,
            num_actions_parameters=2,
            num_prediction_steps=10,
            goal_input_dim=2,
        ):
        super(ViNTLikeModel, self).__init__()

        if 'dino' in encoder_type:
            self.img_encoder = DINO(
                encoder_type=encoder_type,
                input_size=input_size,
                freeze=freeze_encoder,
            )
        elif 'theia' in encoder_type:
            self.img_encoder = TheiaEncoder(
                encoder_type=encoder_type,
                input_size=input_size,
                freeze=freeze_encoder,
                pretrained=pretrained_encoder,
            )

        self.compressor = Flatten(
            input_dim = self.img_encoder.output_dim,
            input_size = self.img_encoder.output_featuremap_size,
            compression_channels=num_flatten_channels,
        )

        self.goal_encoder = GoalEncoder(
            input_dim=goal_input_dim,
            output_dim=self.compressor.output_dim,
            hidden_layer_dims=[64, 64],
        )

        self.seq_encoder = SeqEncoder(
            seq_len=sequence_length,
            input_dim=self.compressor.output_dim,
            num_layers=4,
            nhead=4,
            ff_dim_factor=4,
            output_type='cls',
            use_cls_token=True,
            pos_enc_type='sinusoidal',
        )

        self.prediction_head = PredictionHead(
            self.compressor.output_dim,
            output_layer_dims=[1024, 512, 256, 128, 64, num_actions_parameters*num_prediction_steps]
        )

        self.output_shaper = OutputShaper(
            num_action_params=num_actions_parameters,
            len_trajectory_pred=num_prediction_steps,
        )

    def forward(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, C, H, W]
            goal: [B, S, G]

        Returns:
            x: [B, 1, A, P]
        """
        # Rearrange the input for the encoder
        b_sz = x.shape[0]
        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        
        # (b s) c h w -> (b s) p c
        x = self.img_encoder(x)

        # (b s) p c -> (b s) c
        x = self.compressor(x)

        goal = goal[:, -1]  # Use only the last goal in the sequence

        # b, goal_input_dim -> b, c
        goal = self.goal_encoder(goal)

        x = einops.rearrange(x, '(b s) c -> b s c', b=b_sz)
        x = self.seq_encoder(x, goal)
        x = self.prediction_head(x)
        x = self.output_shaper(x)
        x = einops.rearrange(x, 'b a p -> b 1 a p')
        return {'out': x}

if __name__ == '__main__':

    model = ViNTLikeModel()
    x = torch.randn(1, 5, 3, 224, 224)
    g = torch.randn(1, 5, 2)
    print(model(x,g)['out'].shape)

