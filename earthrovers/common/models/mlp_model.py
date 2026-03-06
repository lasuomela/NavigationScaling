"""
A simple MLP model.
"""
import torch
import torch.nn as nn
import einops

from earthrovers.common.models.layers import DINO, GoalEncoder, StateEncoder, Flatten, PredictionHead, OutputShaper, ActionDecoder, GoalEmbedder
from earthrovers.common.models.theia_encoder import TheiaEncoder

from huggingface_hub import PyTorchModelHubMixin

class MLPModel(nn.Module, PyTorchModelHubMixin):
    """
    A simple MLP model that encodes a single image and a goal vector,
    fuses them together, and decodes a sequence of future actions.
    """
    def __init__(
            self,
            encoder_type='facebook/dinov2-with-registers-small',
            input_size=(3, 224, 224),
            pretrained_encoder=True,
            freeze_encoder=True,
            num_flatten_channels=6,
            num_actions_parameters=2,
            num_prediction_steps=10,
            goal_input_dim=2,
            goal_output_dim=64,
            goal_embed=False,
            state_fusion_type='mlp',
            action_decoder_type='mlp',
        ):
        super(MLPModel, self).__init__()

        self.goal_embed = goal_embed
        self.state_fusion_type = state_fusion_type
        self.action_decoder_type = action_decoder_type

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

        if state_fusion_type != 'patch_transformer':
            self.compressor = Flatten(
                input_dim = self.img_encoder.output_dim,
                input_size = self.img_encoder.output_featuremap_size,
                compression_channels=num_flatten_channels,
            )

        if goal_embed:
            self.goal_embedder = GoalEmbedder()
            goal_input_dim = self.goal_embedder.output_dim

        self.goal_encoder = GoalEncoder(
            input_dim=goal_input_dim,
            output_dim=goal_output_dim,
            hidden_layer_dims=[64, 64],
        )

        if state_fusion_type == 'mlp':
            obs_input_dim = self.compressor.output_dim
        elif state_fusion_type == 'patch_transformer':
            obs_input_dim = self.img_encoder.output_dim

        self.state_encoder = StateEncoder(
            obs_input_dim=obs_input_dim,
            goal_input_dim=goal_output_dim,
            output_dim=obs_input_dim,
            fusion_type=state_fusion_type,
        )

        if action_decoder_type == 'mlp':
            self.prediction_head = PredictionHead(
                obs_input_dim,
                output_layer_dims=[1024, 512, 256, 128, 64, num_actions_parameters*num_prediction_steps]
            )

            self.output_shaper = OutputShaper(
                num_action_params=num_actions_parameters,
                len_trajectory_pred=num_prediction_steps,
            )
        elif action_decoder_type == 'transformer':
            self.prediction_head = ActionDecoder(
                input_dim=obs_input_dim,
                model_dim=256,
                num_actions=num_prediction_steps,
                num_action_params=num_actions_parameters,
                num_layers=4,
                num_heads=4,
                hidden_dim=512,
            )


    def forward(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, 'b 1 c h w -> b c h w')
        goal = einops.rearrange(goal, 'b 1 c -> b c')

        if self.goal_embed:
            goal = self.goal_embedder(goal)
        
        x = self.img_encoder(x)

        if self.state_fusion_type != 'patch_transformer':
            x = self.compressor(x)

        goal = self.goal_encoder(goal)
        x = self.state_encoder(x, goal)
        x = self.prediction_head(x)

        if self.action_decoder_type == 'mlp':
            x = self.output_shaper(x)

        x = einops.rearrange(x, 'b a p -> b 1 a p')
        if torch.jit.is_tracing():
            return x
        return {'out': x}

if __name__ == '__main__':
    # login to huggingface
    import huggingface_hub

    model = MLPModel(
        goal_embed=True,
        state_fusion_type='patch_transformer',
        action_decoder_type='transformer',
        encoder_type='facebook/dinov3-vitb16-pretrain-lvd1689m',
    )
    x = torch.randn(1, 1, 3, 224, 224)
    g = torch.randn(1, 1, 2)
    print(model)
    print(model(x,g)['out'].shape)


