"""
A transformer-based model that takes a sequence of images and goal vectors as input,
encodes them using a vision transformer,
processes the sequence with a transformer decoder,
and outputs a sequence of action chunks.
"""
import torch
import torch.nn as nn
import einops
import transformers

from huggingface_hub import PyTorchModelHubMixin

from earthrovers.common.models.layers import DINO, GoalEncoder, StateEncoder, Flatten, PredictionHead, OutputShaper, WaypointShaper
from earthrovers.common.models.theia_encoder import TheiaEncoder
from earthrovers.common.models.deepseekv3.navigation_deepseek import DeepseekV3ForRobotNavigation
from earthrovers.common.models.deepseekv3.configuration_deepseek import DeepseekV3Config

class TransformerModelV2(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            encoder_type='theaiinstitute/theia-base-patch16-224-cddsv',
            input_size=(3, 224, 224),
            pretrained_encoder=True,
            freeze_encoder=True,
            late_goal_fusion=False,
            num_actions_parameters=2,
            num_prediction_steps=10,
            goal_input_dim=2,
            goal_output_dim=64,
            sequence_length=1,
            predict_waypoints=False,
            num_waypoints=10,
            num_flatten_channels=6,
            ds_num_hidden_layers=8,
            ds_num_attention_heads=16,
            ds_intermediate_size=2048,
            ds_qk_rope_head_dim=32,
            ds_qk_nope_head_dim=64,
            ds_v_head_dim=64,
            ds_kv_lora_rank=256,
            ds_q_lora_rank=512,
            **kwargs         
        ):
        super(TransformerModelV2, self).__init__()
        self.late_goal_fusion = late_goal_fusion

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
            output_dim=goal_output_dim,
            hidden_layer_dims=[64, 64],
        )

        self.state_encoder = StateEncoder(
            obs_input_dim=self.compressor.output_dim,
            goal_input_dim=goal_output_dim,
            output_dim=self.compressor.output_dim,
            fusion_type='mlp',
        )

        ds_config = DeepseekV3Config(
            num_hidden_layers=ds_num_hidden_layers,
            num_attention_heads=ds_num_attention_heads,
            num_key_value_heads=ds_num_attention_heads, # num_attention_heads == num_key_value_heads means MHA
            hidden_size=self.compressor.output_dim,
            n_routed_experts=None, # if None, utilize MLP in forward pass
            intermediate_size=ds_intermediate_size, # MLP hidden size if n_routed_experts == None
            qk_rope_head_dim=ds_qk_rope_head_dim,
            qk_nope_head_dim=ds_qk_nope_head_dim,
            v_head_dim=ds_v_head_dim,
            kv_lora_rank=ds_kv_lora_rank,
            q_lora_rank=ds_q_lora_rank,
            attention_dropout=0.1,
            use_cache=True,
            max_position_embeddings=sequence_length,
        )
        self.sequence_processor = DeepseekV3ForRobotNavigation(ds_config)
        self._attn_mask = torch.ones(1, ds_config.max_position_embeddings)

        self.prediction_head = PredictionHead(
            self.compressor.output_dim,
        )

        self.output_shaper = OutputShaper(
            num_action_params=num_actions_parameters,
            len_trajectory_pred=num_prediction_steps,
        )

        if predict_waypoints:
            self.waypoint_prediction_head = PredictionHead(
                input_dim=self.compressor.output_dim,
                output_layer_dims=[256, 128, 64, num_waypoints * 4],
            )
            self.waypoint_shaper = WaypointShaper(
                num_waypoint_params=4,
                len_trajectory_pred=num_waypoints,
            )

    def forward(
            self,
            x: torch.Tensor,
            goal: torch.Tensor,
            past_key_values: transformers.Cache = None,
        ) -> torch.Tensor:
        """
        Args:
            x: [B, S, C, H, W] | [B, C, H, W] - input image tensor
            goal: [B, S, G] | [B, G] - goal tensor
        Returns:
            x: [B, S, A] | [B, A] - action tensor
        """
        # Check if input is just a single image
        if (len(x.shape) == 4) & (len(goal.shape) == 2):
            x = einops.repeat(x, 'b c h w -> b s c h w', s=1)
            goal = einops.repeat(goal, 'b g -> b s g', s=1)
            s_sz = 0
        else:
            s_sz = x.shape[1]

        assert len(x.shape) == 5, f"Input tensor x should be of shape [B, S, C, H, W], but got {x.shape}"
        assert len(goal.shape) == 3, f"Goal tensor should be of shape [B, S, G], but got {goal.shape}"

        # Rearrange the input for the encoder
        b_sz = x.shape[0]
        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        goal = einops.rearrange(goal, 'b s g -> (b s) g')

        # (b s) c h w -> (b s) p c
        x = self.img_encoder(x)

        # (b s) p c -> (b s) c
        x = self.compressor(x)

        # (b s) g -> (b s) c
        goal = self.goal_encoder(goal)
        
        # (b s) c -> (b s) c
        if not self.late_goal_fusion:
            x = self.state_encoder(x, goal)
        x = einops.rearrange(x, '(b s) c -> b s c', b=b_sz)

        seq_length = x.shape[1]
        if past_key_values is not None:
            # If past_key_values is provided, we need to adjust the sequence length
            seq_length += past_key_values.get_usable_length(seq_length)

        attn_mask = einops.repeat(self._attn_mask[:, :seq_length], '1 s -> b s', b=b_sz).to(x.device)
        sequence_out = self.sequence_processor(
            inputs_embeds=x,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        )

        x = sequence_out['last_hidden_state']

        if self.late_goal_fusion:
            goal = einops.rearrange(goal, '(b s) c -> b s c', b=b_sz)
            # b s c -> b s c
            x = self.state_encoder(x, goal)

        # b s c -> b s c
        actions = self.prediction_head(x)
        # b s c -> b s n a
        actions = self.output_shaper(actions)

        if s_sz == 0:
            actions = einops.rearrange(actions, 'b 1 n a -> b n a')

        out = {'out': actions}

        if hasattr(self, 'waypoint_prediction_head'):
            # b s c -> b s c
            waypoints = self.waypoint_prediction_head(x)
            # b s c -> b s n a
            waypoints = self.waypoint_shaper(waypoints)
            if s_sz == 0:
                waypoints = einops.rearrange(waypoints, 'b 1 n a -> b n a')

            out['waypoints'] = waypoints

        if 'past_key_values' in sequence_out:
            out['past_key_values'] = sequence_out['past_key_values']

        return out

if __name__ == '__main__':
    model = TransformerModelV2(
        encoder_type="theaiinstitute/theia-base-patch16-224-cddsv",
        sequence_length=100,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    x = torch.randn(1, 100, 3, 224, 224, device=device)
    g = torch.randn(1, 100, 2, device=device)
    out = model(x, g)
    out = out['out']
    print(out.shape)

    import time

    for i in range(100):
        start = time.time()
        out = model(x[:, [i]], g[:, [i]])
        print('Time taken:', time.time() - start)
        out = out['out']

