"""
A model with transformer image encoder, causal self-attention sequence processor, and a DiT action decoder.
"""
import torch
import torch.nn as nn
import einops
import transformers
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from huggingface_hub import PyTorchModelHubMixin

from earthrovers.common.models.layers import DINO, GoalEncoder, StateEncoder, Flatten
from earthrovers.common.models.theia_encoder import TheiaEncoder
from earthrovers.common.models.deepseekv3.navigation_deepseek import DeepseekV3ForRobotNavigation
from earthrovers.common.models.deepseekv3.configuration_deepseek import DeepseekV3Config
from earthrovers.common.models.diffusion.dit_block import _DitNoiseDecoder
from earthrovers.common.models.diffusion.conditional_unet1d import ConditionalUnet1D

class DiffusionTransformerModel(nn.Module, PyTorchModelHubMixin):
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
            num_flatten_channels=6,
            ds_num_hidden_layers=8,
            ds_num_attention_heads=16,
            ds_intermediate_size=2048,
            ds_qk_rope_head_dim=32,
            ds_qk_nope_head_dim=64,
            ds_v_head_dim=64,
            ds_kv_lora_rank=256,
            ds_q_lora_rank=512,
            noise_decoder_type='unet',
            diffusion_unet_step_embed_dim=128,
            diffusion_unet_down_dims=[256,512,1024],
            diffusion_unet_kernel_size=5,
            diffusion_unet_num_groups=8,
            dit_nblocks=6,
            dit_nheads=8,
            dit_feedforward_dim=2048,
            **kwargs         
        ):
        super(DiffusionTransformerModel, self).__init__()

        self.num_actions_parameters = num_actions_parameters
        self.num_prediction_steps = num_prediction_steps
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

        if noise_decoder_type == 'unet':
            self.noise_decoder = ConditionalUnet1D(
                input_dim=num_actions_parameters,
                global_cond_dim=self.compressor.output_dim,
                diffusion_step_embed_dim=diffusion_unet_step_embed_dim,
                down_dims=diffusion_unet_down_dims,
                kernel_size=diffusion_unet_kernel_size,
                n_groups=diffusion_unet_num_groups,
                cond_predict_scale=True,
            )
        elif noise_decoder_type == 'dit':
            self.noise_decoder = _DitNoiseDecoder(
                ac_dim=num_actions_parameters,
                ac_chunk=num_prediction_steps,
                hidden_dim=self.compressor.output_dim,
                num_blocks=dit_nblocks,
                dim_feedforward=dit_feedforward_dim,
                nhead=dit_nheads,
            )
        else:
            raise ValueError(f"Unknown noise decoder type: {noise_decoder_type}")

    def forward_encoder(
            self,
            x: torch.Tensor,
            goal: torch.Tensor,
            past_key_values: transformers.Cache = None,
        ) -> torch.Tensor:
        """
        Forward pass for the encoder part of the model.

        Args:
            x: [B, S, C, H, W] - input image tensor
            goal: [B, S, G] - goal tensor
            past_key_values: transformers.Cache - optional KV cache for inference
        Returns:
            x: [B, S, C] - encoded state tensor
        """

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

        return x, sequence_out.get('past_key_values', None)

    def forward(
            self,
            x: torch.Tensor,
            goal: torch.Tensor,
            gt_actions: torch.Tensor,
            diffusion_schedule: SchedulerMixin,
        ) -> torch.Tensor:
        """
        Train forward pass of the model.

        Args:
            x: [B, S, C, H, W] | [B, C, H, W] - input image tensor
            goal: [B, S, G] | [B, G] - goal tensor
            gt_actions: [B, S, self.num_prediction_steps, self.num_action_parameters] |
                        [B, self.num_prediction_steps, self.num_action_parameters] - ground truth actions tensor
            diffusion_schedule: diffusion schedule for noise addition

        Returns:
            gt_noise: [B, S, C] - ground truth noise tensor
            pred_noise: [B, S, C] - predicted noise tensor
        """
        b_sz = x.shape[0]
        # Check if input is just a single image
        if (len(x.shape) == 4) & (len(goal.shape) == 2):
            x = einops.repeat(x, 'b c h w -> b s c h w', s=1)
            goal = einops.repeat(goal, 'b g -> b s g', s=1)
            gt_actions = einops.repeat(gt_actions, 'b p a -> b s p a', s=1)
            s_sz = 0
        else:
            s_sz = x.shape[1]

        assert len(x.shape) == 5, f"Input tensor x should be of shape [B, S, C, H, W], but got {x.shape}"
        assert len(goal.shape) == 3, f"Goal tensor should be of shape [B, S, G], but got {goal.shape}"

        x, _ = self.forward_encoder(x, goal)

        # Reshape gt_actions and the conditioning vector to match the expected shape
        gt_actions = einops.rearrange(gt_actions, 'b s p a -> (b s) p a')
        x = einops.rearrange(x, 'b s c -> (b s) c')
        if isinstance(self.noise_decoder, _DitNoiseDecoder):
            x = einops.rearrange(x, 'bs c -> bs 1 c')

        # Sample a noise vector and time steps
        gt_noise = torch.randn_like(gt_actions)
        time_steps = torch.randint(
            low=0, high=len(diffusion_schedule), size=(b_sz * s_sz,), device=x.device
        ).long()

        noisy_actions = diffusion_schedule.add_noise(gt_actions, gt_noise, time_steps)
        predicted_noise = self.noise_decoder(
            noisy_actions=noisy_actions,
            time=time_steps,
            obs_enc=x,
        )

        # Reshape the ground truth noise and predicted noise to the original batch size and sequence length
        if s_sz != 0:
            gt_noise = einops.rearrange(gt_noise, '(b s) p a -> b s p a', b=b_sz, s=s_sz)
            predicted_noise = einops.rearrange(predicted_noise, '(b s) p a -> b s p a', b=b_sz, s=s_sz)

        out = {'target': gt_noise, 'prediction': predicted_noise}
        return out

    def predict(
            self,
            x: torch.Tensor,
            goal: torch.Tensor,
            diffusion_schedule: SchedulerMixin,
            num_actions: int = 1,
            diffusion_steps: int = 10,
            past_key_values: transformers.Cache = None,
        ) -> torch.Tensor:
        """
        Generate actions given an input image and goal.

        Args:
            x: [B, S, C, H, W] | [B, C, H, W] - input image tensor
            goal: [B, S, G] | [B, G] - goal tensor
            diffusion_schedule: the sampling scheme for generating actions
            num_actions: int - number of actions to predict per each observation
            diffusion_steps: int - number of diffusion steps to perform
            past_key_values: transformers.Cache - optional KV cache for inference
        Returns:
            out:
                [B, S, N, P, A] | [B, N, P, A] - predicted actions
                past_key_values: transformers.Cache - optional KV cache for inference
        """
        b_sz = x.shape[0]
        # Check if input is just a single image
        if (len(x.shape) == 4) & (len(goal.shape) == 2):
            x = einops.repeat(x, 'b c h w -> b s c h w', s=1)
            goal = einops.repeat(goal, 'b g -> b s g', s=1)
            s_sz = 0
        else:
            s_sz = x.shape[1]

        assert len(x.shape) == 5, f"Input tensor x should be of shape [B, S, C, H, W], but got {x.shape}"
        assert len(goal.shape) == 3, f"Goal tensor should be of shape [B, S, G], but got {goal.shape}"

        x, past_key_values = self.forward_encoder(x, goal, past_key_values=past_key_values)

        # Reshape the conditioning vector to match the expected shape
        x = einops.rearrange(x, 'b s c -> (b s) c')
        x = einops.repeat(x, 'bs c -> (bs n) c', n=num_actions)
        if isinstance(self.noise_decoder, _DitNoiseDecoder):
            x = einops.rearrange(x, 'bsn c -> bsn 1 c')
        bsn_sz = x.shape[0]

        # Prepare the diffusion schedule
        diffusion_schedule.set_timesteps(diffusion_steps)
        diffusion_schedule.alpha_cumprod = diffusion_schedule.alphas_cumprod.to(x.device)

        # Sample an initial noise vector
        noisy_actions = torch.randn(
            (bsn_sz, self.num_prediction_steps, self.num_actions_parameters),
            device=x.device,
        )

        # Perform the diffusion process
        for timestep in diffusion_schedule.timesteps:
            batch_timestep = einops.repeat(
                timestep.to(x.device), '-> b', b=bsn_sz
            )

            # Predict the noise at the current timestep
            predicted_noise = self.noise_decoder(
                noisy_actions=noisy_actions,
                time=batch_timestep,
                obs_enc=x,
            )

            # Compute the actions at the previous timestep
            noisy_actions = diffusion_schedule.step(
                model_output=predicted_noise, timestep=timestep, sample=noisy_actions
            ).prev_sample

        # Reshape the predicted actions to the original batch size and sequence length
        if s_sz != 0:
            noisy_actions = einops.rearrange(
                noisy_actions, '(b s n) p a -> b s n p a', b=b_sz, s=s_sz, n=num_actions
            )
        else:
            noisy_actions = einops.rearrange(
                noisy_actions, '(b n) p a -> b n p a', b=b_sz, n=num_actions
            )

        if past_key_values is not None:
            return {'out': noisy_actions, 'past_key_values': past_key_values}
        else:
            return {'out': noisy_actions}




if __name__ == '__main__':
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    model = DiffusionTransformerModel(
        encoder_type="theaiinstitute/theia-base-patch16-224-cddsv",
        sequence_length=100,
        noise_decoder_type='unet',
        late_goal_fusion=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    b, s = 4, 100
    x = torch.randn(b, s, 3, 224, 224, device=device)
    g = torch.randn(b, s, 2, device=device)
    actions = torch.randn(b, s, 10, 2, device=device)  # 10 prediction steps, 2 action parameters

    diffusion_schedule = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )

    # Train forward pass
    out = model(x, g, actions, diffusion_schedule)
    print("Train Output Shapes:", out['target'].shape, out['prediction'].shape)

    # Test inference
    with torch.inference_mode():
        out = model.predict(x, g, diffusion_schedule, num_actions=5, diffusion_steps=10)
        print("Predicted Actions Shape:", out['out'].shape)

    # # Test with a kv cache - this part won't work until a new kv cache implementation is done
    # import time
    # from earthrovers.common.models.deepseekv3.kv_cache import DeepseekV3RollingCache
    # cache = DeepseekV3RollingCache(window_length=10, qk_rope_head_dim=32)

    # with torch.inference_mode():
    #     for i in range(100):
    #         start = time.time()
    #         out = model.predict(x[[0], [i]], g[[0], [i]], diffusion_schedule, num_actions=5, diffusion_steps=10, past_key_values=cache)
    #         print('Time taken:', time.time() - start)
    #         cache = out['past_key_values']
    #         out = out['out']

