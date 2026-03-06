"""
A NoMAD-like architecture, adapted for PointGoals and using pretrained transformer vision encoders.
"""
import torch
import torch.nn as nn
import einops

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from huggingface_hub import PyTorchModelHubMixin

from earthrovers.common.models.layers import DINO, GoalEncoder, Flatten
from earthrovers.common.models.theia_encoder import TheiaEncoder
from earthrovers.common.models.vint_like import SeqEncoder
from earthrovers.common.models.diffusion.conditional_unet1d import ConditionalUnet1D

class NoMADLikeModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            encoder_type='facebook/dinov2-with-registers-small',
            input_size=(3, 224, 224),
            pretrained_encoder=True,
            freeze_encoder=False,
            sequence_length=6,
            num_flatten_channels=6,
            num_actions_parameters=2,
            num_prediction_steps=10,
            goal_input_dim=2,
        ):
        super(NoMADLikeModel, self).__init__()

        self.num_actions_parameters = num_actions_parameters
        self.num_prediction_steps = num_prediction_steps

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

        self.noise_decoder = ConditionalUnet1D(
            input_dim=num_actions_parameters,
            global_cond_dim=self.compressor.output_dim,
            down_dims=[64, 128, 256],
            cond_predict_scale=False,
        )

    def forward_encoder(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder part of the model.
        
        Args:
            x: [B, S, C, H, W]
            goal: [B, S, G]

        Returns:
            x: [B, C] - encoded state tensor
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
        # b s c -> b c
        x = self.seq_encoder(x, goal)
        return x
    
    def forward_denoiser(
            self,
            x: torch.Tensor,
            gt_actions: torch.Tensor,
            gt_noise: torch.Tensor,
            diffusion_schedule: SchedulerMixin,
    ):
        """
        Forward pass of the denoiser part of the model for training.

        Args:
            x: [B, C] - encoded state tensor
            gt_actions: [B, T, A] - ground truth actions tensor
            gt_noise: [B, T, A] - ground truth noise tensor
            diffusion_schedule: SchedulerMixin - diffusion schedule for noise addition

        Returns:
            gt_noise: [B, T, A] - ground truth noise tensor
            predicted_noise: [B, T, A] - predicted noise tensor
        """
        time_steps = torch.randint(
            low=0, high=len(diffusion_schedule), size=(x.shape[0],), device=x.device
        ).long()

        noisy_actions = diffusion_schedule.add_noise(gt_actions, gt_noise, time_steps)
        predicted_noise = self.noise_decoder(
            noisy_actions=noisy_actions,
            time=time_steps,
            obs_enc=x,
        )
        return gt_noise, predicted_noise
    

    def forward(
            self,
            x: torch.Tensor,
            goal: torch.Tensor,
            gt_actions: torch.Tensor,
            diffusion_schedule: SchedulerMixin,
            **kwargs, # Allow additional keyword arguments for training interface
        ) -> torch.Tensor:
        """
        Train forward pass of the model.

        Args:
            x: [B, S, C, H, W] - input image tensor
            goal: [B, S, G] - goal tensor
            gt_actions: 
                [B, S, self.num_prediction_steps, self.num_action_parameters] - ground truth actions tensor
            diffusion_schedule: diffusion schedule for noise addition

        Returns:
            target: [B, 1, T, C] - ground truth noise
            prediction: [B, 1, T, C] - predicted noise
        """
        x = self.forward_encoder(x, goal)

        # Sample a noise vector and time steps
        gt_actions = gt_actions[:, -1] # Pick the action corresponding to the last observation
        gt_noise = torch.randn_like(gt_actions)

        target, prediction = self.forward_denoiser(
            x=x,
            gt_actions=gt_actions,
            gt_noise=gt_noise,
            diffusion_schedule=diffusion_schedule,
        )

        # Reshape the outputs to match the expected dimensions
        prediction = einops.rearrange(prediction, 'b t c -> b 1 t c')
        target = einops.rearrange(target, 'b t c -> b 1 t c')

        out = {'target': target, 'prediction': prediction}
        return out
    
    def predict(
        self,
        x: torch.Tensor,
        goal: torch.Tensor,
        diffusion_schedule: SchedulerMixin,
        num_actions: int = 1,
        num_steps: int = 10,
        **kwargs, # Allow additional keyword arguments for training interface
    ) -> torch.Tensor:
        """
        Generate actions given an input image and goal.

        Args:
            x: [B, S, C, H, W] - input image tensor
            goal: [B, 1, G] - goal tensor
            diffusion_schedule: the sampling scheme for generating actions
            num_actions: int - number of actions to predict per each observation
            diffusion_steps: int - number of diffusion steps to perform
        Returns:
            out:
                [B, 1, N, P, A] - predicted actions
        """

        b_sz = x.shape[0]
        s_sz = x.shape[1]

        x = self.forward_encoder(x, goal)

        # Reshape the conditioning vector to match the expected shape
        x = einops.repeat(x, 'b c -> (b n) c', n=num_actions)
        bsn_sz = x.shape[0]


        # Prepare the diffusion schedule
        diffusion_schedule.set_timesteps(num_steps)
        diffusion_schedule.alpha_cumprod = diffusion_schedule.alphas_cumprod.to(x.device)

        # Sample an initial noise vector
        noisy_actions = torch.randn(
            (bsn_sz, self.num_prediction_steps, self.num_actions_parameters),
            device=x.device,
        )

        # Reverse the diffusion process
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
        noisy_actions = einops.rearrange(
            noisy_actions, '(b s n) p a -> b s n p a', b=b_sz, s=1, n=num_actions
        )
        return {'out': noisy_actions}




if __name__ == '__main__':
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    model = NoMADLikeModel(
        encoder_type="theaiinstitute/theia-small-patch16-224-cddsv",
    )

    diffusion_schedule = DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )

    x = torch.randn(1, 6, 3, 224, 224)
    g = torch.randn(1, 1, 2)
    actions = torch.randn(1, 1, 10, 2)

    print(model(x,g, actions, diffusion_schedule)['target'].shape)

    with torch.inference_mode():
        out = model.predict(x, g, diffusion_schedule, num_actions=3, num_steps=10)
        print(out['out'].shape)
