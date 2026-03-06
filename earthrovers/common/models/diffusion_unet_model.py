"""
A diffusion-based model for action generation from images and goals.
The model consists of an image encoder, a goal encoder,
a state encoder that fuses the image and goal encodings,
and a conditional UNet that predicts noise or flow fields for action generation.
The model supports both diffusion-based and conditional flow matching-based
training and inference,
with options for guided sampling using previous actions as context
and for temporal ensembling of the action chunks.
"""
import torch
import torch.nn as nn
import einops

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from huggingface_hub import PyTorchModelHubMixin

from earthrovers.common.models.layers import DINO, GoalEncoder, StateEncoder, Flatten
from earthrovers.common.models.theia_encoder import TheiaEncoder
from earthrovers.common.models.diffusion.conditional_unet1d import ConditionalUnet1D
from earthrovers.common.models.diffusion.guided_ddim_scheduler import get_prefix_weights

class DiffusionUnetModel(nn.Module, PyTorchModelHubMixin):
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
            use_goal=True,
            diffusion_unet_step_embed_dim=128,
            diffusion_unet_down_dims=[64, 128, 256],
            diffusion_unet_kernel_size=5,
            diffusion_unet_num_groups=8,
        ):
        super(DiffusionUnetModel, self).__init__()

        self.num_actions_parameters = num_actions_parameters
        self.num_prediction_steps = num_prediction_steps
        self.use_goal = use_goal

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

        if self.use_goal:
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

        self.noise_decoder = ConditionalUnet1D(
            input_dim=num_actions_parameters,
            global_cond_dim=self.compressor.output_dim,
            diffusion_step_embed_dim=diffusion_unet_step_embed_dim,
            down_dims=diffusion_unet_down_dims,
            kernel_size=diffusion_unet_kernel_size,
            n_groups=diffusion_unet_num_groups,
            cond_predict_scale=True,
        )

    def forward_encoder(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder part of the model.
        
        Args:
            x: [B, C, H, W] - input image tensor
            goal: [B, G] - goal tensor

        Returns:
            x: [B, C] - encoded state tensor
        """
        
        x = einops.rearrange(x, 'b 1 c h w -> b c h w')
        goal = einops.rearrange(goal, 'b 1 c -> b c')
        
        x = self.img_encoder(x)
        x = self.compressor(x)
        if self.use_goal:
            goal = self.goal_encoder(goal)
            x = self.state_encoder(x, goal)
        return x
    
    def _forward_denoiser(
            self,
            x: torch.Tensor,
            gt_actions: torch.Tensor,
            gt_noise: torch.Tensor,
            diffusion_schedule: SchedulerMixin,
    ):
        """
        Forward pass of the denoising diffusion part of the model for training.

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

    def _forward_cfm(
            self,
            x: torch.Tensor,
            gt_actions: torch.Tensor,
            gt_noise: torch.Tensor,
            flow_matcher: ConditionalFlowMatcher,
        ) -> torch.Tensor:
        """
        Forward pass of the flow prediction model for CFM training.

        Args:
            x: [B, C] - encoded state tensor
            gt_actions: [B, T, A] - ground truth actions tensor
            gt_noise: [B, T, A] - ground truth noise tensor
            flow_matcher: ConditionalFlowMatcher - the flow matcher for generating actions

        Returns:
            ut: [B, T, A] - ground truth conditional flow field
            vt: [B, T, A] - predicted conditional flow field
        """
        time_steps, noisy_actions, ut = flow_matcher.sample_location_and_conditional_flow(x0=gt_noise, x1=gt_actions)

        # Predict the flow at the sampled time steps
        vt = self.noise_decoder(
            noisy_actions=noisy_actions,
            time=time_steps,
            obs_enc=x,
        )
        return ut, vt
    
    def forward(
            self,
            x: torch.Tensor,
            goal: torch.Tensor,
            gt_actions: torch.Tensor,
            diffusion_schedule: SchedulerMixin = None,
            flow_matcher: ConditionalFlowMatcher = None,
        ) -> torch.Tensor:
        """
        Train forward pass of the model.

        Args:
            x: [B, 1, C, H, W] - input image tensor
            goal: [B, 1, G] - goal tensor
            gt_actions: 
                [B, 1, self.num_prediction_steps, self.num_action_parameters] - ground truth actions tensor
            diffusion_schedule: diffusion schedule for noise addition
            flow_matcher: ConditionalFlowMatcher - the flow matcher for generating actions

        Returns:
            target: [B, 1, T, C] - ground truth noise / conditional flow field
            prediction: [B, 1, T, C] - predicted noise / conditional flow field
        """
        s_sz = x.shape[1]
        assert s_sz == 1, "Input tensor sequence length should be 1."
        x = self.forward_encoder(x, goal)

        # Sample a noise vector and time steps
        gt_actions = einops.rearrange(gt_actions, 'b 1 t c -> b t c')
        gt_noise = torch.randn_like(gt_actions)

        if diffusion_schedule is not None:
            # Diffusion parametrization
            target, prediction = self._forward_denoiser(
                x=x,
                gt_actions=gt_actions,
                gt_noise=gt_noise,
                diffusion_schedule=diffusion_schedule,
            )
        elif flow_matcher is not None:
            # Conditional flow matching
            target, prediction = self._forward_cfm(
                x=x,
                gt_actions=gt_actions,
                gt_noise=gt_noise,
                flow_matcher=flow_matcher,
            )
        else:
            raise ValueError("Either diffusion_schedule or flow_matcher must be provided.")

        # Reshape the outputs to match the expected dimensions
        prediction = einops.rearrange(prediction, 'b t c -> b 1 t c')
        target = einops.rearrange(target, 'b t c -> b 1 t c')        

        out = {'target': target, 'prediction': prediction}
        return out

    def _predict_denoiser(
            self,
            x: torch.Tensor,
            bsn_sz: int,
            diffusion_schedule: SchedulerMixin,
            diffusion_steps: int = 10,
    ):
        """
        Generate actions by reverse diffusion process using an observation as context.

        Args:
            x: [B, C] - encoded state tensor
            bsn_sz: int - batch size for sampling actions
            diffusion_schedule: SchedulerMixin - the sampling scheme for generating actions
            diffusion_steps: int - number of diffusion steps to perform
        """

        # Prepare the diffusion schedule
        diffusion_schedule.set_timesteps(diffusion_steps)
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

        return noisy_actions
    
    def _predict_denoiser_temporal_ensemble(
            self,
            x: torch.Tensor,
            bsn_sz: int,
            previous_actions: torch.Tensor,
            diffusion_schedule: SchedulerMixin,
            diffusion_steps: int = 10,
            execution_horizon: int = 2,
            ensemble_size: int = 4,
            k: float = 0.4,
    ):
        """
        Generate actions by reverse diffusion process using an observation as context
        and ensembling the generated action chunks with the previous actions.

        Args:
            x: [B, C] - encoded state tensor
            bsn_sz: int - batch size for sampling actions
            previous_actions: [B, E, T, C] - previously generated actions
            diffusion_schedule: SchedulerMixin - the sampling scheme for generating actions
            diffusion_steps: int - number of diffusion steps to perform
            execution_horizon: int - number of steps to look ahead for action execution
        """

        latest_action = self._predict_denoiser(
            x=x,
            bsn_sz=bsn_sz,
            diffusion_schedule=diffusion_schedule,
            diffusion_steps=diffusion_steps,
        )

        ## Update 'previous_actions'
        if previous_actions is not None:
            # Drop action from the oldest inference step
            previous_actions = previous_actions[:, 1:, :, :]

            # Drop the parts of actions corresponding to the 'executed' steps
            previous_actions = previous_actions[:, :, execution_horizon:, :]

            # Right-pad the previous actions to correct shape with nan
            previous_actions = torch.concat(
                [
                    previous_actions,
                    torch.full_like(previous_actions[:, :, :execution_horizon, :], float('nan')),
                ],
                dim=2
            )

            # Concatenate the latest actions to the previous actions
            previous_actions = torch.concat(
                [
                    previous_actions,
                    latest_action[:, None, :, :],
                ],
                dim=1
            )
        else:
            previous_actions = torch.full(
                (bsn_sz, ensemble_size-1, self.num_prediction_steps, self.num_actions_parameters),
                float('nan'),
                device=x.device
            )
            previous_actions = torch.concat(
                [
                    previous_actions,
                    latest_action[:, None, :, :],
                ],
                dim=1
            )

        assert previous_actions.shape[1] == ensemble_size, previous_actions.shape

        # Compute the exponential weights for each ensemble column
        mask = ~torch.isnan(previous_actions)
        rows = torch.arange(mask.size(1), dtype=previous_actions.dtype, device=mask.device).unsqueeze(1).unsqueeze(1).expand_as(previous_actions)
        row_counts = (~mask).cumsum(dim=1)
        values = rows - row_counts
        out = torch.where(mask, values, torch.full_like(previous_actions, float('nan')))
        exp_weights = torch.exp(k * out)
        exp_weights = exp_weights / exp_weights.nansum(dim=1, keepdim=True)

        ensembled_action = (previous_actions * exp_weights).nansum(dim=1)

        return ensembled_action, previous_actions


    @torch.inference_mode(False)
    @torch.no_grad()
    def _predict_denoiser_guided(
            self,
            obs: torch.Tensor,
            bsn_sz: int,
            previous_actions: torch.Tensor,
            diffusion_schedule: SchedulerMixin,
            diffusion_steps: int = 10,
            max_guidance_weight: float = 5.0,
            inference_delay: int = 2,
            execution_horizon: int = 2,
            prefix_attention_horizon: int = 6,
            prefix_attention_schedule: str = "exp",
    ):
        """
        Generate actions by reverse diffusion process,
        with pseudo-inverse guidance / "real-time chunking",
        using an observation as context
        """

        # Discard the 'executed' actions from the previous actions
        if previous_actions is not None:
            previous_actions = torch.concat(
                [
                    previous_actions[:, execution_horizon:],
                    torch.zeros_like(previous_actions[:, :execution_horizon]),
                ],
                dim=1
            )
            assert previous_actions.shape[1] == self.num_prediction_steps, previous_actions.shape

        weights = get_prefix_weights(
            inference_delay,
            prefix_attention_horizon,
            self.num_prediction_steps,
            prefix_attention_schedule
        ).to(obs.device)

        # Sample an initial noise vector
        noisy_actions = torch.randn(
            (bsn_sz, self.num_prediction_steps, self.num_actions_parameters),
            device=obs.device,
        )

        # Prepare the diffusion schedule
        diffusion_schedule.set_timesteps(diffusion_steps)
        diffusion_schedule.alphas_cumprod = diffusion_schedule.alphas_cumprod.to(obs.device)

        # Reverse the diffusion process
        for timestep in diffusion_schedule.timesteps:
            batch_timestep = einops.repeat(
                timestep.to(obs.device), '-> b', b=bsn_sz
            )

            # Compute the actions at the previous timestep
            noisy_actions = diffusion_schedule.step(
                model=self.noise_decoder,
                obs=obs,
                sample=noisy_actions,
                previous_actions=previous_actions,
                weights=weights,
                max_guidance_weight=max_guidance_weight,
                timestep=batch_timestep, 
            ).prev_sample

        return noisy_actions
    
    def _predict_cfm(
            self,
            x: torch.Tensor,
            bsn_sz: int,
            flow_steps: int = 10,
    ):
        """
        Generate actions by conditional flow matching using an observation as context.
        """
        # Sample an initial noise vector
        noisy_actions = torch.randn(
            (bsn_sz, self.num_prediction_steps, self.num_actions_parameters),
            device=x.device,
        )

        t = 0
        dt = 1.0 / (flow_steps - 1)

        # Solve the ODE with Euler method
        for _ in range(flow_steps-1):
            noisy_actions = noisy_actions + dt * self.noise_decoder(
                noisy_actions=noisy_actions,
                time=torch.tensor([t], device=x.device).repeat(bsn_sz),
                obs_enc=x,
            )
            t += dt
        
        return noisy_actions
    
    @torch.inference_mode(False)
    @torch.no_grad()
    def _predict_cfm_guided(
            self,
            obs: torch.Tensor,
            bsn_sz: int,
            previous_actions: torch.Tensor,
            flow_steps: int = 10,
            max_guidance_weight: float = 5.0,
            inference_delay: int = 2,
            exceution_horizon: int = 2,
            prefix_attention_horizon: int = 6,
            prefix_attention_schedule: str = "exp",
    ):
        """
        Generate actions by conditional flow matching,
        with pseudo-inverse guidance / "real-time chunking",
        using an observation as context.
        """
        # Discard the 'executed' actions from the previous actions
        previous_actions = torch.concat(
            [
                previous_actions[:, exceution_horizon:],
                torch.zeros_like(previous_actions[:, :exceution_horizon]),
            ],
            dim=1
        )
        assert previous_actions.shape[1] == self.num_prediction_steps, previous_actions.shape

        weights = get_prefix_weights(
            inference_delay,
            prefix_attention_horizon,
            self.num_prediction_steps,
            prefix_attention_schedule
        ).to(obs.device)

        # Sample an initial noise vector
        x_t = torch.randn(
            (bsn_sz, self.num_prediction_steps, self.num_actions_parameters),
            device=obs.device,
        )
        y = previous_actions

        t = torch.tensor(0.0, device=obs.device)
        dt = 1.0 / (flow_steps - 1)

        for _ in range(flow_steps-1):
            v_totals = []
            t_b = torch.tensor([t], device=obs.device)
            for b in range(bsn_sz):
                obs_b = obs[b:b+1]
                x_b = x_t[b:b+1]

                def denoiser(x_t: torch.Tensor) -> torch.Tensor:
                    v_t = self.noise_decoder(
                        noisy_actions=x_t,
                        time=t_b,
                        obs_enc=obs_b,
                    )
                    return x_t + (1.0 - t_b) * v_t, v_t
                
                x_1, vjp_fun, v_t = torch.func.vjp(
                    denoiser,
                    x_b,
                    has_aux=True,
                )

                error = (y - x_1) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]

                # Constants from RTC paper
                # TODO: Add Cobot blog post correction for r
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)

                if t == 0:
                    c = max_guidance_weight
                else:
                    c = torch.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
                guidance_weight = min(c * inv_r2, max_guidance_weight)

                v_totals.append((v_t + guidance_weight * pinv_correction))
                
            v_t = torch.cat(v_totals, dim=0).detach()
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t

    def predict(
            self,
            x: torch.Tensor,
            goal: torch.Tensor,
            parametrization: str = 'eps',
            diffusion_schedule: SchedulerMixin = None,
            num_actions: int = 1,
            num_steps: int = 10,
            previous_actions: torch.Tensor = None,
            max_guidance_weight: float = 5.0,
            inference_delay: int = 2,
            execution_horizon: int = 2,
            prefix_attention_horizon: int = 6,
            prefix_attention_schedule: str = "exp",
        ) -> torch.Tensor:
        """
        Generate actions given an input image and goal.

        Args:
            x: [B, 1, C, H, W] - input image tensor
            goal: [B, 1, G] - goal tensor
            parametrization: str - 'eps', 'eps_ensemble', 'eps_guided', 'cfm' or 'cfm_guided'.
                The sampling scheme to use for generating actions.
            diffusion_schedule: the sampling scheme for generating actions
            num_actions: int - number of actions to predict per each observation
            num_steps: int - number of diffusion/cfm steps to perform
            previous_actions: [B, E, T, C] - previously generated actions for guidance
            max_guidance_weight: float - pseudo-inverse guidance: maximum weight for the guidance term
            inference_delay: int - pseudo-inverse guidance: number of chunk actions executed while inferring the new chunk
            execution_horizon: int - pseudo-inverse guidance: number of steps to look ahead for action execution
            prefix_attention_horizon: int - pseudo-inverse guidance: number of steps to
                apply prefix attention to.
            prefix_attention_schedule: str - pseudo-inverse guidance: schedule for computing the prefix attention weights
        Returns:
            out:
                [B, 1, N, P, A] - predicted actions
        """

        b_sz = x.shape[0]
        # Check if input is just a single image
        if (len(x.shape) == 4) & (len(goal.shape) == 2):
            x = einops.repeat(x, 'b c h w -> b s c h w', s=1)
            goal = einops.repeat(goal, 'b g -> b s g', s=1)
            s_sz = 0
        else:
            s_sz = x.shape[1]

        assert s_sz in [0, 1], f"If provided, input sequence length should be 1, but got {s_sz}."
        assert len(x.shape) == 5, f"Input tensor x should be of shape [B, S, C, H, W], but got {x.shape}"
        assert len(goal.shape) == 3, f"Goal tensor should be of shape [B, S, G], but got {goal.shape}"

        x = self.forward_encoder(x, goal)

        # Reshape the conditioning vector to match the expected shape
        x = einops.repeat(x, 'b c -> (b n) c', n=num_actions)
        bsn_sz = x.shape[0]

        if parametrization == 'eps':
            # Generate actions using reverse diffusion process
            assert diffusion_schedule is not None, "diffusion_schedule must be provided for diffusion parametrization."
            noisy_actions = self._predict_denoiser(
                x=x,
                bsn_sz=bsn_sz,
                diffusion_schedule=diffusion_schedule,
                diffusion_steps=num_steps,
            )
        elif parametrization == 'eps_ensemble':
            assert num_actions == 1, "num_actions must be 1 for eps_ensemble."
            noisy_actions, previous_actions_ensemble = self._predict_denoiser_temporal_ensemble(
                x=x,
                bsn_sz=bsn_sz,
                previous_actions=previous_actions,
                diffusion_schedule=diffusion_schedule,
                diffusion_steps=num_steps,
                execution_horizon=execution_horizon,
                ensemble_size=4,
            )
        elif parametrization == 'eps_guided':
            # Generate actions using guided reverse diffusion process
            assert diffusion_schedule is not None, "diffusion_schedule must be provided for diffusion parametrization."
            noisy_actions = self._predict_denoiser_guided(
                obs=x,
                bsn_sz=bsn_sz,
                previous_actions=previous_actions,
                diffusion_schedule=diffusion_schedule,
                diffusion_steps=num_steps,
                max_guidance_weight=max_guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                prefix_attention_horizon=prefix_attention_horizon,
                prefix_attention_schedule=prefix_attention_schedule,
            )

        elif parametrization == 'cfm':
            # Generate actions using conditional flow matching
            noisy_actions = self._predict_cfm(
                x=x,
                bsn_sz=bsn_sz,
                flow_steps=num_steps,
            )
        elif parametrization == 'cfm_guided':
            # Generate actions using guided conditional flow matching
            if previous_actions is None:
                print("No previous actions provided, using CFM to generate initial actions.")
                noisy_actions = self._predict_cfm(
                    x=x,
                    bsn_sz=bsn_sz,
                    flow_steps=num_steps,
                )
            else:
                assert len(previous_actions.shape) == 3, previous_actions.shape
                noisy_actions = self._predict_cfm_guided(
                    obs=x,
                    bsn_sz=bsn_sz,
                    previous_actions=previous_actions,
                    flow_steps=num_steps,
                    max_guidance_weight=max_guidance_weight,
                    inference_delay=inference_delay,
                    exceution_horizon=execution_horizon,
                    prefix_attention_horizon=prefix_attention_horizon,
                    prefix_attention_schedule=prefix_attention_schedule,
                )

        else:
            raise ValueError("parametrization should be either 'eps' or 'cfm'.")

        # Reshape the predicted actions to the original batch size and sequence length
        if s_sz != 0:
            noisy_actions = einops.rearrange(
                noisy_actions, '(b s n) p a -> b s n p a', b=b_sz, s=s_sz, n=num_actions
            )
        else:
            noisy_actions = einops.rearrange(
                noisy_actions, '(b n) p a -> b n p a', b=b_sz, n=num_actions
            )

        if parametrization == 'eps_ensemble':
            return {'out': noisy_actions, 'action_ensemble': previous_actions_ensemble}
        
        return {'out': noisy_actions}
        

if __name__ == '__main__':
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    model = DiffusionUnetModel(
        encoder_type="theaiinstitute/theia-small-patch16-224-cddsv",
        diffusion_unet_down_dims=[128, 256, 512],
    ).to('cuda')

    diffusion_schedule = DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )

    x = torch.randn(1, 1, 3, 224, 224).to('cuda')
    g = torch.randn(1, 1, 2).to('cuda')
    actions = torch.randn(1, 1, 10, 2).to('cuda')
    previous_actions = torch.concat(
        [
            torch.ones( 1, 10, 1),
            torch.ones(1, 10, 1),
        ],
        dim=-1
    ).to('cuda')

    import time
    with torch.inference_mode():
        out = model.predict(
            x=x,
            goal=g,
            parametrization='eps_ensemble',
            diffusion_schedule=diffusion_schedule,
            num_actions=1,
            num_steps=10,
            previous_actions=None,
        )
        print(out['action_ensemble'])
        start = time.time()
        out = model.predict(
            x=x,
            goal=g,
            parametrization='eps_ensemble',
            diffusion_schedule=diffusion_schedule,
            num_actions=1,
            num_steps=10,
            previous_actions=out['action_ensemble'],
        )
        print(f"Prediction time: {time.time() - start:.4f} seconds")
        print(out['action_ensemble'])
        print(out['out'].squeeze().round(decimals=2))

