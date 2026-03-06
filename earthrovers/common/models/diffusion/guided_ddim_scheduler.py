"""
Implements the PSEUDOINVERSE-GUIDED DIFFUSION MODELS
FOR INVERSE PROBLEMS for DDIM scheduler.

https://openreview.net/pdf?id=9_gsMA8MRKQ
"""
from typing import Optional, Tuple, Union

from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
import torch

def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> torch.Tensor:
    """
    Prefix weights for the pseudo-inverse guidance.
    Copied from https://github.com/Physical-Intelligence/real-time-chunking-kinetix
    by Kevin Black et al.

    With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = min(start, end)
    if schedule == "ones":
        w = torch.ones(total)
    elif schedule == "zeros":
        w = torch.zeros(total)
    elif schedule in ("linear", "exp"):
        idx = torch.arange(total, dtype=torch.float32)
        denom = max((end - start + 1), 1)
        w = torch.clamp((start - 1 - idx) / denom + 1.0, 0.0, 1.0)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1.0)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return w * (torch.arange(total) < end).to(w.dtype)

class PseudoInverseGuidedDDIMScheduler(DDIMScheduler):
    """
    This class is heavily inspired by the jax implementation of
    pseudo-inverse guidance for flow-matching models by Kevin Black et al.
    ( https://github.com/Physical-Intelligence/real-time-chunking-kinetix )
    """

    def step(
        self,
        model: torch.Tensor,
        obs: torch.Tensor,
        sample: torch.Tensor,
        previous_actions: torch.Tensor,
        weights: torch.Tensor,
        max_guidance_weight: float,
        timestep: int,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model (`torch.nn.Module`):
                The model to be used for epsilon prediction.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://huggingface.co/papers/2010.02502
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://huggingface.co/papers/2010.02502
        if self.config.prediction_type == "epsilon":

            def denoiser(x_t: torch.Tensor) -> torch.Tensor:
                eps = model(
                    noisy_actions=x_t,
                    time=timestep,
                    obs_enc=obs,
                )
                pred_original_sample = (x_t - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
                return pred_original_sample, eps
            
            pred_original_sample, vjp_fun, model_output = torch.func.vjp(
                denoiser,
                sample,
                has_aux=True,
            )
            
            pred_epsilon = model_output
            if previous_actions is not None:
                error = previous_actions - pred_original_sample
                error = error * weights[:, None]
                # print('error:')
                # print(error)
                pinv_correction = vjp_fun(error)[0]
            else:
                pinv_correction = 0.0

            # print(pred_epsilon)
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be `epsilon`."
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        # print('')
        # print('variance alpha')
        # print(1 / (1 + variance))

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://huggingface.co/papers/2010.02502
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://huggingface.co/papers/2010.02502
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        # Add pseudoinverse guidance
        if previous_actions is not None:
            r_inv = 1 / (1 - alpha_prod_t) # r ** -2
            r_inv = min(r_inv, max_guidance_weight)
            # scale = 1 / error.pow(2).sum(dim=[1,2])
            # scale = 0.0001

            corr = alpha_prod_t**0.5 * alpha_prod_t_prev**0.5 * pinv_correction
            if timestep != self.timesteps[-1]:
                prev_sample = prev_sample + corr

        # prev_sample = torch.clamp(prev_sample, -1.0, 1.0)

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
