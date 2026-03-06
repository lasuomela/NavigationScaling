from typing import List, Dict

import torch
import einops

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from earthrovers_deployment.policies.base_policy import BaseNavigationPolicy
from earthrovers_deployment.utils import DummyModel
from earthrovers.common.models.nomad_like import NoMADLikeModel

class NoMADNavigationPolicy(BaseNavigationPolicy):

    def __init__(
            self,
            config: Dict,
            device: str,
        ):
        super(NoMADNavigationPolicy, self).__init__(config, device)
        self.obs_cache = None
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
    
    def _load_model(self, model_type: str = None, model_path: str = None):
        if model_type is None:
            model_type = self._config['model_type']
        if model_path is None:
            model_path = self._config.get('ckpt_path', self._config.get('model_path'))

        if model_type == 'DummyModel':
            model = DummyModel()
        elif "model_hf_name" in self._config:
            # Load the model from Hugging Face Hub
            model = NoMADLikeModel.from_pretrained(
                self._config['model_hf_name'],
            )
            model.to(self.device)
        else:
            # Load the lightning checkpoint
            ckpt = torch.load(model_path, weights_only=False)
            hparams = ckpt['hyper_parameters']['config']
            print(hparams)

            model = NoMADLikeModel(
                encoder_type=hparams['encoder_type'],
                sequence_length=self._config['max_sequence_length'],
            )

            # Strip only the preceding 'model.' from the state dict keys
            state_dict = {
                k[len('model.'):] if k.startswith('model.') else k: v
                for k, v in ckpt['state_dict'].items()
            }

            model.load_state_dict(state_dict)
            model.to(self.device)

        model.eval()
        return model

    def _update_obs_cache(self, obs: torch.Tensor):
        if self.obs_cache is None:
            self.obs_cache = [obs] * self._config['max_sequence_length']
        else:
            self.obs_cache = self.obs_cache[1:] + [obs]
        return torch.stack(self.obs_cache, dim=1)
    
    def _prepare_goal_input(self, goal_distance, goal_direction):
        goal =  super()._prepare_goal_input(goal_distance, goal_direction)
        return einops.rearrange(goal, 'b d -> b 1 d')

    def forward(self, data: Dict):
        image = self._transform_image(data['image'])
        image = self._update_obs_cache(image)
        goal_input = self._prepare_goal_input(
            data['goal_distance'],
            data['goal_direction'],
        )

        with torch.inference_mode():
            output = self.model.predict(
                image,
                goal_input,
                num_actions=self._config['num_actions'],
                num_steps=self._config['diffusion_steps'],
                diffusion_schedule=self.diffusion_scheduler,
            )
            x = output['out'][:,0].cpu().numpy()
        return x
