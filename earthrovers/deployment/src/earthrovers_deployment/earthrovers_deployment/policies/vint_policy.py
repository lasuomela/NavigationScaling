from typing import Dict

import torch
import einops

from earthrovers_deployment.policies.base_policy import BaseNavigationPolicy
from earthrovers_deployment.utils import DummyModel
from earthrovers.common.models.vint_like import ViNTLikeModel

class ViNTNavigationPolicy(BaseNavigationPolicy):

    def __init__(
            self,
            config: Dict,
            device: str,
        ):
        super(ViNTNavigationPolicy, self).__init__(config, device)
        self.obs_cache = None
    
    def _load_model(self, model_type: str = None, model_path: str = None):
        if model_type is None:
            model_type = self._config['model_type']
        if model_path is None:
            model_path = self._config.get('ckpt_path', self._config.get('model_path'))

        if model_type == 'DummyModel':
            model = DummyModel()
        elif "model_hf_name" in self._config:
            # Load the model from Hugging Face Hub
            model = ViNTLikeModel.from_pretrained(
                self._config['model_hf_name'],
            )
            model.to(self.device)
        else:
            # Load the lightning checkpoint
            ckpt = torch.load(model_path, weights_only=False)
            hparams = ckpt['hyper_parameters']['config']
            print(hparams)

            model = ViNTLikeModel(
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
            output = self.model(image, goal_input)
            x = output['out'].squeeze().cpu().numpy()
        return x
