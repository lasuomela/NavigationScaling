from typing import List, Dict

import torch

from earthrovers_deployment.policies.base_policy import BaseNavigationPolicy
from earthrovers_deployment.utils import DummyModel
from earthrovers.common.models.deepseekv3.kv_cache import DeepseekV3RollingCache
from earthrovers.common.models.transformer_model_v2 import TransformerModelV2


class TransformerNavigationPolicy(BaseNavigationPolicy):

    def __init__(
            self,
            config: Dict,
            device: str,
        ):
        super(TransformerNavigationPolicy, self).__init__(config, device)

        if config['max_sequence_length'] > 1:
            self.kv_cache = DeepseekV3RollingCache(
                config['max_sequence_length'],
                config['rope_dimension'],
            )
        else:
            self.kv_cache = None

        self._previous_goal = None
    
    def reset_kv_cache(self):
        if (self.kv_cache is not None) and (self._previous_goal is not None):
            self.kv_cache = DeepseekV3RollingCache(
                self._config['max_sequence_length'],
                self._config['rope_dimension'],
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
            model = TransformerModelV2.from_pretrained(
                self._config['model_hf_name'],
            )
            model.to(self.device)
        else:
            # Load the lightning checkpoint
            ckpt = torch.load(model_path, weights_only=False)
            hparams = ckpt['hyper_parameters']['config']
            print(hparams)

            model = TransformerModelV2(
                encoder_type=hparams['encoder_type'],
                sequence_length=self._config['max_sequence_length'],
                predict_waypoints=self._config.get('predict_waypoints', False),
                num_waypoints=self._config.get('num_waypoints', 0),
                late_goal_fusion=self._config.get('late_goal_fusion', False),
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

    def forward(self, data: Dict):
        image = self._transform_image(data['image'])
        goal_input = self._prepare_goal_input(
            data['goal_distance'],
            data['goal_direction'],
        )

        # Check if the goal input has changed
        if not (data['goal_lat_lon'] == self._previous_goal):
            self.reset_kv_cache()
            print("Resetting KV cache due to goal change")
        self._previous_goal = data['goal_lat_lon']

        with torch.inference_mode():
            output = self.model(image, goal_input, self.kv_cache)
            x = output['out'].squeeze().cpu().numpy()
            if 'past_key_values' in output:
                self.kv_cache = output['past_key_values']
        return x
