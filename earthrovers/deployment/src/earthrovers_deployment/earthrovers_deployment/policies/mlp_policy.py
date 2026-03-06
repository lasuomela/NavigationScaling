from typing import List, Dict

import torch

from earthrovers_deployment.policies.base_policy import BaseNavigationPolicy
from earthrovers_deployment.utils import DummyModel
from earthrovers.common.models.mlp_model import MLPModel

class MLPNavigationPolicy(BaseNavigationPolicy):

    def __init__(
            self,
            config: Dict,
            device: str,
        ):
        super(MLPNavigationPolicy, self).__init__(config, device)
    
    def _load_model(self):
        model_type = self._config['model_type']
        if model_type == 'DummyModel':
            model = DummyModel()
        elif "model_hf_name" in self._config:
            # Load the model from Hugging Face Hub
            model = MLPModel.from_pretrained(
                self._config['model_hf_name'],
            )
        else:
            # Load the model from a local Lightning checkpoint
            ckpt = torch.load(
                self._config['ckpt_path'],
                weights_only=False,
            )
            hparams = ckpt['hyper_parameters']['config']
            model = MLPModel(
                encoder_type=hparams['encoder_type'],
                pretrained_encoder=False,
                goal_embed=hparams.get('embed_goal', False),
                state_fusion_type=hparams.get('state_fusion_type', 'mlp'),
                action_decoder_type=hparams.get('action_decoder_type', 'mlp'),
            )
            print(hparams)

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
        image = image.unsqueeze(0)
        goal_input = goal_input.unsqueeze(0)

        with torch.inference_mode():
            output = self.model(image, goal_input)
            if isinstance(output, dict):
                output = output['out']
            output = output.squeeze().cpu().numpy()
        return output
