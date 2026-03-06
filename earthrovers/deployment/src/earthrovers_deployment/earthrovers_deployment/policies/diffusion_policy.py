from typing import Dict

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from earthrovers_deployment.policies.base_policy import BaseNavigationPolicy
from earthrovers_deployment.utils import DummyModel
from earthrovers.common.models.deepseekv3.kv_cache import DeepseekV3RollingCache
from earthrovers.common.models.diffusion_transformer_model import DiffusionTransformerModel
from earthrovers.common.models.diffusion_unet_model import DiffusionUnetModel
from earthrovers.common.models.diffusion.guided_ddim_scheduler import PseudoInverseGuidedDDIMScheduler

class DiffusionNavigationPolicy(BaseNavigationPolicy):

    def __init__(
            self,
            config: Dict,
            device: str,
        ):
        super(DiffusionNavigationPolicy, self).__init__(config, device)

        if config['max_sequence_length'] > 1:
            self.kv_cache = DeepseekV3RollingCache(
                config['max_sequence_length'],
                config['rope_dimension'],
            )
        else:
            self.kv_cache = None

        self._previous_goal = None
        self._previous_actions = None

        if config['parametrization'] in ['eps', 'eps_ensemble']:
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
        elif config['parametrization'] == 'eps_guided':
            self.diffusion_scheduler = PseudoInverseGuidedDDIMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        else:
            self.diffusion_scheduler = None
    
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
            if self._config['net_type'] == 'single':
                model = DiffusionUnetModel.from_pretrained(
                    self._config['model_hf_name'],
                )
            elif self._config['net_type'] == 'sequence':
                model = DiffusionTransformerModel.from_pretrained(
                    self._config['model_hf_name'],
                )
            else:
                raise ValueError(f"Unknown noise prediction network type: {self._config['net_type']}")
            model.to(self.device)
        else:
            # Load the lightning checkpoint
            ckpt = torch.load(model_path, weights_only=False)
            hparams = ckpt['hyper_parameters']['config']
            print(hparams)

            if self._config['net_type'] == 'single':
                model = DiffusionUnetModel(
                    encoder_type=hparams['encoder_type'],
                    diffusion_unet_down_dims=hparams['diffusion_unet_down_dims'],
                )
            elif self._config['net_type'] == 'sequence':
                model = DiffusionTransformerModel(
                    encoder_type=hparams['encoder_type'],
                    sequence_length=self._config['max_sequence_length'],
                    noise_decoder_type=self._config['noise_decoder'],
                    diffusion_unet_down_dims=hparams['diffusion_unet_down_dims'],
                    )
            else:
                raise ValueError(f"Unknown noise prediction network type: {self._config['net_type']}")

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
            kwargs = {
                'num_actions': self._config.get('num_actions', 6),
                'num_steps': self._config['diffusion_steps'],
                'parametrization': self._config.get('parametrization', 'eps'),
            }
            if self.kv_cache is not None:
                kwargs['past_key_values'] = self.kv_cache

            if kwargs['parametrization'] in ['cfm_guided', 'eps_guided']:
                kwargs['previous_actions'] = self._previous_actions
                kwargs['max_guidance_weight'] = self._config['max_guidance_weight']
                kwargs['inference_delay'] = self._config['inference_delay']
                kwargs['execution_horizon'] = self._config['execution_horizon']
                kwargs['prefix_attention_horizon'] = self._config['prefix_attention_horizon']
                kwargs['prefix_attention_schedule'] = self._config['prefix_attention_schedule']
                kwargs['num_steps'] = self._config['guided_diffusion_steps']

            if kwargs['parametrization'] == 'eps_ensemble':
                kwargs['previous_actions'] = self._previous_actions
                kwargs['execution_horizon'] = self._config['execution_horizon']

            output = self.model.predict(
                image,
                goal_input,
                diffusion_schedule=self.diffusion_scheduler,
                **kwargs
            )
            x = output['out']

            if kwargs['parametrization'] in ['cfm_guided', 'eps_guided']:
                assert x.shape[1] == 1, "At the moment, only single action prediction is supported for pseudoinverse guided flow matching."
                self._previous_actions = x[:, 0, :]
            elif kwargs['parametrization'] == 'eps_ensemble':
                assert x.shape[1] == 1, "At the moment, only single action prediction is supported for eps_ensemble."
                self._previous_actions = output['action_ensemble']

            if 'past_key_values' in output:
                self.kv_cache = output['past_key_values']

        return x.cpu().numpy()
