"""
The original Theia implementation has a Processor baked into the model.
This makes training inefficient and complicates downstream applications.

This script redefines the Theia model without the Processor.
"""

from typing import Any, Optional, List

import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoModel
import logging

# These imports require the Theia repo to be installed
from theia.models.feature_translators import build_feature_translator
from theia.models.rvfm import RobotVisionFM

import huggingface_hub
import safetensors

def build_backbone(model_name: str, pretrained: bool = False, image_size: int = 224, **kwargs: Any) -> nn.Module:
    """Build the backbone visual encoder of robot vision foundation model.

    Args:
        model_name (str): name of the model.
        pretrained (bool): whether to use pretrained weights. Defaults to False.
        image_size (int): size of the image. Assume a square image. Defaults to 224
        kwargs (Any): any kwargs specific to some models. For example,
            `num_reg_tokens` for `DeiTReg` when `"reg"` in `model_name`

    Returns:
        nn.Module: backbone network.
    """
    if "deit" in model_name:
        return DeiTNoProcessor(model_name=model_name, pretrained=pretrained, image_size=image_size)
    else:
        raise NotImplementedError(f"Requested {model_name} is not implemented.")

class DeiTNoProcessor(nn.Module):
    """DeiT model.

    Paper: Training data-efficient image transformers & distillation through attention
        https://arxiv.org/abs/2012.12877
    Huggingface Reference: https://huggingface.co/docs/transformers/en/model_doc/deit

    Attributes:
        model_name (str): name of the model.
        pretrained (bool): whether to use pretrained weights.
    """

    def __init__(
        self,
        model_name: str = "facebook/deit-small-patch16-224",
        pretrained: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.image_size = image_size
        model = AutoModel.from_pretrained(model_name)

        if pretrained:
            self.model = model
        else:
            deit_config = model.config
            self.model = AutoModel.from_config(deit_config)
            del model

        self.model.pooler = nn.Identity()

    def get_feature_size(
        self,
        keep_spatial: bool = False,
        return_torch_size: bool = False,
    ) -> torch.Size | tuple[int, ...]:
        """Get the size of the feature.

        Args:
            keep_spatial (bool): keep spatial dim of the feature shape. Defaults to False.
            return_torch_size (bool): if true, return torch.Size type. Defaults to False.

        Returns:
            torch.Size | tuple[int, ...]: returned feature shape.
        """
        with torch.inference_mode():
            image_size = (224, 224)
            x = torch.zeros((1, 3, *image_size), dtype=torch.float32)
            y = self.forward(x)[:, 1:]  # for getting feature size, discard cls token
            size = y.size()[1:][::-1]
            if keep_spatial:
                assert math.isqrt(size[-1])
                h = w = int(math.sqrt(size[-1]))
                size = (size[0], h, w)
                if return_torch_size:
                    size = torch.Size(size)
            return size

    def forward(
        self,
        x: torch.Tensor,
        do_resize: bool = True,
        interpolate_pos_encoding: Optional[bool] = None,
        do_rescale: bool = True,
        do_normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the model

        Args:
            x (torch.Tensor): model input.

            - arguments for self.processor. Details can be find at
                https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/deit#transformers.DeiTImageProcessor
            do_resize (bool): if do resizing in processor. Defaults to True.
            interpolate_pos_encoding (bool): if interpolate the positional embedding. Defaults to None.
            do_rescale (bool): if do rescaling (0-255 -> 0-1) in processor. Defaults to True.
            do_normalize (bool): if do normalize in processor. Defaults to True.

        Returns:
            torch.Tensor: model output.
        """
        y = self.model(x, interpolate_pos_encoding=interpolate_pos_encoding)
        return y.last_hidden_state

class TheiaNoProcessor(RobotVisionFM):

    """
    Attributes:
        backbone (str | nn.Module): backbone network. Defaults to "deit-small-patch16-224".
        pretrained (bool): whether to use pretrained weights. Default to False.
        translator (str | nn.Module): feature translator module. Defaults to "conv".
        target_feature_sizes (Optional[dict[str, torch.Size | tuple[int, ...]]]):
            a dict to hold target feature sizes.
        translator_kwargs (Optional[dict[str, Any]]): other keyword arguments to the translator.
        target_loss_weights (Optional[dict[str, float]]):
            weights to balance loss from different target models. If not specified, use even weights.
        checkpoint_path: (Optional[str]): filename of pretrained weights to load.
        feature_reduce_method: (Optional[str]): how to reduce the feature in downstream applications.
    """

    # Override the __init__ method
    def __init__(
        self,
        backbone: str | nn.Module = "facebook/deit-small-patch16-224",
        pretrained: bool = False,
        translator: str | nn.Module = "lconv",
        target_feature_sizes: Optional[dict[str, torch.Size | tuple[int, ...]]] = None,
        translator_kwargs: Optional[dict[str, Any]] = None,
        target_loss_weights: Optional[dict[str, float]] = None,
        checkpoint_path: Optional[str] = None,
        feature_reduce_method: Optional[str] = None,
        image_size: int = 224,
        **kwargs: Any
    ) -> None:
        super(RobotVisionFM, self).__init__()

        self.target_feature_sizes = target_feature_sizes
        self.preprocessor = None
        self.pretrained = pretrained

        # backbone
        self.image_size = image_size
        self.backbone: nn.Module = build_backbone(backbone, pretrained, image_size=image_size, **kwargs)

        self.final_spatial = None
        if hasattr(self.backbone, "final_spatial"):
            self.final_spatial = self.backbone.final_spatial

        # handle output feature (feature reduce)
        self.feature_reduce_method = feature_reduce_method
        self.no_cls = hasattr(self.backbone, "no_cls")
        self.num_reg_tokens = self.backbone.num_reg_tokens if hasattr(self.backbone, "num_reg_tokens") else 0

        # translator
        backbone_feature_size = self.backbone.get_feature_size(keep_spatial=True)
        if self.target_feature_sizes:
            translator_kwargs = {} if translator_kwargs is None else OmegaConf.to_container(translator_kwargs)
            translator_kwargs["backbone_feature_size"] = backbone_feature_size
            translator_kwargs["target_feature_sizes"] = target_feature_sizes
            self.translator = build_feature_translator(translator, **translator_kwargs)

        # loss
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.cos_target = torch.ones((1), dtype=torch.int, requires_grad=False)
        self.target_loss_weights = target_loss_weights

    def load_pretrained_weights(self, repo_id: str) -> None:
        """Load pretrained weights from Huggingface.

        Args:
            repo_id (str): Name of the checkpoint on Theia's huggingface page.
        """
        print(f"Loading pretrained weights from Huggingface hub: {repo_id}")
        ckpt = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
        )
        safetensors.torch.load_model(model=self, filename=ckpt, strict=False)


    def freeze_encoder(self) -> None:
        """Freeze backbone (encoder) `self.backbone`. """
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def freeze_everything(self) -> None:
        """Freeze all parameters in the model."""
        if hasattr(self, "translator"):
            self.freeze_translator()
        self.freeze_encoder()

class TheiaEncoder(nn.Module):

    """
    Wrapper around the Theia model to use it as an encoder.
    https://github.com/bdaiinstitute/theia/tree/main
    """

    theia_backbones = {
        "theaiinstitute/theia-tiny-patch16-224-cddsv": "facebook/deit-tiny-patch16-224",
        "theaiinstitute/theia-tiny-patch16-224-cdiv": "facebook/deit-tiny-patch16-224",
        "theaiinstitute/theia-small-patch16-224-cddsv": "facebook/deit-small-patch16-224",
        "theaiinstitute/theia-base-patch16-224-cddsv": "facebook/deit-base-patch16-224",
    }

    img_preprocess_type = "IMAGENET_STANDARD"

    def __init__(
        self,
        encoder_type: str,
        pretrained: bool = True,
        freeze: bool = True,
        input_size: List[int] = [3, 224, 224],
    ) -> None:
        super(TheiaEncoder, self).__init__()

        # Load the Theia model with AutoProcessor stripped out
        self.encoder = TheiaNoProcessor(backbone=self.theia_backbones[encoder_type])

        assert input_size[1] == input_size[2] == self.encoder.image_size, f"""
            Theia encoder expects {self.encoder.image_size}x{self.encoder.image_size} images
            """

        if pretrained:
            self.encoder.load_pretrained_weights(encoder_type)
        if freeze:
            self.encoder.freeze_everything()

        # Get the output token shape from the encoder
        with torch.no_grad():
            dummy_input = torch.zeros(
                (1, 3, self.encoder.image_size, self.encoder.image_size),
                dtype=torch.float32,
            )
            out_shape = self.encoder.forward_feature(dummy_input).shape

        self.encoder_dim = out_shape[-1]
        self._num_patches = out_shape[-2]
        self.out_size = int(math.sqrt(self._num_patches))

    @property
    def output_dim(self):
        return self.encoder_dim
    
    @property
    def output_featuremap_size(self):
        return [self.out_size, self.out_size]

    def forward(
        self, img: torch.tensor,
    ) -> torch.Tensor:
        # get the observation encoding
        encoding = self.encoder.forward_feature(img)
        return encoding

if __name__ == '__main__':
    """
    Test the TheiaNoProcessor model by comparing it with the original Theia model from Huggingface.
    """

    import numpy as np
    import torchvision.transforms.v2 as T
    transform = T.Compose([
            T.CenterCrop(224),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    dummy_in = torch.ones((3, 224, 224), dtype=torch.uint8)*100
    encoder = TheiaEncoder("theaiinstitute/theia-small-patch16-224-cddsv", pretrained=True)
    dummy_in = transform(dummy_in).unsqueeze(0)
    out = encoder(dummy_in)
    print(out.shape)
    print(encoder.output_dim)
    print(encoder.output_featuremap_size)

    