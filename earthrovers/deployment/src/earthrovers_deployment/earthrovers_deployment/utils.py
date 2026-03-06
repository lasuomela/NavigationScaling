from typing import List

import torch
import torchvision.transforms.v2 as T

from earthrovers.common.constants import IMAGE_NORMALIZATION_VALUES

class DummyModel(torch.nn.Module):
    def forward(self, *args, **kwargs):
        out = torch.zeros(1, 2, dtype=torch.float32)
        out[:, 0] = 0.5
        return out

def get_image_transform(
        image_size: List[int],
        normalization_type: str,
        square_method: str = None,
    ) -> T.Compose:
    """
    Get the image transformation pipeline for the model.

    Args:
        image_size: (H, W) The size to resize the image to.
        normalization_type: The type of normalization to apply.
        square_method: The method to make the image square. Options: 'pad', None

    Returns:
        T.Compose: The image transformation pipeline.
    """

    transforms = []
    transforms += [T.ToImage()]
    if square_method == 'pad':
        # Keep the aspect ratio and pad the image to make it square
        assert image_size[0] == image_size[1], "Image size must be square when using 'pad' square_method"
        # Resize shorter side to image_size
        transforms += [T.Resize(image_size[0])]
        # Zero-pad to square
        transforms += [T.CenterCrop(image_size)]
    elif square_method == 'resize':
        # Resize the image to the specified size
        transforms += [T.Resize(image_size)]
    else:
        raise ValueError(f"Invalid square_method: {square_method}")
    
    # Convert the image to a tensor
    transforms += [T.ToDtype(torch.float32, scale=True)]
    # Normalize the image
    transforms += [T.Normalize(**IMAGE_NORMALIZATION_VALUES[normalization_type])]
    return T.Compose(transforms)