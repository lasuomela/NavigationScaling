from typing import Dict, Any

import torch
import numpy as np

from earthrovers.common.models.theia_encoder import TheiaEncoder
from earthrovers.common.models.layers import DINO

def get_encoder_img_preprocess_type(encoder_type: str) -> Dict[str, str]:
    """
    Get the normalization values for an image encoder.
    """
    if 'theia' in encoder_type:
        return TheiaEncoder.img_preprocess_type
    elif 'dino' in encoder_type:
        return DINO.img_preprocess_type
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

@torch.jit.script
def compute_distance(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the distance between two points given their latitudes and longitudes.
    """
    R = 6371.0  # Earth radius in km

    # Convert degrees to radians
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    # Compute the change in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Compute the distance
    # http://www.movable-type.co.uk/scripts/latlong.html
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c

    # Limit the distance to [0, 1]
    distance = torch.clamp(distance, 0.0, 1.0)
    return distance

@torch.jit.script
def compute_direction(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the direction to a point given its latitude, longitude, and the current yaw.
    """
    # Convert degrees to radians
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    # Compute the change in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Compute the direction
    # http://www.movable-type.co.uk/scripts/latlong.html
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    direction = torch.atan2(y, x) - yaw

    # Normalize the direction to [-pi, pi]
    direction = (direction + np.pi) % (2 * np.pi) - np.pi

    # Normalize to [-1, 1]
    direction /= np.pi
    return direction