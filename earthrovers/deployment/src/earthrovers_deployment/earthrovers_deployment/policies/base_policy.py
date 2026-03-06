from typing import Dict

import torch
import inspect

from earthrovers_deployment.utils import get_image_transform, DummyModel

from rclpy.logging import get_logger

class BaseNavigationPolicy(torch.nn.Module):

    def __init__(
            self,
            config: Dict,
            device: str,
        ):
        super(BaseNavigationPolicy, self).__init__()
        self._config = config
        self.logger = get_logger("navigation_policy")
        self.device = device
        self.model = self._load_model()
        self.transform = get_image_transform(
            config['image_size'],
            config['normalization_type'],
            config['square_method'],
        )
    
    def _load_model(self):
        model_type = self._config['model_type']
        if model_type == 'DummyModel':
            model = DummyModel()
        model.eval()
        return model

    def _transform_image(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        return image
    
    def _prepare_goal_input(
            self,
            goal_distance: torch.Tensor,
            goal_direction: torch.Tensor,
        ):
        """
        Compute the distance and bearing to the goal position.

        Args:
            current_position: Current position of the robot. (Latitude, Longitude) in degrees, WGS84 / EPSG:4326.
            current_orientation: Current orientation of the robot. (Heading) in [-pi, +pi] radians, clockwise positive from North.
            goal_position: Goal position of the robot. (Latitude, Longitude) in degrees, WGS84 / EPSG:4326.
        """
        return torch.stack([goal_distance, goal_direction], dim=-1).unsqueeze(0).to(dtype=torch.float32, device=self.device)

    def forward(self, data: Dict):
        return None

def dynamic_load(root, model):
    module_path = f"{root.__name__}.{model}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]

    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseNavigationPolicy)]
    assert len(classes) == 1, classes
    return classes[0][1]