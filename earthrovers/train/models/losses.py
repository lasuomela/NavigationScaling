import torch

@torch.compile
class ScaledMSELoss(torch.nn.Module):
    """
    MSE loss, but each loss of each batch element is scaled 
    proportional to angular velocity of the ground truth.
    """
    def __init__(self, scale_min=1.0, scale_max=10.0):
        super(ScaledMSELoss, self).__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
            target_cmds: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Args:
            prediction [B, S, H, 2]: Model output.
            target [B, S, H, 2]: Prediction target.
            target_cmds [B, S, H, 2]: Optional. Target commands (vx, omega). For use with diffusion models.

        Returns:
            loss: Scaled MSE loss.
        """
        unreduced_loss = self.mse(prediction, target)

        if target_cmds is None:
            angular_velocity = target[..., [1]]
        else:
            angular_velocity = target_cmds[..., [1]]
            
        angular_velocity_magnitude = angular_velocity.abs().max(dim=-2, keepdims=True).values

        # Angular velocity magnitude is always in [0, 1].
        scale = self.scale_min + angular_velocity_magnitude * (self.scale_max - self.scale_min)
        loss = unreduced_loss * scale
        return loss.mean()


