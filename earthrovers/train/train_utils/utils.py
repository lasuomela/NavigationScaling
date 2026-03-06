from lightning.pytorch.callbacks import Callback
import random
    
from earthrovers.train.train_utils.visualization import plot_obs_and_controls, unnormalize_img

class LogImgsCallback(Callback):
    """
    Callback to log images during training/validation.
    """
    def __init__(
            self,
            img_mean: list,
            img_std: list,
            num_batches_to_log: int = 10,
            imgs_per_batch: int = 1,
            log_trigger: str = "step", # "step" or "epoch"
        ):
        self.num_batches_to_log = num_batches_to_log
        self.imgs_per_batch = imgs_per_batch
        self.log_trigger = log_trigger
        self.img_mean = img_mean
        self.img_std = img_std

        assert self.log_trigger in ["step", "epoch"], f"Invalid log_trigger: {self.log_trigger}"
        assert self.imgs_per_batch > 0, f"Invalid imgs_per_batch: {self.imgs_per_batch}"
        
        self.viz_batch_idxs = None
        self.viz_queue = []

    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
        if trainer.is_global_zero and (not trainer.sanity_checking):
            if (batch_idx in self.viz_batch_idxs) and ('actions' in outputs):
                n = self.imgs_per_batch
                # Pick idxs of n random images from the batch
                viz_idxs = sorted(random.sample(range(batch["front_camera"].shape[0]), n))

                # If each batch entry contains a sequence, pick the last entry of the sequence
                if len(batch["front_camera"].shape) == 5:
                    for key in ["front_camera", "goal_input", "target", "current_position", "current_yaw", "goal_position", "waypoints"]:
                        if key in batch:
                            batch[key] = batch[key][:, -1]

                    for key in ["actions", "waypoints"]:
                        if key in outputs:
                            outputs[key] = outputs[key][:, -1]

                # Build per-sample dictionaries
                viz_images = [img for img in unnormalize_img(
                    batch["front_camera"][viz_idxs],
                    self.img_mean,
                    self.img_std,
                    batch["flipped"][viz_idxs])
                ]
                viz_batch = [
                    {
                        "ride_id": batch["ride_id"][0][i],
                        "obs_img": img,
                        "goal_input": batch["goal_input"][i],
                        "gt_controls": batch["target"][i],
                        "pred_controls": outputs["actions"][i],
                        "current_position": batch["current_position"][i],
                        "current_yaw": batch["current_yaw"][i],
                        "goal_position": batch["goal_position"][i],
                        "flipped": batch["flipped"][i],
                        "gt_waypoints": batch["waypoints"][i] if "waypoints" in batch else None,
                        "pred_waypoints": outputs["waypoints"][i] if "waypoints" in outputs else None,
                    }
                    for i, img in zip(viz_idxs, viz_images)
                ]

                # Generate figures
                figs = [plot_obs_and_controls(**entry) for entry in viz_batch]

                # Option 1: log images with `WandbLogger.log_image`
                if self.log_trigger == "step":
                    trainer.logger.log_image(key=f"{trainer.state.stage.value}_images", images=figs)

                elif self.log_trigger == "epoch":
                    self.viz_queue.extend(figs)

    def _on_epoch_start(self, num_batches):
        num_batches = num_batches[0] if isinstance(num_batches, list) else num_batches
        self.viz_batch_idxs = random.sample(range(num_batches), min(self.num_batches_to_log, num_batches))

    def _on_epoch_end(self, trainer, pl_module):
        if self.log_trigger == "epoch":
            trainer.logger.log_image(key=f"{trainer.state.stage.value}_images", images=self.viz_queue)
            self.viz_queue = []

class LogValImgsCallback(LogImgsCallback):
    def on_validation_epoch_start(self, trainer, pl_module):
        """
        Sample batch idxs from which to log images.
        """
        self._on_epoch_start(trainer.num_val_batches)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._on_epoch_end(trainer, pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called when the validation batch starts."""
        if not trainer.sanity_checking and (batch_idx in self.viz_batch_idxs):
            # Set a flag to indicate that the mode prediction output will be visualized
            batch["visualize_prediction"] = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called when the validation batch ends."""
        if not trainer.sanity_checking:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

class LogTrainImgsCallback(LogImgsCallback):
    def on_train_epoch_start(self, trainer, pl_module):
        """
        Sample batch idxs from which to log images.
        """
        self._on_epoch_start(trainer.num_training_batches)

    def on_train_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called when the train batch starts."""
        if batch_idx in self.viz_batch_idxs:
            # Set a flag to indicate that the mode prediction output will be visualized
            batch["visualize_prediction"] = True

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called when the train batch ends."""
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)