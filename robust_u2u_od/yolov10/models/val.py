from ultralytics.models.yolov10 import (
    YOLOv10DetectionValidator as ORIGINAL_YOLOv10DetectionValidator,
)
from ultralytics.utils.plotting import plot_images

from ..data import build_yolo_dataset
from ..utils import DEFAULT_CFG, DEFAULT_ROBUST_CFG


class YOLOv10DetectionValidator(ORIGINAL_YOLOv10DetectionValidator):
    def __init__(self, *args, cfg=DEFAULT_CFG, **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, stride=self.stride
        )


class RobustYOLOv10DetectionValidator(YOLOv10DetectionValidator):
    def __init__(self, *args, cfg=DEFAULT_ROBUST_CFG, **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)

    def preprocess(self, batch: dict):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch = super().preprocess(batch)
        batch["clear_img"] = batch["clear_img"].to(self.device, non_blocking=True)
        batch["clear_img"] = (
            batch["clear_img"].half() if self.args.half else batch["clear_img"].float()
        ) / 255
        return batch

    def plot_val_samples(self, batch: dict, ni: int):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
        plot_images(
            batch["clear_img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels_clear.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
