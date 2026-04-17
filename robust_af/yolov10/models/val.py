import json

from ultralytics.models.yolov10 import (
    YOLOv10DetectionValidator as ORIGINAL_YOLOv10DetectionValidator,
)
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import init_seeds

from ..data import build_yolo_dataset
from ..utils import DEFAULT_CFG, DEFAULT_ROBUST_CFG


class YOLOv10DetectionValidator(ORIGINAL_YOLOv10DetectionValidator):
    def __init__(self, *args, cfg=DEFAULT_CFG, **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)

    def __call__(self, trainer=None, model=None):
        if trainer is None:
            # avoid any randomness during inference (e.g., GDIP)
            init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        stats = super().__call__(trainer, model)
        if not self.training:
            # save eval results in json for easier checking
            with open(self.save_dir / "results.json", "w") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(stats, f)
        return stats

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
        # handle calling from trainer
        if self.training:
            batch["clear"] = super().preprocess(batch["clear"])
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

        clear_batch = batch.get("clear", None)
        if clear_batch is not None:
            plot_images(
                clear_batch["img"],
                clear_batch["batch_idx"],
                clear_batch["cls"].squeeze(-1),
                clear_batch["bboxes"],
                paths=clear_batch["im_file"],
                fname=self.save_dir / f"val_batch{ni}_labels_clear.jpg",
                names=self.names,
                on_plot=self.on_plot,
            )
