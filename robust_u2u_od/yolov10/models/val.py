from ultralytics.models.yolov10 import (
    YOLOv10DetectionValidator as ORIGINAL_YOLOv10DetectionValidator,
)

from ..data import build_yolo_dataset
from ..utils import DEFAULT_CFG


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
