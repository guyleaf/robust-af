from ultralytics.models.yolov10.train import (
    YOLOv10DetectionTrainer as ORIGINAL_YOLOv10DetectionTrainer,
)
from ultralytics.utils.torch_utils import de_parallel

from ..data import build_yolo_dataset
from ..utils import DEFAULT_CFG


class YOLOv10DetectionTrainer(ORIGINAL_YOLOv10DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )
