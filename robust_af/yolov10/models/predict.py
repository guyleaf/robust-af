from ultralytics.models.yolov10 import (
    YOLOv10DetectionPredictor as ORIGINAL_YOLOv10DetectionPredictor,
)

from ..utils import DEFAULT_CFG, DEFAULT_ROBUST_CFG


class YOLOv10DetectionPredictor(ORIGINAL_YOLOv10DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)


class RobustYOLOv10DetectionPredictor(YOLOv10DetectionPredictor):
    def __init__(self, cfg=DEFAULT_ROBUST_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
