from ultralytics.models.yolov10 import (
    YOLOv10DetectionPredictor as ORIGINAL_YOLOv10DetectionPredictor,
)
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import init_seeds

from ..utils import DEFAULT_CFG, DEFAULT_ROBUST_CFG


class YOLOv10DetectionPredictor(ORIGINAL_YOLOv10DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        # avoid any randomness during inference (e.g., GDIP)
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        return super().__call__(source, model, stream, *args, **kwargs)

    def predict_cli(self, source=None, model=None):
        # avoid any randomness during inference (e.g., GDIP)
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        return super().predict_cli(source, model)


class RobustYOLOv10DetectionPredictor(YOLOv10DetectionPredictor):
    def __init__(self, cfg=DEFAULT_ROBUST_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
