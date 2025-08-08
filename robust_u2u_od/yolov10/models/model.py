from ultralytics.models import YOLOv10 as ORIGINAL_YOLOv10
from ultralytics.models.yolov10 import (
    YOLOv10DetectionPredictor,
    YOLOv10DetectionValidator,
)
from ultralytics.nn.tasks import YOLOv10DetectionModel

from .train import YOLOv10DetectionTrainer


class YOLOv10(ORIGINAL_YOLOv10):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }
