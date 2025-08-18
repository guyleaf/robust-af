from ultralytics.models import YOLOv10 as ORIGINAL_YOLOv10
from ultralytics.nn.tasks import YOLOv10DetectionModel

from ..nn.tasks import RobustYOLOv10DetectionModel
from .predict import RobustYOLOv10DetectionPredictor, YOLOv10DetectionPredictor
from .train import RobustYOLOv10DetectionTrainer, YOLOv10DetectionTrainer
from .val import RobustYOLOv10DetectionValidator, YOLOv10DetectionValidator


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


class RobustYOLOv10(YOLOv10):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": RobustYOLOv10DetectionModel,
                "trainer": RobustYOLOv10DetectionTrainer,
                "validator": RobustYOLOv10DetectionValidator,
                "predictor": RobustYOLOv10DetectionPredictor,
            },
        }
