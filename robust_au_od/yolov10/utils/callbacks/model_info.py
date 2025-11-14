from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import de_parallel

MSG_PREFIX = colorstr("model_info_callbacks: ")


def on_pretrain_routine_end(trainer: DetectionTrainer):
    """Initiate and start project if module is present."""
    LOGGER.info(f"{MSG_PREFIX}Final state")
    de_parallel(trainer.model).info(detailed=True)
    LOGGER.info("")


callbacks = {"on_pretrain_routine_end": on_pretrain_routine_end}
