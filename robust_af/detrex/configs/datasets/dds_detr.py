from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from omegaconf import DictConfig

from robust_af.detrex.configs import get_config
from robust_af.detrex.data.datasets.register_dds import DATASET_NAME

_base = get_config("datasets/coco_detr.py")

# normal version of dataset

dataloader: DictConfig = _base.dataloader
dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_test", filter_empty=False
)

# robust version of dataset

robust_dataloader = _base.robust_dataloader
# offline augmentation
robust_dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_test_degraded", filter_empty=False
)
