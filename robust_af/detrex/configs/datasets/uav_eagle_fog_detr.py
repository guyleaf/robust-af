from detectron2.config import LazyCall as L
from omegaconf import DictConfig

from robust_af.detrex.configs import get_config
from robust_af.detrex.data.build import get_detection_dataset_dicts
from robust_af.detrex.data.datasets.register_uav_eagle import DATASET_NAME

_base = get_config(f"datasets/{DATASET_NAME}_detr.py")

# normal version of dataset

dataloader: DictConfig = _base.dataloader

# robust version of dataset

robust_dataloader = _base.robust_dataloader

# offline augmentation (degraded-only)
robust_dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_test_degraded_fog", filter_empty=False
)
