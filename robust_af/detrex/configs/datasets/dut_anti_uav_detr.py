from detectron2.config import LazyCall as L
from omegaconf import DictConfig

from robust_af.detrex.configs import get_config
from robust_af.detrex.data.build import get_detection_dataset_dicts
from robust_af.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME

_base = get_config("datasets/coco_detr.py")

# normal version of dataset

dataloader: DictConfig = _base.dataloader
dataloader.train.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_train", filter_empty=False
)
dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_val", filter_empty=False
)
dataloader.train_test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_train", filter_empty=False
)

# robust version of dataset

robust_dataloader = _base.robust_dataloader
robust_dataloader.train.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_train", filter_empty=False
)
# offline augmentation
robust_dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_val_degraded", filter_empty=False
)
# online augmentation
robust_dataloader.train_test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_train", filter_empty=False
)
