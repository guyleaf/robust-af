from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detrex.config import get_config
from omegaconf import DictConfig

from robust_u2u_od.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME

dataloader: DictConfig = get_config("common/data/coco_detr.py").dataloader
dataloader.train.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_train", filter_empty=False
)
dataloader.train.persistent_workers = True
dataloader.train.pin_memory = True

dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_val", filter_empty=False
)
