from detectron2.config import LazyCall as L
from detectron2.data import (
    MetadataCatalog,
    get_detection_dataset_dicts,
)
from detrex.config import get_config
from omegaconf import DictConfig

from robust_u2u_od.detrex.data.datasets.register_robust_anti_uav import DATASET_NAME

dataset_name = DATASET_NAME

# always called by get_config
# so, we assume the dataset is registered.
dataset_metadata = MetadataCatalog.get(dataset_name)

dataloader: DictConfig = get_config("common/data/coco_detr.py").dataloader
dataloader.train.dataset = L(get_detection_dataset_dicts)(names=f"{dataset_name}_train")
dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{dataset_name}_val", filter_empty=False
)
