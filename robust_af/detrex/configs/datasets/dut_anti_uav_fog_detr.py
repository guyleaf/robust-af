import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from omegaconf import DictConfig

from robust_af.detrex.configs import get_config
from robust_af.detrex.data.build import get_detection_dataset_dicts
from robust_af.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME
from robust_af.detrex.data.transforms import Degradation

_base = get_config(f"datasets/{DATASET_NAME}_detr.py")

# normal version of dataset

dataloader: DictConfig = _base.dataloader

# robust version of dataset

robust_dataloader = _base.robust_dataloader
robust_dataloader.train.mapper.robust_augmentations = [
    L(Degradation)(degradations=["fog", "identity"])
]

# offline augmentation (degraded-only)
robust_dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_val_degraded_fog", filter_empty=False
)

# online augmentation (degraded-only)
robust_dataloader.train_test.mapper.augmentations = [
    L(Degradation)(name="fog"),
    L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
]
