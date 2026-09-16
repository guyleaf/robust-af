from copy import deepcopy

from detectron2.config import LazyCall as L

from robust_af.detrex.data.build import get_detection_dataset_dicts
from robust_af.detrex.data.datasets.register_robust_dut_anti_uav_low import (
    DATASET_NAME,
)

from .robust_dut_anti_uav_detr import dataloader, robust_dataloader  # noqa: F401

# normal version of dataset

dataloader = deepcopy(dataloader)
dataloader.train.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_train", filter_empty=False
)

dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_val", filter_empty=False
)

# robust version of dataset

robust_dataloader = deepcopy(robust_dataloader)
robust_dataloader.train.dataset = deepcopy(dataloader.train.dataset)
robust_dataloader.test.dataset = deepcopy(dataloader.test.dataset)
