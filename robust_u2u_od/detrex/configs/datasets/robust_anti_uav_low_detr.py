from detectron2.data import MetadataCatalog

from .robust_anti_uav_detr import (  # noqa: F401
    DATASET_NAME,
    dataloader,
    robust_dataloader,
)

MetadataCatalog.get(DATASET_NAME).tags += [
    "Low-Poly",
]
