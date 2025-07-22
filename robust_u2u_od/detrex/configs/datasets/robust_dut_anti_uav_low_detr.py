from detectron2.data import MetadataCatalog

from .robust_dut_anti_uav_detr import DATASET_NAME, dataloader  # noqa: F401

MetadataCatalog.get(DATASET_NAME).tags += [
    "Low-Poly",
]
