from detectron2.data import MetadataCatalog

from robust_u2u_od.detrex.data.datasets.register_robust_anti_uav import DATASET_NAME

MetadataCatalog.get(DATASET_NAME).tags += [
    "Small",
    "Low-Poly",
]
