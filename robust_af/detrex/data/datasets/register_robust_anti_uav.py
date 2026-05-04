import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

DATASET_FOLDER = "Robust_Anti_UAV"
DATASET_NAME = DATASET_FOLDER.lower()
IMAGES_ROOT = "images"
ANN_FILES = dict(
    train=os.path.join("annotations", "train.json"),
    val=os.path.join("annotations", "val.json"),
    val_degraded=os.path.join("annotations", "degraded_val.json"),
    val_degraded_fog=os.path.join("annotations", "degraded_fog_val.json"),
)

METADATA = dict(
    num_classes=1,
    tags=[
        "Robust Anti-UAV",
        "Harmonized",
        "Image2Weather",
        "Clear",
        "Cloudy",
    ],
)


def register_robust_anti_uav(root: str, name: str, **kwargs):
    images_root = os.path.join(root, IMAGES_ROOT)

    for subset, ann_file in ANN_FILES.items():
        ann_file = os.path.join(root, ann_file)
        register_coco_instances(f"{name}_{subset}", {}, ann_file, images_root)

    # one metadata for all subsets
    MetadataCatalog.get(name).set(**{**METADATA, **kwargs})


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
_root = os.path.join(_root, DATASET_FOLDER)
register_robust_anti_uav(_root, DATASET_NAME)
