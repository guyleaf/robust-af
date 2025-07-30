import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

DATASET_NAME = "robust_anti_uav"
IMAGES_ROOT = "harmonized"
TRAIN_ANN_FILE = os.path.join("annotations", "train.json")
VAL_ANN_FILE = os.path.join("annotations", "val.json")

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


def register_robust_anti_uav(root: str, name: str):
    images_root = os.path.join(root, IMAGES_ROOT)
    train_ann_file = os.path.join(root, TRAIN_ANN_FILE)
    val_ann_file = os.path.join(root, VAL_ANN_FILE)

    register_coco_instances(f"{name}_train", {}, train_ann_file, images_root)
    register_coco_instances(f"{name}_val", {}, val_ann_file, images_root)
    # one metadata for all subsets
    MetadataCatalog.get(name).set(**METADATA)


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_robust_anti_uav(_root, DATASET_NAME)
