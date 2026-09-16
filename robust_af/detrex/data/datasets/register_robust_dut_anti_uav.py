import os

from detectron2.data import MetadataCatalog

from .coco import register_coco_instances

DATASET_FOLDER = "Robust_DUT_Anti_UAV"
DATASET_NAME = DATASET_FOLDER.lower()
IMAGES_ROOT = "images"
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
        "DUT Anti-UAV",
    ],
)


def register_robust_dut_anti_uav(root: str, name: str, **kwargs):
    images_root = os.path.join(root, IMAGES_ROOT)
    train_ann_file = os.path.join(root, TRAIN_ANN_FILE)
    val_ann_file = os.path.join(root, VAL_ANN_FILE)

    register_coco_instances(f"{name}_train", {}, train_ann_file, images_root)
    register_coco_instances(f"{name}_val", {}, val_ann_file, images_root)
    # one metadata for all subsets
    MetadataCatalog.get(name).set(**{**METADATA, **kwargs})


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
_root = os.path.join(_root, DATASET_FOLDER)
register_robust_dut_anti_uav(_root, DATASET_NAME)
