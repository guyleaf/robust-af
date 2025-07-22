import json
import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.file_io import PathManager

DATASET_NAME = "dut_anti_uav"
IMAGES_ROOT = "images"
TRAIN_ANN_FILE = os.path.join("annotations", "train.json")
VAL_ANN_FILE = os.path.join("annotations", "val.json")

METADATA = dict(
    num_classes=1,
    tags=[
        "DUT Anti-UAV",
    ],
)


def _get_number_images(json_file: str):
    json_file = PathManager.get_local_path(json_file)
    with open(json_file, "r") as f:
        content = json.load(f)
        return len(content["images"])


def register_robust_anti_uav(root: str, name: str):
    images_root = os.path.join(root, IMAGES_ROOT)
    train_ann_file = os.path.join(root, TRAIN_ANN_FILE)
    val_ann_file = os.path.join(root, VAL_ANN_FILE)

    num_train_images = _get_number_images(train_ann_file)
    num_val_images = _get_number_images(val_ann_file)

    register_coco_instances(
        f"{name}_train", dict(num_images=num_train_images), train_ann_file, images_root
    )
    register_coco_instances(
        f"{name}_val", dict(num_images=num_val_images), val_ann_file, images_root
    )
    # one metadata for all subsets
    MetadataCatalog.get(name).set(**METADATA)


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_robust_anti_uav(_root, DATASET_NAME)
