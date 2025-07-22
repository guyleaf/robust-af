import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from ...utils import count_coco_images

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


def register_dut_anti_uav(root: str, name: str):
    images_root = os.path.join(root, IMAGES_ROOT)
    train_ann_file = os.path.join(root, TRAIN_ANN_FILE)
    val_ann_file = os.path.join(root, VAL_ANN_FILE)

    num_train_images = count_coco_images(train_ann_file)
    num_val_images = count_coco_images(val_ann_file)

    register_coco_instances(
        f"{name}_train", dict(num_images=num_train_images), train_ann_file, images_root
    )
    register_coco_instances(
        f"{name}_val", dict(num_images=num_val_images), val_ann_file, images_root
    )
    # one metadata for all subsets
    MetadataCatalog.get(name).set(**METADATA)


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_dut_anti_uav(_root, DATASET_NAME)
