import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

DATASET_FOLDER = "DUT_Anti_UAV"
DATASET_NAME = DATASET_FOLDER.lower()
IMAGES_ROOT = "images"
TRAIN_ANN_FILE = os.path.join("annotations", "train.json")
VAL_ANN_FILE = os.path.join("annotations", "val.json")
TEST_ANN_FILE = os.path.join("annotations", "test.json")

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
    test_ann_file = os.path.join(root, TEST_ANN_FILE)

    register_coco_instances(f"{name}_train", {}, train_ann_file, images_root)
    register_coco_instances(f"{name}_val", {}, val_ann_file, images_root)
    register_coco_instances(f"{name}_test", {}, test_ann_file, images_root)
    # one metadata for all subsets
    MetadataCatalog.get(name).set(**METADATA)


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
_root = os.path.join(_root, DATASET_FOLDER, "detection")
register_dut_anti_uav(_root, DATASET_NAME)
