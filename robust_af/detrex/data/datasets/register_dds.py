import os

from detectron2.data import MetadataCatalog

from .coco import register_coco_instances

DATASET_FOLDER = "DDS"
DATASET_NAME = DATASET_FOLDER.lower()
IMAGES_ROOT = "images"
ANN_FILES = dict(
    test=os.path.join("annotations", "test.json"),
    test_degraded=os.path.join("annotations", "degraded_test.json"),
    test_degraded_fog=os.path.join("annotations", "degraded_fog_test.json"),
)

METADATA = dict(
    num_classes=1,
    tags=[
        "DDS",
    ],
)


def register_uav_eagle(root: str, name: str):
    images_root = os.path.join(root, IMAGES_ROOT)

    for subset, ann_file in ANN_FILES.items():
        ann_file = os.path.join(root, ann_file)
        register_coco_instances(f"{name}_{subset}", {}, ann_file, images_root)

    # one metadata for all subsets
    MetadataCatalog.get(name).set(**METADATA)


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
_root = os.path.join(_root, DATASET_FOLDER)
register_uav_eagle(_root, DATASET_NAME)
