import os

from .register_robust_anti_uav import DATASET_FOLDER, METADATA, register_robust_anti_uav

DATASET_FOLDER = f"{DATASET_FOLDER}_Low"
DATASET_NAME = DATASET_FOLDER.lower()

_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
_root = os.path.join(_root, DATASET_FOLDER)
register_robust_anti_uav(_root, DATASET_NAME, tags=METADATA["tags"] + ["Low-Poly"])
