from detectron2.config import LazyCall as L  # noqa: F401
from detectron2.data import MetadataCatalog

from robust_af.detrex.configs import get_config
from robust_af.detrex.data.datasets.register_dds import (
    DATASET_NAME as TEST_DATASET_NAME,
)
from robust_af.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME

# from robust_af.detrex.data.datasets.register_uav_eagle import (
#     DATASET_NAME as TEST_DATASET_NAME,
# )
# from robust_af.detrex.data.datasets.register_dut_anti_uav import (
#     DATASET_NAME as TEST_DATASET_NAME,
# )
from robust_af.models.image_adapters.baselines import DIP, GDIP, DENet  # noqa: F401

from ...models.robust_dino_r50_v2 import model

test_subset = True
degraded = True
fog = False
test_dataset_name = TEST_DATASET_NAME

dataset = get_config(f"datasets/{test_dataset_name}_detr.py")
if degraded:
    dataloader = dataset.robust_dataloader
else:
    dataloader = dataset.dataloader

name: str = dataloader.test.dataset.names
if degraded:
    assert name.endswith("degraded")
    if fog:
        name = name.replace("degraded", "degraded_fog")
if test_subset:
    name = name.replace("_val", "_test")
dataloader.test.dataset.names = name

train = get_config("train.py").train

metadata = MetadataCatalog.get(test_dataset_name)

suffix = "_test" if test_subset else ""
suffix += "_degraded" if degraded else ""
suffix += "_fog" if fog else ""
output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}/{test_dataset_name}/robust_dino_r50_v2_4scale_50ep_5e-5_lr_gdip_fine_tuning_from_24ep{suffix}"

# ==============================================================

model.robust_module = None
model.robust_image_module = None
# model.robust_image_module = L(DENet)(compat_mode=False)
model.robust_image_module = L(GDIP)(multi_level=False)
# model.robust_image_module = L(DIP)()

# modify model config
# use the original implementation of dab-detr position embedding.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# change the number of classes
model.num_classes = metadata.num_classes

# set output dir
train.output_dir = output_dir

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.test.num_workers = 4

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# set the random seed
train.seed = 2025
