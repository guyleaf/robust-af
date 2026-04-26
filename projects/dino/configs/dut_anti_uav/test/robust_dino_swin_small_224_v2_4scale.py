from detectron2.config import LazyCall as L  # noqa: F401
from detectron2.data import MetadataCatalog

from robust_af.detrex.configs import get_config

# from robust_af.detrex.data.datasets.register_dds import (
#     DATASET_NAME as TEST_DATASET_NAME,
# )
from robust_af.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME

# from robust_af.detrex.data.datasets.register_uav_eagle import (
#     DATASET_NAME as TEST_DATASET_NAME,
# )
from robust_af.detrex.data.datasets.register_dut_anti_uav import (
    DATASET_NAME as TEST_DATASET_NAME,
)
from robust_af.detrex.modeling.processors import MultiScaleProcessor  # noqa: F401
from robust_af.models.feature_adapters import SimpleNN, SpatialAFR  # noqa: F401

from ...models.robust_dino_swin_small_224_v2 import model

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
output_dir = f"./outputs/dino_swin_small_224_4scale/{DATASET_NAME}/{test_dataset_name}/robust_dino_swin_small_224_v2_4scale_36ep_1e-5_lr_spatial_afr_relu_no_train_heads_from_24ep{suffix}"

# ==============================================================

# model.robust_module = None

# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(SimpleNN)(embed_dims=192),
#     p2=L(SimpleNN)(embed_dims=384),
#     p3=L(SimpleNN)(embed_dims=768),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(AMFGv2)(embed_dims=192),
#     p2=L(AMFGv2)(embed_dims=384),
#     p3=L(AMFGv2)(embed_dims=768),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(SpatialAMFGv2)(embed_dims=192),
#     p2=L(SpatialAMFGv2)(embed_dims=384),
#     p3=L(SpatialAMFGv2)(embed_dims=768),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(AFR)(embed_dims=192, activation="ReLU"),
#     p2=L(AFR)(embed_dims=384, activation="ReLU"),
#     p3=L(AFR)(embed_dims=768, activation="ReLU"),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(SpatialAFR)(embed_dims=192),
#     p2=L(SpatialAFR)(embed_dims=384),
#     p3=L(SpatialAFR)(embed_dims=768),
# )
model.robust_module = L(MultiScaleProcessor)(
    p1=L(SpatialAFR)(embed_dims=192, activation="ReLU"),
    p2=L(SpatialAFR)(embed_dims=384, activation="ReLU"),
    p3=L(SpatialAFR)(embed_dims=768, activation="ReLU"),
)
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(SpatialAFR)(embed_dims=192, spatial_cfg=dict(conv=False, activation=None), activation="ReLU"),
#     p2=L(SpatialAFR)(embed_dims=384, spatial_cfg=dict(conv=False, activation=None), activation="ReLU"),
#     p3=L(SpatialAFR)(embed_dims=768, spatial_cfg=dict(conv=False, activation=None), activation="ReLU"),
# )

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
