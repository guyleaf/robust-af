from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog

# from robust_af.detrex.data.datasets.register_uav_eagle import (
#     DATASET_NAME as TEST_DATASET_NAME,
# )
# from robust_af.detrex.data.datasets.register_dds import (
#     DATASET_NAME as TEST_DATASET_NAME,
# )
from robust_af.detrex.configs import get_config
from robust_af.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME
from robust_af.detrex.data.datasets.register_dut_anti_uav import (
    DATASET_NAME as TEST_DATASET_NAME,
)
from robust_af.detrex.modeling.processors import MultiScaleProcessor
from robust_af.models.feature_adapters import SpatialAFR

from ...models.robust_dino_r50_v2 import model

degraded = True
test_dataset_name = TEST_DATASET_NAME

dataset = get_config(f"datasets/{test_dataset_name}_detr.py")
if degraded:
    dataloader = dataset.robust_dataloader
else:
    dataloader = dataset.dataloader

train = get_config("train.py").train

metadata = MetadataCatalog.get(test_dataset_name)

suffix = "_degraded" if degraded else ""
output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}/{test_dataset_name}/robust_dino_r50_v2_4scale_36ep_1e-5_lr_spatial_afr_relu_no_cst_loss_no_train_heads_from_24ep{suffix}"

# ==============================================================

# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SimpleNN)(embed_dims=512),
#     res4=L(SimpleNN)(embed_dims=1024),
#     res5=L(SimpleNN)(embed_dims=2048),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(AMFGv2)(embed_dims=512),
#     res4=L(AMFGv2)(embed_dims=1024),
#     res5=L(AMFGv2)(embed_dims=2048),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SpatialAMFGv2)(embed_dims=512),
#     res4=L(SpatialAMFGv2)(embed_dims=1024),
#     res5=L(SpatialAMFGv2)(embed_dims=2048),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(AFR)(embed_dims=512, activation="ReLU"),
#     res4=L(AFR)(embed_dims=1024, activation="ReLU"),
#     res5=L(AFR)(embed_dims=2048, activation="ReLU"),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SpatialAFR)(embed_dims=512),
#     res4=L(SpatialAFR)(embed_dims=1024),
#     res5=L(SpatialAFR)(embed_dims=2048),
# )
model.robust_module = L(MultiScaleProcessor)(
    res3=L(SpatialAFR)(embed_dims=512, activation="ReLU"),
    res4=L(SpatialAFR)(embed_dims=1024, activation="ReLU"),
    res5=L(SpatialAFR)(embed_dims=2048, activation="ReLU"),
)
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SpatialAFR)(
#         embed_dims=512, spatial_cfg=dict(conv=False, activation=None), activation="ReLU"
#     ),
#     res4=L(SpatialAFR)(
#         embed_dims=1024,
#         spatial_cfg=dict(conv=False, activation=None),
#         activation="ReLU",
#     ),
#     res5=L(SpatialAFR)(
#         embed_dims=2048,
#         spatial_cfg=dict(conv=False, activation=None),
#         activation="ReLU",
#     ),
# )

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
