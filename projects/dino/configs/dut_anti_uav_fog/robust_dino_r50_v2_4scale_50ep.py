import os

from detectron2.config import LazyCall as L

from robust_af.detrex.configs import get_config
from robust_af.detrex.modeling import MultiScaleProcessor
from robust_af.models.feature_adapters import SimpleNN, SpatialAFR  # noqa: F401

from .robust_dino_r50_v2_4scale_12ep import (  # noqa: F401
    DATASET_NAME,
    dataloader,
    model,
    num_batches,
    optimizer,
    train,
)

# get default config
lr_multiplier = get_config(
    f"schedules/{DATASET_NAME}_schedule.py"
).detr_schedulers.lr_multiplier_50ep_warmup_8bs

lr = 1e-5
num_epochs = 50

# ==============================================================

model.robust_module = L(MultiScaleProcessor)(
    res3=L(SimpleNN)(embed_dims=512),
    res4=L(SimpleNN)(embed_dims=1024),
    res5=L(SimpleNN)(embed_dims=2048),
)
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(AMFGv2)(embed_dims=512),
#     res4=L(AMFGv2)(embed_dims=1024),
#     res5=L(AMFGv2)(embed_dims=2048),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(AFR)(embed_dims=512, activation="ReLU"),
#     res4=L(AFR)(embed_dims=1024, activation="ReLU"),
#     res5=L(AFR)(embed_dims=2048, activation="ReLU"),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SpatialAFR)(embed_dims=512, activation="ReLU"),
#     res4=L(SpatialAFR)(embed_dims=1024, activation="ReLU"),
#     res5=L(SpatialAFR)(embed_dims=2048, activation="ReLU"),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SpatialAFR)(embed_dims=512, spatial_cfg=dict(conv=False, activation=None), activation="ReLU"),
#     res4=L(SpatialAFR)(embed_dims=1024, spatial_cfg=dict(conv=False, activation=None), activation="ReLU"),
#     res5=L(SpatialAFR)(embed_dims=2048, spatial_cfg=dict(conv=False, activation=None), activation="ReLU"),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SpatialAFR)(
#         embed_dims=512,
#         activation="ReLU",
#         # use_bn=True,
#         # spatial_cfg=dict(spatial_attention=1),
#     ),
#     res4=L(SpatialAFR)(
#         embed_dims=1024,
#         activation="ReLU",
#         # use_bn=True,
#         # spatial_cfg=dict(spatial_attention=1),
#     ),
#     res5=L(SpatialAFR)(
#         embed_dims=2048,
#         activation="ReLU",
#         # use_bn=True,
#         # spatial_cfg=dict(spatial_attention=1),
#     ),
# )
# train.sync_bn = True

# no cst loss
# model.criterion.loss_cst = None

# modify training config
train.output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}_fog/robust_dino_r50_v2_4scale_50ep_1e-5_lr_simple_nn_no_train_heads_from_24ep"

# max training iterations
train.max_iter = num_epochs * num_batches

# modify optimizer config
optimizer.lr = lr

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# wandb settings
params = dict(
    dir=train.output_dir,
    name=os.path.basename(train.output_dir),
    group="robust_dino_r50_v2_4scale_50ep",
)
train.wandb["params"].update(params)

# set the random seed
# [42, 123, 456, 789, 2025]
train.seed = 2025
