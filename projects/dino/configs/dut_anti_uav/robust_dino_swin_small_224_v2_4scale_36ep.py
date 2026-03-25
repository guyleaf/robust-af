import os

from detectron2.config import LazyCall as L

from robust_af.detrex.configs import get_config
from robust_af.detrex.modeling import MultiScaleProcessor
from robust_af.models.feature_adapters import AFR

from .robust_dino_swin_small_224_v2_4scale_12ep import (  # noqa: F401
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
).detr_schedulers.lr_multiplier_36ep_warmup_8bs

lr = 1e-5
num_epochs = 36

# ==============================================================

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
model.robust_module = L(MultiScaleProcessor)(
    p1=L(AFR)(embed_dims=192, activation="ReLU"),
    p2=L(AFR)(embed_dims=384, activation="ReLU"),
    p3=L(AFR)(embed_dims=768, activation="ReLU"),
)
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(SpatialAFR)(embed_dims=192, activation="ReLU"),
#     p2=L(SpatialAFR)(embed_dims=384, activation="ReLU"),
#     p3=L(SpatialAFR)(embed_dims=768, activation="ReLU"),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(SpatialAFR)(embed_dims=192, spatial_cfg=dict(conv=False, activation=None)),
#     p2=L(SpatialAFR)(embed_dims=384, spatial_cfg=dict(conv=False, activation=None)),
#     p3=L(SpatialAFR)(embed_dims=768, spatial_cfg=dict(conv=False, activation=None)),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(SpatialAFR)(embed_dims=192, activation="ReLU", spatial_cfg=dict(conv=False, activation=None)),
#     p2=L(SpatialAFR)(embed_dims=384, activation="ReLU", spatial_cfg=dict(conv=False, activation=None)),
#     p3=L(SpatialAFR)(embed_dims=768, activation="ReLU", spatial_cfg=dict(conv=False, activation=None)),
# )

# no cst loss
# model.criterion.loss_cst = None

# modify training config
train.output_dir = f"./outputs/dino_swin_small_224_4scale/{DATASET_NAME}/robust_dino_swin_small_224_v2_4scale_36ep_1e-5_lr_afr_relu_no_train_heads_from_24ep"

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
    group="robust_dino_swin_small_224_v2_4scale_36ep",
)
train.wandb["params"].update(params)

# set the random seed
# [42, 123, 456, 789, 2025]
train.seed = 2025
