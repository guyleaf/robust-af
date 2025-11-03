# ruff: noqa: F401
from detectron2.layers import ShapeSpec

from robust_au_od.detrex.configs import get_config

from .dino_r50_4scale_12ep import (
    DATASET_NAME,
    dataloader,
    metadata,
    model,
    num_epochs,
    num_images,
    optimizer,
    train,
)
from .dino_r50_4scale_12ep import batch_size as base_batch_size
from .dino_r50_4scale_12ep import lr as base_lr

lr_multiplier = get_config(
    f"schedules/{DATASET_NAME}_schedule.py"
).detr_schedulers.lr_multiplier_12ep_4bs

# ==============================================================

# by default, use 4 gpus.
# each gpu is 4/4 = 1
batch_size = 4
lr = base_lr * (batch_size / base_batch_size)

output_dir = f"./outputs/dino_r50_5scale/{DATASET_NAME}/dino_r50_5scale_12ep_5e-5_lr"

# wandb settings
tags = [*metadata.tags]
notes = ""

# ==============================================================

# modify model config to generate 4 scale backbone features
# and 5 scale input features
model.backbone.out_features = ["res2", "res3", "res4", "res5"]

model.neck.input_shapes = {
    "res2": ShapeSpec(channels=256),
    "res3": ShapeSpec(channels=512),
    "res4": ShapeSpec(channels=1024),
    "res5": ShapeSpec(channels=2048),
}
model.neck.in_features = ["res2", "res3", "res4", "res5"]
model.neck.num_outs = 5
model.transformer.num_feature_levels = 5

# modify output_dir
train.output_dir = output_dir

# max training iterations
# because drop_last=True
num_batches = num_images // batch_size
train.max_iter = num_epochs * num_batches
# eval per epoch
train.eval_period = num_batches
train.checkpointer.period = num_batches

# modify optimizer config
optimizer.lr = lr

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.batch_size = batch_size

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = output_dir

# wandb settings
params = dict(dir=output_dir, group="dino_r50_5scale_12ep", tags=tags, notes=notes)
train.wandb["params"].update(params)
