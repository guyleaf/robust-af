import os

from robust_au_od.detrex.configs import get_config

from .dino_swin_small_224_4scale_12ep import (  # noqa: F401
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
).detr_schedulers["lr_multiplier_36ep_warmup_8bs"]

# ==============================================================

lr = 5e-5
num_epochs = 36
output_dir = f"./outputs/dino_swin_small_224_4scale/{DATASET_NAME}/dino_swin_small_224_4scale_36ep_5e-5_lr_new_mapper_warmup_3_again"

# ==============================================================

# modify model config
# use the original implementation of dab-detr position embedding in 36 epochs training.
# model.position_embedding.temperature = 20
# model.position_embedding.offset = 0.0

# modify training config
train.output_dir = output_dir

train.max_iter = num_epochs * num_batches

# modify optimizer config
optimizer.lr = lr
optimizer.weight_decay = 1e-4

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = output_dir

# wandb settings
params = dict(
    dir=output_dir,
    name=os.path.basename(output_dir),
    group="dino_swin_small_224_4scale_36ep",
)
train.wandb["params"].update(params)

# set the random seed
# [42, 123, 456, 789, 2025]
train.seed = 456
