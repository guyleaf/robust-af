import os

from robust_au_od.detrex.configs import get_config

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

# modify training config
train.output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}/robust_dino_r50_v2_4scale_50ep_1e-5_lr_spatial_afr_no_train_heads_from_24ep"

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
