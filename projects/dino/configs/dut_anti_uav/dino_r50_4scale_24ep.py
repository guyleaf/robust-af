import os

from robust_au_od.detrex.configs import get_config

from .dino_r50_4scale_12ep import (  # noqa: F401
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
).detr_schedulers.lr_multiplier_24ep_8bs

num_epochs = 24

# ==============================================================

# modify model config
# use the original implementation of dab-detr position embedding in 24 epochs training.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# modify training config
train.output_dir = (
    f"./outputs/dino_r50_4scale/{DATASET_NAME}/dino_r50_4scale_24ep_1e-4_lr_4"
)

# max training iterations
train.max_iter = num_epochs * num_batches

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# wandb settings
params = dict(
    dir=train.output_dir,
    name=os.path.basename(train.output_dir),
    group="dino_r50_4scale_24ep",
)
train.wandb["params"].update(params)
