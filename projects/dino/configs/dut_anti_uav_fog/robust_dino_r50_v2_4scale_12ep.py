import os

from robust_af.detrex.configs import get_config

from ..dut_anti_uav.robust_dino_r50_v2_4scale_12ep import (  # noqa: F401
    DATASET_NAME,
    batch_size,
    lr_multiplier,
    model,
    num_batches,
    optimizer,
    train,
)

# get default config
dataloader = get_config(f"datasets/{DATASET_NAME}_fog_detr.py").robust_dataloader

lr = 1e-5

# ==============================================================

# modify training config
train.output_dir = train.output_dir.replace(
    f"/{DATASET_NAME}/", f"/{DATASET_NAME}_fog/"
)

# modify optimizer config
optimizer.lr = lr

# modify dataloader config
dataloader.train.num_workers = 4

# please notice that this is total batch size.
# suppose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = batch_size

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# wandb settings
params = dict(dir=train.output_dir, name=os.path.basename(train.output_dir))
train.wandb["params"].update(params)

# set the random seed
# [42, 123, 456, 789, 2025]
train.seed = 2025
