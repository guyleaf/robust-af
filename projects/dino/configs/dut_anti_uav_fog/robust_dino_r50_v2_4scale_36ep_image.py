import os

from detectron2.config import LazyCall as L

from robust_af.detrex.configs import get_config
from robust_af.models.image_adapters.baselines import DIP, GDIP, DENet  # noqa: F401

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
).detr_schedulers.lr_multiplier_36ep_warmup_8bs

lr = 1e-5
num_epochs = 36

# ==============================================================

model.robust_module = None
# model.robust_image_module = L(DENet)(compat_mode=False)
# model.robust_image_module = L(GDIP)(multi_level=True)
model.robust_image_module = L(DIP)()

model.criterion.loss_cst = None
model.criterion.loss_image_cst = None
# model.criterion.loss_image_cst = L(nn.MSELoss)(reduction="none")
model.criterion.weight_dict = {"loss_image_cst": 20.0}

# modify training config
train.output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}_fog/robust_dino_r50_v2_4scale_36ep_1e-5_lr_dip_no_train_heads_from_24ep"

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
    group="robust_dino_r50_v2_4scale_36ep",
)
train.wandb["params"].update(params)

# set the random seed
# [42, 123, 456, 789, 2025]
train.seed = 2025

# train.clip_grad.enabled = False
# train.clip_grad.params.max_norm = 1.0
