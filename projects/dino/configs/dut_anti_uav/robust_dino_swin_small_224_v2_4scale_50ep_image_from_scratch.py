import os

from detectron2.config import LazyCall as L

from robust_af.detrex.configs import get_config
from robust_af.models.image_adapters.baselines import DENet

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
).detr_schedulers.lr_multiplier_50ep_warmup_8bs

lr = 1e-5
num_epochs = 50

# ==============================================================

# no freeze
model.train_all = True

model.robust_module = None
model.robust_image_module = L(DENet)(compat_mode=False)

model.criterion.loss_cst = None
model.criterion.loss_image_cst = None
# model.criterion.loss_image_cst = L(nn.MSELoss)(reduction="none")
model.criterion.weight_dict = {"loss_image_cst": 20.0}

# modify training config
train.init_checkpoint = "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth?matching_heuristics=True"
train.output_dir = f"./outputs/dino_swin_small_224_4scale/{DATASET_NAME}/robust_dino_swin_small_224_v2_4scale_50ep_1e-5_lr_denet_warmup_from_scratch"

# max training iterations
train.max_iter = num_epochs * num_batches

# modify optimizer config
optimizer.lr = lr
optimizer.params.lr_factor_func = lambda module_name: (
    0.1 if "backbone" in module_name else 1
)

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# wandb settings
params = dict(
    dir=train.output_dir,
    name=os.path.basename(train.output_dir),
    group="robust_dino_swin_small_224_v2_4scale_50ep",
    job_type="from scratch",
)
train.wandb["params"].update(params)

# set the random seed
# [42, 123, 456, 789, 2025]
train.seed = 2025
