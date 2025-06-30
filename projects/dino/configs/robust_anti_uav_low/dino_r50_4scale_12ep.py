import os

from detectron2.data import MetadataCatalog
from detrex.config import get_config as get_upstream_config

from robust_u2u_od.detrex.configs import get_config
from robust_u2u_od.detrex.data.datasets.register_robust_anti_uav import DATASET_NAME

from ..models.dino_r50 import model

metadata = MetadataCatalog.get(DATASET_NAME)
train_metadata = MetadataCatalog.get(f"{DATASET_NAME}_train")

# get default config
dataloader = get_config("datasets/robust_anti_uav_low_detr.py").dataloader
optimizer = get_upstream_config("common/optim.py").AdamW
lr_multiplier = get_config(
    "schedules/robust_anti_uav_schedule.py"
).detr_schedulers.lr_multiplier_12ep_8bs
train = get_upstream_config("common/train.py").train

# ==============================================================

# TODO: auto-scale lr by batch size
# each gpu is 16/8 = 2
base_batch_size = 16
base_lr = 1e-4

# by default, use 4 gpus.
# each gpu is 8/4 = 2
total_batch_size = 8
lr = base_lr * (total_batch_size / base_batch_size)

num_epochs = 12
output_dir = os.path.expanduser(
    f"~/work/work_dirs/detrex/dino_r50_4scale/{DATASET_NAME}_s_low/dino_r50_4scale_12ep"
)

# wandb settings
tags = [*metadata.tags, "Test"]
notes = ""

# ==============================================================

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = output_dir

# max training iterations
num_images: int = train_metadata.num_images
# because drop_last=True
num_batches = num_images // total_batch_size
train.max_iter = num_epochs * num_batches
train.eval_period = num_batches
train.log_period = 20
train.checkpointer.period = num_batches

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# change the number of classes
model.num_classes = train_metadata.thing_classes

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = lr
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1 if "backbone" in module_name else 1
)

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = total_batch_size

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# wandb settings
train.wandb = dict(
    enabled=True,
    params=dict(
        dir=os.path.join(output_dir, "wandb"),
        project="robust-u2u-od",
        group="dino_r50_4scale_12ep",
        job_type="from scratch",
        tags=tags,
        notes=notes,
    ),
)
