from detectron2.data import MetadataCatalog
from detrex.config import get_config as get_upstream_config

from robust_au_od.detrex.configs import get_config
from robust_au_od.detrex.data.datasets.register_robust_dut_anti_uav_low import (
    DATASET_NAME,
)
from robust_au_od.detrex.utils import count_coco_images

from ..models.robust_dino_r50 import model

# get default config
dataloader = get_config(f"datasets/{DATASET_NAME}_detr.py").robust_dataloader
optimizer = get_upstream_config("common/optim.py").AdamW
lr_multiplier = get_config(
    f"schedules/{DATASET_NAME}_schedule.py"
).detr_schedulers.lr_multiplier_12ep_8bs
train = get_config("train.py").train

metadata = MetadataCatalog.get(DATASET_NAME)
train_metadata = MetadataCatalog.get(f"{DATASET_NAME}_train")

# ==============================================================

# TODO: auto-scale lr by batch size
# each gpu is 16/8 = 2
base_batch_size = dataloader.train.total_batch_size
base_lr = optimizer.lr

# by default, use 4 gpus.
# each gpu is 8/4 = 2
total_batch_size = 8
lr = base_lr * (total_batch_size / base_batch_size)

num_epochs = 12
output_dir = f"./outputs/robust_dino_r50_4scale/{DATASET_NAME}/robust_dino_r50_4scale_12ep_spatial_att_1_sync_bn_test"

# wandb settings
tags = ["1x1 Patch", *metadata.tags]
notes = ""

# ==============================================================

model.transformer.encoder.robust_layer.spatial_attention = 1
model.vis_period = 2000

# modify training config
train.init_checkpoint = "/home/leafying/work/work_dirs/detrex/dino_r50_4scale/robust_dut_anti_uav_low/dino_r50_4scale_12ep/model_final.pth"
train.output_dir = output_dir

train.sync_bn = True

# max training iterations
num_images = count_coco_images(train_metadata.json_file)
# because drop_last=True
num_batches = num_images // total_batch_size
train.max_iter = num_epochs * num_batches
# eval per epoch
train.eval_period = num_batches
train.log_period = 20
train.checkpointer.period = num_batches
train.checkpointer.max_to_keep = 3
train.best_checkpointer = dict(val_metric="bbox/APs", mode="max")

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# change the number of classes
model.num_classes = metadata.num_classes

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

# modify dataloader config
dataloader.train.num_workers = 4

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
        dir=output_dir,
        project="robust-au-od",
        group="robust_dino_r50_4scale_12ep",
        job_type="from scratch",
        tags=tags,
        notes=notes,
    ),
)

# set the random seed
train.seed = 2025
