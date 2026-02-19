import os

from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog
from detrex.config import get_config as get_upstream_config

from robust_af.detrex.configs import get_config
from robust_af.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME
from robust_af.detrex.modeling import MultiScaleProcessor
from robust_af.detrex.utils import count_coco_images
from robust_af.models.robust_layers.afr import SpatialAFR

from ...models.robust_dino_r50_v2 import model

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
# base_batch_size = dataloader.train.total_batch_size
# base_lr = optimizer.lr

# by default, use 4 gpus.
# each gpu is 8/4 = 2
batch_size = 8
# lr = base_lr * (batch_size / base_batch_size)
lr = 1e-5

num_epochs = 12
eval_per_epochs = 1
output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}/robust_dino_r50_v2_4scale_12ep_1e-5_lr_spatial_afr_selector_no_train_heads_from_36ep"

# wandb settings
tags = [*metadata.tags]
notes = ""

# ==============================================================

# model.transformer.encoder.robust_layer.spatial_attention = 4
# model.vis_period = 2000
# model.criterion.weight_dict = {k: 10.0 for k in model.criterion.weight_dict}
model.train_heads = False
model.robust_module = L(MultiScaleProcessor)(
    res3=L(SpatialAFR)(embed_dims=512, selector=True),
    res4=L(SpatialAFR)(embed_dims=1024, selector=True),
    res5=L(SpatialAFR)(embed_dims=2048, selector=True),
)

# modify model config
# use the original implementation of dab-detr position embedding.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# modify training config
train.init_checkpoint = "/home/leafying/work/work_dirs/detrex/dino_r50_4scale/dut_anti_uav/dino_r50_4scale_36ep_5e-5_lr/model_best_0021449.pth"
train.output_dir = output_dir

# train.sync_bn = True

# max training iterations
num_images = count_coco_images(train_metadata.json_file)
# because drop_last=True
num_batches = num_images // batch_size
train.max_iter = num_epochs * num_batches
train.eval_period = eval_per_epochs * num_batches
# NOTE: log_period should be divisble by num_batches in order to log eval metrics.
# Otherwise, some platform will ignore them, such as wandb because of requirement of monotonically increasing.
train.log_period = 10
train.checkpointer.period = num_batches
train.checkpointer.max_to_keep = 3

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
# optimizer.params.lr_factor_func = (
#     lambda module_name: 0.1 if "backbone" in module_name else 1
# )
# optimizer.params.lr_factor_func = (
#     lambda module_name: 0.1
#     if any(k in module_name for k in ("backbone", "class_embed", "bbox_embed"))
#     else 1
# )

# modify dataloader config
dataloader.train.num_workers = 4

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = batch_size

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = output_dir

# wandb settings
train.wandb = dict(
    enabled=True,
    params=dict(
        dir=output_dir,
        name=os.path.basename(output_dir),
        project="detrex",
        group="robust_dino_r50_v2_4scale_12ep",
        job_type="from scratch",
        tags=tags,
        notes=notes,
    ),
)

# set the random seed
train.seed = 2025

# evaluate train subset during validation (require `dataloader.train_test``) (heavy computation)
train.eval_train = True
