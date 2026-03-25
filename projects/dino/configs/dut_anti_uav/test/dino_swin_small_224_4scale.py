from detectron2.data.catalog import MetadataCatalog

from robust_af.detrex.configs import get_config

# from robust_af.detrex.data.datasets.register_uav_eagle import (
#     DATASET_NAME as TEST_DATASET_NAME,
# )
# from robust_af.detrex.data.datasets.register_dds import (
#     DATASET_NAME as TEST_DATASET_NAME,
# )
from robust_af.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME
from robust_af.detrex.data.datasets.register_dut_anti_uav import (
    DATASET_NAME as TEST_DATASET_NAME,
)

from ...models.dino_swin_small_224 import model

degraded = False
test_dataset_name = TEST_DATASET_NAME

dataset = get_config(f"datasets/{test_dataset_name}_detr.py")
if degraded:
    dataloader = dataset.robust_dataloader
else:
    dataloader = dataset.dataloader

train = get_config("train.py").train
metadata = MetadataCatalog.get(test_dataset_name)

suffix = "_degraded" if degraded else ""
output_dir = f"./outputs/dino_swin_small_224_4scale/{DATASET_NAME}/{test_dataset_name}/dino_swin_small_224_4scale_24ep_5e-5_lr_new_mapper_warmup_again{suffix}"

# ==============================================================

# NOTE: for tsne vis
# model.backbone.out_indices = [0, 1, 2, 3]
# model.neck.input_shapes = {
#     "p0": ShapeSpec(channels=96),
#     "p1": ShapeSpec(channels=192),
#     "p2": ShapeSpec(channels=384),
#     "p3": ShapeSpec(channels=768),
# }

# set output dir
train.output_dir = output_dir

# change the number of classes
model.num_classes = metadata.num_classes

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.test.num_workers = 4

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# set the random seed
train.seed = 2025
