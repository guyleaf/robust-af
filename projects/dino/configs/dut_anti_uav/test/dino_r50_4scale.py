from detectron2.data.catalog import MetadataCatalog
from detectron2.layers import ShapeSpec

from robust_au_od.detrex.configs import get_config
from robust_au_od.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME
from robust_au_od.detrex.data.datasets.register_dut_anti_uav import (
    DATASET_NAME as TEST_DATASET_NAME,
)

from ...models.dino_r50 import model

degraded = False
test_dataset_name = TEST_DATASET_NAME

dataset = get_config(f"datasets/{test_dataset_name}_detr.py")
if degraded:
    dataloader = dataset.robust_dataloader
else:
    dataloader = dataset.dataloader

train = get_config("train.py").train
metadata = MetadataCatalog.get(test_dataset_name)

use_paper_pos = True
suffix = "_degraded" if degraded else ""
output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}/{test_dataset_name}/dino_r50_4scale_36ep_5e-5_lr{suffix}"

# ==============================================================

# dataloader.test.dataset.names = f"{test_dataset_name}_test"

# for tsne vis
model.backbone.out_features = ["stem", "res2", "res3", "res4", "res5"]
model.neck.input_shapes = {
    "stem": ShapeSpec(channels=64),
    "res2": ShapeSpec(channels=256),
    "res3": ShapeSpec(channels=512),
    "res4": ShapeSpec(channels=1024),
    "res5": ShapeSpec(channels=2048),
}

if use_paper_pos:
    # use the original implementation of dab-detr position embedding if training epochs > 12.
    model.position_embedding.temperature = 20
    model.position_embedding.offset = 0.0

# set output dir
train.output_dir = output_dir

# change the number of classes
model.num_classes = metadata.num_classes

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.test.num_workers = 4
if degraded:
    # (robust_dataloader only) disable skip connection in degradations
    dataloader.test.mapper.identity = False

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# set the random seed
train.seed = 2025
