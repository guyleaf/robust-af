import os

from robust_u2u_od.detrex.configs import get_config
from robust_u2u_od.detrex.data.datasets.register_dut_anti_uav import (
    DATASET_NAME as TESTING_DATASET_NAME,
)
from robust_u2u_od.detrex.data.datasets.register_robust_anti_uav import DATASET_NAME

from .dino_r50_4scale_12ep import model, train

dataloader = get_config(f"datasets/{TESTING_DATASET_NAME}_detr.py").dataloader

output_dir = os.path.expanduser(os.getenv("R_U2U_OD_WORKDIR", "./outputs"))
output_dir = os.path.join(
    output_dir,
    f"dino_r50_4scale/{DATASET_NAME}_low/{TESTING_DATASET_NAME}/dino_r50_4scale_12ep_testing",
)

# ==============================================================

# set output dir
train.output_dir = output_dir

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.test.num_workers = 4

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
