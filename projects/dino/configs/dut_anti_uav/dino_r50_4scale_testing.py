from robust_au_od.detrex.configs import get_config
from robust_au_od.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME
from robust_au_od.detrex.data.datasets.register_dut_anti_uav import (
    DATASET_NAME as TEST_DATASET_NAME,
)

from .dino_r50_4scale_12ep import model, train

test_dataset_name = TEST_DATASET_NAME
dataloader = get_config(f"datasets/{test_dataset_name}_detr.py").robust_dataloader

output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}/{test_dataset_name}/dino_r50_4scale_12ep_1e-4_lr_testing_degraded"

# ==============================================================

# dataloader.test.dataset.names = f"{test_dataset_name}_test"

# set output dir
train.output_dir = output_dir

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.test.num_workers = 4
# (robust_dataloader only) disable skip connection in degradations
dataloader.test.mapper.identity = False

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# set the random seed
train.seed = 2025
