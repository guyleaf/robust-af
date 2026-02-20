from robust_af.detrex.configs import get_config
from robust_af.detrex.data.datasets.register_robust_dut_anti_uav import DATASET_NAME

from .robust_dino_r50_4scale_12ep_spatial import model, train

test_dataset_name = f"{DATASET_NAME}_low"
dataloader = get_config(f"datasets/{test_dataset_name}_detr.py").robust_dataloader

output_dir = f"./outputs/robust_dino_r50_4scale/{DATASET_NAME}_low/{test_dataset_name}/robust_dino_r50_4scale_12ep_spatial_testing_degraded_2"

# ==============================================================

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
