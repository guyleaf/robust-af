from robust_au_od.detrex.configs import get_config
from robust_au_od.detrex.data.datasets.register_robust_anti_uav import DATASET_NAME

from .dino_r50_4scale_12ep import model, train

test_dataset_name = f"{DATASET_NAME}_low"
dataloader = get_config(f"datasets/{test_dataset_name}_detr.py").robust_dataloader

output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}_low/{test_dataset_name}/robust_dino_r50_4scale_12ep_testing_3"

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
