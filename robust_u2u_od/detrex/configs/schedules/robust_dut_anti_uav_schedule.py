from robust_u2u_od.detrex.data.datasets.register_robust_dut_anti_uav import DATASET_NAME

from .common_schedule import default_detr_schedulers

detr_schedulers = default_detr_schedulers(f"{DATASET_NAME}_train")
