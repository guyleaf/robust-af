
from detectron2.data import MetadataCatalog
from detrex.config.configs.common.common_schedule import multistep_lr_scheduler
from omegaconf import OmegaConf

from robust_u2u_od.detrex.data.datasets.register_robust_anti_uav import DATASET_NAME

# train with 8/4/2/1 GPUs (bs per gpu = 2)
BATCH_SIZES = [16, 8, 4, 2]
EPOCHS = [
    (50, 40, 0),
    (36, 30, 0),
    (24, 20, 0),
    (12, 11, 0),
    (50, 40, 1e-3),
    (12, 11, 1e-3),
]


def default_robust_anti_uav_scheduler(
    total_batch_size: int,
    epochs: int = 50,
    decay_epochs: int = 40,
    warmup_epochs: int = 0,
):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers. LR is decayed once at the end of training.

    Args:
        total_batch_size (int): total batch size (all gpus)
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    num_images: int = MetadataCatalog.get(f"{DATASET_NAME}_train").num_images
    num_batches = num_images // total_batch_size

    # num_batches == num_iters in training
    total_steps = epochs * num_batches
    decay_steps = decay_epochs * num_batches
    warmup_steps = warmup_epochs * num_batches

    return multistep_lr_scheduler(
        values=[1, 0.1],
        warmup_steps=warmup_steps,
        num_updates=total_steps,
        milestones=[decay_steps, total_steps],
    )


def default_robust_anti_uav_detr_schedulers(
    batch_sizes: list[int] = BATCH_SIZES, epochs: list[tuple[int, int, int]] = EPOCHS
):
    schedulers = OmegaConf.create()
    for bs in batch_sizes:
        for total_epochs, decay_epochs, warmup_epochs in epochs:
            key = f"lr_multiplier_{epochs}ep"
            if warmup_epochs != 0:
                key += "_warmup"
            key += f"_{bs}bs"

            schedulers[key] = default_robust_anti_uav_scheduler(
                total_epochs, decay_epochs, warmup_epochs
            )
    return schedulers


detr_schedulers = default_robust_anti_uav_detr_schedulers()
