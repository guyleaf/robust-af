from detectron2.data import MetadataCatalog
from detrex.config.configs.common.common_schedule import (
    cosine_lr_scheduler,
    linear_lr_scheduler,
    multistep_lr_scheduler,
)
from omegaconf import OmegaConf

from robust_au_od.detrex.utils import count_coco_images

# train with 8/4/2/1 GPUs (bs per gpu = 2)
BATCH_SIZES = [16, 8, 4, 2]
MULTISTEP_SETTINGS = [
    (50, 40, 0),
    (36, 30, 0),
    (24, 20, 0),
    (12, 11, 0),
    (50, 40, 1e-3),
    (36, 30, 1e-3),
    (24, 20, 1e-3),
    (12, 11, 1e-3),
]
LINEAR_SETTINGS = [
    (50, 0),
    (36, 0),
    (24, 0),
    (12, 0),
    (50, 1e-3),
    (36, 1e-3),
    (24, 1e-3),
    (12, 1e-3),
]
COSINE_SETTINGS = LINEAR_SETTINGS


def default_multistep_lr_scheduler(
    dataset_name: str,
    total_batch_size: int,
    epochs: int = 50,
    decay_epochs: int = 40,
    warmup_epochs: float = 0,
):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers. LR is decayed once at the end of training.

    Args:
        total_batch_size (int): total batch size (all gpus)
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (float): warmup epochs (support fraction).

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    json_file: int = MetadataCatalog.get(dataset_name).json_file
    num_images = count_coco_images(json_file)
    num_batches = num_images // total_batch_size

    # num_batches == num_iters in training
    total_steps = epochs * num_batches
    decay_steps = decay_epochs * num_batches
    warmup_steps = warmup_epochs * num_batches

    return multistep_lr_scheduler(
        values=[1, 0.1],
        warmup_steps=warmup_steps,
        num_updates=total_steps,
        milestones=[decay_steps],
    )


def default_linear_lr_scheduler(
    dataset_name: str,
    total_batch_size: int,
    epochs: int = 50,
    warmup_epochs: float = 0,
):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers. LR is decayed once at the end of training.

    Args:
        total_batch_size (int): total batch size (all gpus)
        epochs (int): total training epochs.
        warmup_epochs (float): warmup epochs (support fraction).

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    json_file: int = MetadataCatalog.get(dataset_name).json_file
    num_images = count_coco_images(json_file)
    num_batches = num_images // total_batch_size

    # num_batches == num_iters in training
    total_steps = epochs * num_batches
    warmup_steps = warmup_epochs * num_batches

    return linear_lr_scheduler(
        1,
        0.1,
        total_steps,
        warmup_steps,
    )


def default_cosine_lr_scheduler(
    dataset_name: str,
    total_batch_size: int,
    epochs: int = 50,
    warmup_epochs: float = 0,
):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers. LR is decayed once at the end of training.

    Args:
        total_batch_size (int): total batch size (all gpus)
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (float): warmup epochs (support fraction).

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    json_file: int = MetadataCatalog.get(dataset_name).json_file
    num_images = count_coco_images(json_file)
    num_batches = num_images // total_batch_size

    # num_batches == num_iters in training
    total_steps = epochs * num_batches
    warmup_steps = warmup_epochs * num_batches

    return cosine_lr_scheduler(
        1,
        0.1,
        total_steps,
        warmup_steps,
    )


def default_detr_schedulers(
    dataset_name: str,
    batch_sizes: list[int] = BATCH_SIZES,
    multistep_settings: list[tuple[int, int, float]] = MULTISTEP_SETTINGS,
    linear_settings: list[tuple[int, float]] = LINEAR_SETTINGS,
    cosine_settings: list[tuple[int, float]] = COSINE_SETTINGS,
):
    schedulers = OmegaConf.create()
    for bs in batch_sizes:
        for total_epochs, decay_epochs, warmup_epochs in multistep_settings:
            key = f"lr_multiplier_{total_epochs}ep"
            if warmup_epochs != 0:
                key += "_warmup"
            key += f"_{bs}bs"

            schedulers[key] = default_multistep_lr_scheduler(
                dataset_name,
                bs,
                epochs=total_epochs,
                decay_epochs=decay_epochs,
                warmup_epochs=warmup_epochs,
            )
        for total_epochs, warmup_epochs in linear_settings:
            key = f"linear_lr_multiplier_{total_epochs}ep"
            if warmup_epochs != 0:
                key += "_warmup"
            key += f"_{bs}bs"

            schedulers[key] = default_linear_lr_scheduler(
                dataset_name,
                bs,
                epochs=total_epochs,
                warmup_epochs=warmup_epochs,
            )
        for total_epochs, warmup_epochs in cosine_settings:
            key = f"cosine_lr_multiplier_{total_epochs}ep"
            if warmup_epochs != 0:
                key += "_warmup"
            key += f"_{bs}bs"

            schedulers[key] = default_cosine_lr_scheduler(
                dataset_name,
                bs,
                epochs=total_epochs,
                warmup_epochs=warmup_epochs,
            )
    return schedulers
