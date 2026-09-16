from copy import deepcopy

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detrex.config import get_config
from omegaconf import DictConfig

from robust_af.detrex.data.build import get_detection_dataset_dicts
from robust_af.detrex.data.dataset_mappers import RobustDetrDatasetMapper
from robust_af.detrex.data.datasets.register_robust_dut_anti_uav import DATASET_NAME

# normal version of dataset

dataloader: DictConfig = get_config("common/data/coco_detr.py").dataloader
dataloader.train.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_train", filter_empty=False
)
dataloader.train.persistent_workers = True
dataloader.train.pin_memory = True

dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names=f"{DATASET_NAME}_val", filter_empty=False
)

# robust version of dataset

robust_dataloader = deepcopy(dataloader)
robust_dataloader.train.mapper = L(RobustDetrDatasetMapper)(
    augmentations=[
        L(T.RandomFlip)(),
        L(T.RandomApply)(
            tfm_or_aug=L(T.AugmentationList)(
                augs=[
                    L(T.ResizeShortestEdge)(
                        short_edge_length=(400, 500, 600),
                        sample_style="choice",
                    ),
                    L(T.RandomCrop)(
                        crop_type="absolute_range",
                        crop_size=(384, 600),
                    ),
                ]
            ),
            prob=0.5,
        ),
        L(T.ResizeShortestEdge)(
            short_edge_length=(
                480,
                512,
                544,
                576,
                608,
                640,
                672,
                704,
                736,
                768,
                800,
            ),
            max_size=1333,
            sample_style="choice",
        ),
    ],
    is_train=True,
    use_instance_mask=False,
    image_format="RGB",
)


robust_dataloader.test.mapper = L(RobustDetrDatasetMapper)(
    augmentations=[
        L(T.ResizeShortestEdge)(
            short_edge_length=800,
            max_size=1333,
        ),
    ],
    is_train=False,
    use_instance_mask=False,
    image_format="RGB",
)
