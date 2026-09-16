from copy import deepcopy

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detrex.config import get_config
from omegaconf import DictConfig, OmegaConf

from robust_af.detrex.data.build import get_detection_dataset_dicts
from robust_af.detrex.data.dataset_mappers import (
    DetrDatasetMapper,
    RobustDetrDatasetMapper,
)
from robust_af.detrex.data.transforms import Degradation
from robust_af.detrex.evaluation import COCOEvaluator

# normal version of dataset

dataloader: DictConfig = get_config("common/data/coco_detr.py").dataloader
dataloader.evaluator = OmegaConf.merge(dataloader.evaluator, L(COCOEvaluator)())
dataloader.train.dataset = OmegaConf.merge(
    dataloader.train.dataset, L(get_detection_dataset_dicts)()
)
dataloader.train.persistent_workers = True
dataloader.train.pin_memory = True
dataloader.train.mapper = L(DetrDatasetMapper)(
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

dataloader.test.dataset = OmegaConf.merge(
    dataloader.test.dataset, L(get_detection_dataset_dicts)()
)
dataloader.test.mapper = L(DetrDatasetMapper)(
    augmentations=[
        L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
    ],
    is_train=False,
    use_instance_mask=False,
    image_format="RGB",
)

# (optional) train for test
dataloader.train_test = deepcopy(dataloader.test)
dataloader.train_test.dataset = L(get_detection_dataset_dicts)(
    names="coco_2017_train", filter_empty=False
)

# robust version of dataset

robust_dataloader = deepcopy(dataloader)

# train
robust_dataloader.train.mapper = L(RobustDetrDatasetMapper)(
    augmentations=robust_dataloader.train.mapper.augmentations,
    robust_augmentations=[L(Degradation)()],
    is_train=True,
    use_instance_mask=False,
    image_format="RGB",
)

# test
# offline augmentation to get consistent performance between val and test.
# TODO: register degraded ver. of COCO
robust_dataloader.test.dataset = L(get_detection_dataset_dicts)(
    names="coco_2017_val_degraded", filter_empty=False
)

# (optional) train for test
# online augmentation same with train subset
robust_dataloader.train_test.mapper = L(DetrDatasetMapper)(
    augmentations=[
        L(Degradation)(identity=False),
        L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
    ],
    is_train=False,
    use_instance_mask=False,
    image_format="RGB",
)
