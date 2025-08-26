import random
from copy import deepcopy
from types import SimpleNamespace
from typing import Callable, Optional, Union

import cv2
import numpy as np
from ultralytics.data.augment import (
    Compose,
    CopyPaste,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomPerspective,
)
from ultralytics.utils import LOGGER, RANK

from ...transforms import apply_degradation
from ..utils import get_worker_id, restore_random_states, save_random_states


class Identity:
    def __call__(self, labels: dict):
        return labels


class Degradation:
    """
    Applies a degradation to an image.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        identity: bool = True,
        ignored_degradations: Union[list[str], set[str]] = [],
    ) -> None:
        """
        Initializes the Degradation class.

        Args:
            seed (int, optional): The seed to determine the randomness.
            identity (bool): Whether include identity (skip connection) in degradations.
            ignored_degradations (array-like): The degradations to be ignored.
        """
        self.seed = seed
        self.identity = identity
        self.ignored_degradations = ignored_degradations
        self._init_random(None)

    def _init_random(self, seed: Optional[int]):
        if seed is None:
            self.np_random_generator = None
            self.py_random = None
        else:
            assert seed >= 0
            self.np_random_generator = np.random.default_rng(seed)
            self.py_random = random.Random(seed)

    def _apply_degradation(self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_degradation(
            img,
            identity=self.identity,
            ignored_transforms=self.ignored_degradations,
            np_random_generator=self.np_random_generator,
            py_random=self.py_random,
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(img)

    def __call__(self, labels: dict):
        """
        Applies random flip to an image and updates any instances like bounding boxes or keypoints accordingly.

        Args:
            labels (dict): A dictionary containing the keys 'img' and 'instances'. 'img' is the image to be flipped.
                           'instances' is an object containing bounding boxes and optionally keypoints.

        Returns:
            (dict): The same dict with the flipped image and updated instances under the 'img' and 'instances' keys.
        """
        if self.np_random_generator is None or self.py_random is None:
            if self.seed is None:
                # determine the seed by global generator
                seed = random.randrange(0, 2**32)
            else:
                # init random generators lazily to work correctly
                # ex. transform in a dataloader worker
                worker_id = get_worker_id()
                seed = (self.seed + RANK + 1 + worker_id) % (2**32)
            self._init_random(seed)

        img = labels["img"]
        img = self._apply_degradation(img)
        labels["img"] = img
        return labels


class MultiBranch:
    r"""Multiple branch pipeline wrapper.

    Referenced from https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/datasets/transforms/wrappers.py#L13.

    Generate multiple data-augmented versions of the same image.

    Args:
        main_branch (str): which branch is the main branch without appending branch name to the result
        reproduce_randomness (bool): Reproduce the randomness among branches
        branch_transforms (dict): Dict of different pipeline configs
            to be composed.
    """

    def __init__(
        self,
        main_branch: str,
        reproduce_randomness: bool = False,
        **branch_transforms: dict,
    ) -> None:
        assert main_branch in branch_transforms
        assert isinstance(reproduce_randomness, bool)
        self.main_branch = main_branch
        self.reproduce_randomness = reproduce_randomness
        self.branch_transforms = {
            branch: transform if isinstance(transform, Compose) else Compose(transform)
            for branch, transform in branch_transforms.items()
        }

    def __call__(self, labels: dict) -> dict:
        """
        Applies augmentations to an image and updates any instances like bounding boxes or keypoints accordingly.

        Args:
            labels (dict): A dictionary containing the keys 'img' and 'instances'. 'img' is the image to be flipped.
                           'instances' is an object containing bounding boxes and optionally keypoints.

        Returns:
            (dict): The dict with the augmented images and updated instances under the 'img' and 'instances' keys and branches' version.
        """
        multi_results = {}

        # NOTE: this method can only control the randomness of input-independent augmentations
        random_states = save_random_states()
        for branch, transform in self.branch_transforms.items():
            if self.reproduce_randomness:
                restore_random_states(*random_states)
            multi_results[branch] = transform(deepcopy(labels))

        # keep compatibility
        new_labels = multi_results.pop(self.main_branch)
        new_labels.update(multi_results)
        # for branch, result in multi_results.items():
        #     for k, v in result.items():
        #         new_labels[f"{branch}_{k}"] = v
        return new_labels


def robust_v8_transforms(
    dataset,
    imgsz,
    hyp: SimpleNamespace,
    robust_transform: Callable,
    stretch: bool = False,
):
    """Convert images to a size suitable for Robust YOLOv8 training."""
    shared_pre_transforms = [
        CopyPaste(p=hyp.copy_paste),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        ),
    ]
    pre_robust_transform = Compose(
        [
            robust_transform,
            Mosaic(dataset, imgsz=imgsz, pre_transform=robust_transform, p=hyp.mosaic),
            *shared_pre_transforms,
        ]
    )
    pre_transform = Compose(
        [Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), *shared_pre_transforms]
    )

    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning(
                "WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'"
            )
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(
                f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}"
            )

    shared_transforms = [
        # disable color-related augmentations
        # robust_transform will handle them.
        # Albumentations(p=1.0),
        # RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction="vertical", p=hyp.flipud),
        RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
    ]
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            *shared_transforms,
        ]
    ), Compose(
        [
            pre_robust_transform,
            MixUp(dataset, pre_transform=pre_robust_transform, p=hyp.mixup),
            *shared_transforms,
        ]
    )  # transforms


# def _renew_compose_transforms(transforms: Compose):
#     # perform shallow copy to keep transform instances
#     transform_list = copy(transforms.tolist())
#     for i, transform in enumerate(transform_list):
#         if isinstance(transform, Compose):
#             transform_list[i] = _renew_compose_transforms(transform)
#     return Compose(transform_list)


# def _find_mix_transforms(transforms: Compose):
#     transform_list = transforms.tolist()
#     for i, transform in enumerate(transform_list):
#         if isinstance(transform, BaseMixTransform):
#             yield i, transform_list, transform
#         elif isinstance(transform, Compose):
#             yield from _find_mix_transforms(transform)


# def make_degradation_transforms(hyp: SimpleNamespace, transforms: Compose):
#     degradation_transform = Degradation(
#         seed=hyp.degradation["seed"],
#         identity=hyp.degradation["identity"],
#         ignored_degradations=hyp.degradation["ignored_degradations"],
#     )

#     # share the transform instances but different compose instance
#     transforms = _renew_compose_transforms(transforms)

#     # find the mix transforms to prepend the Degradation to pre_transform
#     for i, transform_list, transform in _find_mix_transforms(transforms):
#         if transform.pre_transform is not None:
#             pre_transform = copy(transform.pre_transform.tolist())
#         else:
#             pre_transform = []
#         pre_transform.insert(0, degradation_transform)

#         transform = copy(transform)
#         transform.pre_transform = pre_transform
#         transform_list[i] = transform

#     degraded_transforms = Compose(
#         [
#             degradation_transform,
#             transforms,
#         ]
#     )
#     return transforms
