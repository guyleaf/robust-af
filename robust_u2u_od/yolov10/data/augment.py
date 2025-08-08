import random
from copy import deepcopy
from typing import Optional, Union

import cv2
import numpy as np
import torch
from ultralytics.data import BaseDataset
from ultralytics.data.augment import Compose
from ultralytics.utils import RANK

from ...transforms import apply_degradation


class Degradation:
    """
    Applies a degradation to an image.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        transforms: Compose,
        seed: Optional[int] = None,
        identity: bool = True,
        ignored_degradations: Union[list[str], set[str]] = [],
    ) -> None:
        """
        Initializes the Degradation class.

        Args:
            seed (float, optional): The seed to determine the randomness.
            identity (bool): Whether include identity (skip connection) in degradations.
            ignored_degradations (array-like): The degradations to be ignored.
        """
        if seed is not None:
            seed += RANK + 1
        self._init_random(seed)

        self.transforms = transforms
        self.identity = identity
        self.ignored_degradations = ignored_degradations

        # TODO: support mosaic & mixup with multiple degraded images by wrapping get_image_and_label()?

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

    def _save_random_states(self):
        py_random_state = random.getstate()
        np_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()
        return py_random_state, np_random_state, torch_random_state

    def _restore_random_states(
        self,
        py_random_state: tuple,
        np_random_state: dict,
        torch_random_state: torch.Tensor,
    ):
        random.setstate(py_random_state)
        np.random.set_state(np_random_state)
        torch.random.set_rng_state(torch_random_state)

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
            # determine the seed by global generator
            seed = random.randrange(0, 2**32)
            self._init_random(seed)

        # pop img first to avoid extra copying
        clear_img = labels.pop("img")
        clear_labels = deepcopy(labels)

        img = self._apply_degradation(clear_img)

        # NOTE: this method can only control the randomness of input-independent augmentations

        # normal ver.
        # NOTE: the augmentations may have in-place operation
        random_states = self._save_random_states()
        clear_labels["img"] = clear_img
        clear_labels = self.transforms(clear_labels)
        clear_img = clear_labels["img"]

        # degraded ver.
        self._restore_random_states(*random_states)
        labels["img"] = img
        labels = self.transforms(labels)
        img = labels["img"]
        assert clear_img.shape == img.shape, (
            "The degraded and normal images should be a pair."
        )

        labels["clear_img"] = clear_img
        labels["img"] = img
        return labels
