import random
from typing import Optional, Union

import numpy as np
import torchvision.transforms.v2 as T
from PIL import Image
from rtdetrv2.core import register
from rtdetrv2.misc.dist_utils import get_rank
from torch.utils.data import get_worker_info

from ....transforms import apply_degradation


def _get_worker_id():
    worker_info = get_worker_info()
    if worker_info is None:
        return 0
    else:
        return worker_info.id


@register()
class Degradation(T.Transform):
    """
    Applies a degradation to an image.
    """

    _transformed_types = (Image.Image,)

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
        super().__init__()
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

    def _apply_degradation(self, img: Image.Image):
        assert isinstance(img, Image.Image) and img.mode == "RGB"
        img = np.asarray(img)
        img = apply_degradation(
            img,
            identity=self.identity,
            ignored_transforms=self.ignored_degradations,
            np_random_generator=self.np_random_generator,
            py_random=self.py_random,
        )
        return Image.fromarray(img)

    def transform(self, inpt: Image.Image, params: dict):
        if self.np_random_generator is None or self.py_random is None:
            if self.seed is None:
                # determine the seed by global generator
                seed = random.randrange(0, 2**32)
            else:
                # init random generators lazily to work correctly
                # ex. transform in a dataloader worker
                worker_id = _get_worker_id()
                seed = (self.seed + get_rank() + worker_id) % (2**32)
            self._init_random(seed)
        inpt = self._apply_degradation(inpt)
        return inpt
