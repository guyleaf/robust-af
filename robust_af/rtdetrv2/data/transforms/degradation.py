import random
from typing import Optional, Union

import numpy as np
import torchvision.transforms.v2 as T
from PIL import Image
from rtdetrv2.core import register
from rtdetrv2.misc.dist_utils import get_rank

from ....transforms import apply_degradation, sample_degradation_name
from ....utils import get_worker_id


@register()
class Degradation(T.Transform):
    """
    Applies a degradation to an image.
    """

    _transformed_types = (Image.Image,)

    def __init__(
        self,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        identity: bool = True,
        degradations: Optional[Union[list[str], set[str]]] = None,
        ignored_degradations: Union[list[str], set[str]] = [],
    ) -> None:
        """
        Initializes the Degradation class.

        Args:
            name (str, optional): Use specific degradation. If None, randomly sample from degradations.
            seed (int, optional): The seed to determine the randomness.
            identity (bool): Whether include identity (skip connection) in degradations.
            degradations (array-like, optional): Apply specific degradations. If it is not None, identity option will be ignored.
            ignored_degradations (array-like): The degradations to be ignored.
        """
        super().__init__()
        self.name = name
        self.seed = seed
        self.identity = identity
        self.degradations = degradations
        self.ignored_degradations = ignored_degradations
        self._init_random(None)

        assert not (self.name is not None and self.degradations is not None), (
            "The name and degradations cannot be set at the same time."
        )
        if self.name is not None:
            print(f"Specified degradation: {self.name}")
        elif self.degradations is not None:
            print(f"Specified degradations: {', '.join(self.degradations)}")

    def _init_random(self, seed: Optional[int]):
        if seed is None:
            self.np_random_generator = None
            self.py_random = None
        else:
            assert seed >= 0
            self.np_random_generator = np.random.default_rng(seed)
            self.py_random = random.Random(seed)

    def _apply_degradation(self, img: Image.Image, name: str):
        assert isinstance(img, Image.Image) and img.mode == "RGB"
        img = np.asarray(img)
        img = apply_degradation(
            name,
            img,
            np_random_generator=self.np_random_generator,
            py_random=self.py_random,
        )
        return Image.fromarray(img)

    def make_params(self, flat_inputs: list):
        if self.np_random_generator is None or self.py_random is None:
            if self.seed is None:
                # determine the seed by global generator
                seed = random.randrange(0, 2**32)
            else:
                # init random generators lazily to work correctly
                # ex. transform in a dataloader worker
                worker_id = get_worker_id()
                seed = (self.seed + get_rank() + worker_id) % (2**32)
            self._init_random(seed)

        name = self.name
        if name is None:
            name = sample_degradation_name(
                identity=self.identity,
                transforms=self.degradations,
                ignored_transforms=self.ignored_degradations,
                np_random_generator=self.np_random_generator,
                py_random=self.py_random,
            )
        self._params = dict(name=name)
        return self._params

    def transform(self, inpt: Image.Image, params: dict):
        inpt = self._apply_degradation(inpt, name=params["name"])
        return inpt

    def forward(self, *inputs):
        outputs = super().forward(*inputs)
        _, target, _ = outputs
        target["degradation"] = self._params["name"]
        return outputs
