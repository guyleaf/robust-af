import logging
import random
from typing import Optional

import numpy as np
from detectron2.data.transforms import (
    Augmentation,
    NoOpTransform,
    Transform,
)
from detrex.utils import get_rank

from ....transforms import (
    DEGRADATION_TRANSFORMS,
    apply_degradation,
    sample_degradation_name,
)
from ....utils import get_worker_id

LOGGER = logging.getLogger(__name__)


class DegradationTransform(Transform):
    def __init__(
        self,
        name: str,
        np_random_generator: Optional[np.random.Generator] = None,
        py_random: Optional[random.Random] = None,
    ):
        super().__init__()
        assert name in DEGRADATION_TRANSFORMS
        self.name = name
        self.np_random_generator = np_random_generator
        self.py_random = py_random

    def apply_image(self, img: np.ndarray):
        img = apply_degradation(
            self.name,
            img,
            np_random_generator=self.np_random_generator,
            py_random=self.py_random,
        )
        return img

    def apply_coords(self, coords):
        return coords

    def inverse(self) -> Transform:
        """
        The inverse is a no-op.
        """
        return NoOpTransform()


class Degradation(Augmentation):
    """
    Degradation Augmentation is for selecting a random/specific name from degradations.

    The returned `Transform` object is non-deterministic transformation for the specific degradation.
    """

    input_args: Optional[tuple[str]] = tuple()

    def __init__(
        self,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        identity: bool = True,
        ignored_degradations: list[str] = [],
    ):
        super().__init__()
        self.name = name
        self.seed = seed
        self.identity = identity
        self.ignored_degradations = ignored_degradations
        self._init_random(None)

        if self.name is not None:
            LOGGER.info(f"Specified degradation: {self.name}")

    def _init_random(self, seed: Optional[int]):
        if seed is None:
            self.np_random_generator = None
            self.py_random = None
        else:
            assert seed >= 0
            self.np_random_generator = np.random.default_rng(seed)
            self.py_random = random.Random(seed)

    def get_transform(self, *args):
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
                ignored_transforms=self.ignored_degradations,
                np_random_generator=self.np_random_generator,
                py_random=self.py_random,
            )
        return DegradationTransform(
            name, np_random_generator=self.np_random_generator, py_random=self.py_random
        )
