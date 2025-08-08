import random
from contextlib import ContextDecorator
from types import TracebackType
from typing import Optional, Type

import numpy as np
import torch


class RandomContext(ContextDecorator):
    def __init__(
        self,
        np_random_generator: Optional[np.random.Generator] = None,
        py_random: Optional[random.Random] = None,
        torch_generator: Optional[torch.Generator] = None,
        # TODO: support GPU version?
    ):
        self.random_generator = np_random_generator
        self.py_random = py_random
        self.torch_generator = torch_generator

        self.original_np_bit_generator = None
        self.original_py_random_state = None
        self.original_torch_rng_state = None

    def __enter__(self):
        if self.random_generator is not None:
            self.original_np_bit_generator = np.random.get_bit_generator()
            np.random.set_bit_generator(self.random_generator.bit_generator)

        if self.py_random is not None:
            self.original_py_random_state = random.getstate()
            random.setstate(self.py_random.getstate())

        if self.torch_generator is not None:
            assert self.torch_generator.device.type == "cpu"
            self.original_torch_rng_state = torch.random.get_rng_state()
            torch.random.set_rng_state(self.torch_generator.get_state())

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        if self.original_np_bit_generator is not None:
            np.random.set_bit_generator(self.original_np_bit_generator)

        if self.original_py_random_state is not None:
            self.py_random.setstate(random.getstate())
            random.setstate(self.original_py_random_state)

        if self.original_torch_rng_state is not None:
            self.torch_generator.set_state(torch.random.get_rng_state())
            torch.random.set_rng_state(self.original_torch_rng_state)
