import random
from contextlib import ContextDecorator
from types import TracebackType
from typing import Optional, Type

import numpy as np
import torch


# TODO: rename to xxx_global_random_states
def save_random_states():
    py_random_state = random.getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    return py_random_state, np_random_state, torch_random_state


def restore_random_states(
    py_random_state: tuple,
    np_random_state: dict,
    torch_random_state: torch.Tensor,
):
    random.setstate(py_random_state)
    np.random.set_state(np_random_state)
    torch.random.set_rng_state(torch_random_state)


class RandomContext(ContextDecorator):
    """A context decorator for controlling the global random states with the following steps.

    1. Saving the global random states
    2. (Optional) Setting the global random states with generators
        ...after finishing
    3. (Optional) Setting the global random states back to generators
    4. Restoring the global random states

    By default, it can be used for reproducing the global random states.

    If skip_if_none is True and all generators are None, no actions during the context.
    """

    def __init__(
        self,
        np_random_generator: Optional[np.random.Generator] = None,
        py_random: Optional[random.Random] = None,
        torch_generator: Optional[torch.Generator] = None,
        skip_if_none: bool = False,
        # TODO: support GPU version?
    ):
        self.np_random_generator = np_random_generator
        self.py_random = py_random
        self.torch_generator = torch_generator
        self.disabled = skip_if_none and (
            np_random_generator is None
            and py_random is None
            and torch_generator is None
        )

        self.original_np_bit_generator = None
        self.original_py_random_state = None
        self.original_torch_rng_state = None

    @property
    def with_numpy(self):
        return self.np_random_generator is not None

    @property
    def with_python(self):
        return self.py_random is not None

    @property
    def with_torch(self):
        return self.torch_generator is not None

    def __enter__(self):
        if self.disabled:
            return

        # 1.
        self.original_np_bit_generator = np.random.get_bit_generator()
        self.original_py_random_state = random.getstate()
        self.original_torch_rng_state = torch.random.get_rng_state()

        # 2.
        if self.with_numpy:
            np.random.set_bit_generator(self.np_random_generator.bit_generator)

        if self.with_python:
            random.setstate(self.py_random.getstate())

        if self.with_torch:
            assert self.torch_generator.device.type == "cpu"
            torch.random.set_rng_state(self.torch_generator.get_state())

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        if self.disabled:
            return

        # 3.
        # In numpy, the bit generator manages its own random states
        # So, we don't need to set back to the generator.

        if self.with_python:
            self.py_random.setstate(random.getstate())

        if self.with_torch:
            self.torch_generator.set_state(torch.random.get_rng_state())

        # 4.
        if self.original_np_bit_generator is not None:
            np.random.set_bit_generator(self.original_np_bit_generator)

        if self.original_py_random_state is not None:
            random.setstate(self.original_py_random_state)

        if self.original_torch_rng_state is not None:
            torch.random.set_rng_state(self.original_torch_rng_state)


# (deprecated, backward compatibility) legacy class
# TODO: remove it
ReproducibleRandomContext = RandomContext
