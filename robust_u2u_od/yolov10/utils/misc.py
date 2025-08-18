import random

import numpy as np
import torch


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
