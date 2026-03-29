import functools
import math
from typing import Optional, Union

import torch


def tanh01(x: torch.Tensor):
    return torch.tanh(x) * 0.5 + 0.5


def tanhlr(x: torch.Tensor, left: float, right: float, bias: float = 0):
    """Returns tanh constrained to a particular range

    Args:
        x (torch.Tensor): Input tensor
        left (float): Left bound
        right (float): Right bound

    Returns:
        torch.Tensor: Constrained tanh
    """
    return tanh01(x + bias) * (right - left) + left


def get_tanh(left: float, right: float, initial: Optional[float] = None):
    if initial is not None:
        if left < initial < right:
            bias = math.atanh(2 * (initial - left) / (right - left) - 1)
        else:
            raise ValueError(f"The initial ({initial}) must be in ({left}, {right}).")
    else:
        bias = 0
    return functools.partial(tanhlr, left=left, right=right, bias=bias)


def rgb2lum(image: torch.Tensor):
    # [b, h, w]
    image = 0.27 * image[:, 0] + 0.67 * image[:, 1] + 0.06 * image[:, 2]
    # [b, 1, h, w]
    return image[:, None]


def lerp(
    a: Union[torch.Tensor, int, float],
    b: Union[torch.Tensor, int, float],
    alpha: Union[torch.Tensor, int, float],
):
    return (1 - alpha) * a + alpha * b
