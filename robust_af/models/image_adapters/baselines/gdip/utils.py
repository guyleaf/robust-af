from typing import Union

import torch


def tanh01(x: torch.Tensor):
    """Shifts tanh from the [-1, 1] range to the [0, 1 range] and returns it for the given input.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Constrained tanh
    """
    return torch.tanh(x) * 0.5 + 0.5


def tanhlr(x: torch.Tensor, left: float, right: float):
    """Returns tanh constrained to a particular range

    Args:
        x (torch.Tensor): Input tensor
        left (float): Left bound
        right (float): Right bound

    Returns:
        torch.Tensor: Constrained tanh
    """
    return tanh01(x) * (right - left) + left


def rgb2lum(img: torch.Tensor):
    """_summary_

    Args:
        img (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    img = 0.27 * img[:, 0] + 0.67 * img[:, 1] + 0.06 * img[:, 2]
    return img[:, None]


def lerp(
    a: Union[torch.Tensor, int, float],
    b: Union[torch.Tensor, int, float],
    alpha: Union[torch.Tensor, int, float],
):
    return (1 - alpha) * a + alpha * b


def min_max_normalize(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())
