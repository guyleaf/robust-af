import os
from contextlib import contextmanager
from functools import partial
from typing import Optional, TypeVar

import torch

T = TypeVar("T", bound=object)


def format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """Formats an integer for printing in a table or model representation.

    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int): The integer to format.
        sig_figs (int): The number of significant figures to keep.
            Defaults to 3.
        hide_zero (bool): If True, x=0 is replaced with an empty string
            instead of '0'. Defaults to False.

    Returns:
        str: The formatted string.

    Copied from https://github.com/open-mmlab/mmengine/blob/main/mmengine/analysis/print_helper.py#L20
    """
    if hide_zero and x == 0:
        return ""

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return f"{{:.{sig_figs}f}}".format(x).rstrip("0").rstrip(".")

    if abs(x) > 1e14:
        return fmt(x / 1e15) + "P"
    if abs(x) > 1e11:
        return fmt(x / 1e12) + "T"
    if abs(x) > 1e8:
        return fmt(x / 1e9) + "G"
    if abs(x) > 1e5:
        return fmt(x / 1e6) + "M"
    if abs(x) > 1e2:
        return fmt(x / 1e3) + "K"
    return str(x)


def allow_tf32_precision(mode: bool = True):
    """(Recommended) Use NVIDIA_TF32_OVERRIDE=0 to disable globally."""
    assert isinstance(mode, bool)
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = mode

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = mode


def is_debug_mode():
    mode = os.environ.get("DEBUG", "false")
    return mode.lower() in ("1", "true")


@contextmanager
def wrap_method(
    instance: T,
    name: Optional[str] = None,
    args: tuple = tuple(),
    kwargs: dict = {},
):
    if name is None:
        name = "__call__"

    old_func = getattr(instance, name)
    func = partial(old_func, *args, **kwargs)
    setattr(instance, name, func)

    try:
        yield instance
    finally:
        setattr(instance, name, old_func)
