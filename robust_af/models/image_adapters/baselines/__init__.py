# ruff: noqa: F401
from denet.core import DENet

from .gdip import GDIP
from .ia_yolo import DIP

# if interface is compatible, re-export here directly.
# Otherwise, make a thin adapter for it in the other new file.

__all__ = list(globals().keys())
