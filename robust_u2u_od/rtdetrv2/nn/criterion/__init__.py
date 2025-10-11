# ruff: noqa: F401
import torch.nn as nn
from rtdetrv2.core import register

from .robust_criterion import RobustCriterion

MSELoss = register()(nn.MSELoss)
L1Loss = register()(nn.L1Loss)
SmoothL1Loss = register()((nn.SmoothL1Loss))

__all__ = list(globals().keys())
