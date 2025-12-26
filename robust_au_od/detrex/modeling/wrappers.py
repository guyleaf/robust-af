import torch.nn as nn


def build_module_dict(**kwargs: nn.Module):
    return nn.ModuleDict(kwargs)
