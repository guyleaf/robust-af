from typing import Union

import torch
import torch.nn as nn
from rtdetrv2.core import GLOBAL_CONFIG, register


@register()
class MultiScaleProcessor(nn.Module):
    def __init__(self, modules: list[Union[dict, nn.Module]]):
        super().__init__()
        module_mapper = nn.ModuleList()
        for module in modules:
            if isinstance(module, dict):
                name = module.pop("type")
                module: nn.Module = getattr(
                    GLOBAL_CONFIG[name]["_pymodule"], GLOBAL_CONFIG[name]["_name"]
                )(**module)
                module_mapper.append(module)
            elif isinstance(module, nn.Module):
                module_mapper.append(module)
            else:
                raise ValueError("The module should be a dict or nn.Module.")

        self.module_mapper = module_mapper

    def forward(self, *inputs: list[torch.Tensor]):
        new_feats = []
        for feats, module in zip(zip(*inputs), self.module_mapper):
            new_feats.append(module(*feats))
        return new_feats
