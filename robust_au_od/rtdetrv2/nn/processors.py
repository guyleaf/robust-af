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

    def forward(self, feats: list[torch.Tensor]):
        assert len(feats) == len(self.module_mapper)
        new_feats = []
        for feat, module in zip(feats, self.module_mapper):
            new_feats.append(module(feat))
        return new_feats
