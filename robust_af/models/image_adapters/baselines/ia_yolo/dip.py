from typing import Union

import torch
import torch.nn as nn

from .cnn_pp import CNNPP
from .filters import FILTERS, Filter

LOW_LIGHT_CFG = dict(
    filter_cfgs=dict(
        usm=dict(coef_range=(0, 2.5)),
        gamma=dict(coef_range=2.5),
        tone=dict(curve_steps=4),
    ),
    ignored_filters=["defog"],
)


class DIP(nn.Module):
    def __init__(
        self,
        filter_cfgs: dict = {},
        ignored_filters: Union[list[str], set[str]] = [],
        pp_cfg: dict = {},
    ):
        """Initialize DIP module (IA-YOLO)

        Defaults to the config of normal version.

        Args:
            pp_cfg (dict, optional): configs for CNNPP module. Defaults to {}.
            filter_cfgs (dict, optional): config for each filter. Defaults to {}.
            ignored_filters (list[str], optional): _description_. Defaults to [].
        """
        super().__init__()

        # build filters
        filters: list[Filter] = []
        ignored_filters = set(ignored_filters)
        num_filter_parameters = 0
        for name, filter_cls in FILTERS.items():
            if name in ignored_filters:
                continue

            cfg = filter_cfgs.get(name, {})
            instance = filter_cls(**cfg)

            filters.append(instance)
            num_filter_parameters += instance.num_parameters
        self.filters = filters

        # build predictor
        self.param_predictor = CNNPP(num_filter_params=num_filter_parameters, **pp_cfg)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # [b, num_parameters]
        parameters = self.param_predictor(images)
        offset = 0
        for instance in self.filters:
            num_parameters = instance.num_parameters
            filter_parameters = parameters[:, offset : offset + num_parameters]
            images = instance(images, filter_parameters)
            offset += num_parameters
        return images
