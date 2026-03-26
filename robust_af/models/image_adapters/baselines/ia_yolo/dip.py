import torch
import torch.nn as nn

from .cnn_pp import CNNPP
from .filters import Filters

LOW_LIGHT_CFG = dict(
    filter_cfgs=dict(
        usm=dict(coef_range=(0, 2.5)),
        gamma=dict(coef_range=2.5),
        tone=dict(curve_steps=4),
    ),
    ignored_filters=["defog"],
)


class DIP(nn.Module):
    def __init__(self, pp_cfg: dict = {}, **kwargs):
        """Initialize DIP module (IA-YOLO)

        Defaults to the config of normal version.

        Args:
            pp_cfg (dict, optional): configs for CNNPP module. Defaults to {}.
            filter_cfgs (dict, optional): config for each filter. Defaults to {}.
            ignored_filters (list[str], optional): _description_. Defaults to [].
        """
        super().__init__()

        # build filters
        self.filters = Filters(**kwargs)

        # build predictor
        self.param_predictor = CNNPP(
            num_filter_params=self.filters.num_parameters, **pp_cfg
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # [b, num_parameters]
        parameters = self.param_predictor(images)
        images = self.filters(images, parameters)
        return images
