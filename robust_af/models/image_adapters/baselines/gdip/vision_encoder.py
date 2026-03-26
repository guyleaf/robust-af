from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_layers: int = 5,
        base_channels: int = 64,
        conv_cfg: dict = dict(kernel_size=3, stride=1),
        pool_cfg: dict = dict(kernel_size=(3, 3), stride=(2, 2)),
        out_features: Optional[list[str]] = None,
    ):
        super().__init__()
        if out_features is None:
            out_features = [f"conv_{num_layers - 1}"]

        # build layers
        self.layers = nn.ModuleList()
        self._layer_channels: OrderedDict[str, int] = OrderedDict()
        out_channels = base_channels
        for i in range(num_layers):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, **conv_cfg), nn.ReLU(inplace=True)
            )
            self.layers.append(conv)
            self._layer_channels[f"conv_{i}"] = out_channels

            # build pooling layer except for the last layer
            if i < num_layers - 1:
                pool = nn.AvgPool2d(**pool_cfg)
                self.layers.append(pool)
                self._layer_channels[f"pool_{i}"] = out_channels

            in_channels = out_channels
            out_channels *= 2

        self.out_features = set(out_features)
        assert len(self.out_features & self._layer_channels.keys()) == len(
            self.out_features
        )

    @property
    def layer_channels(self):
        return {name: self._layer_channels[name] for name in self.out_features}

    def forward(self, x) -> dict[str, torch.Tensor]:
        outputs = {}
        for name, layer in zip(self._layer_channels, self.layers):
            x = layer(x)
            if name in self.out_features:
                outputs[name] = x
        return outputs
