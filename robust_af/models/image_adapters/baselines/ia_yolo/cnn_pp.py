from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_conv_lrelu(
    in_channels: int,
    out_channels: int,
    # downsample=True scenario
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
    bias: bool = True,
    negative_slope: float = 0.1,
    **kwargs,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs,
        ),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    )


class CNNPP(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_filter_params: int = 15,
        base_channels: int = 16,
        fc_hidden_dim: int = 64,
        image_size: int = 256,
        pool_size: int = 8,
    ):
        super().__init__()
        self.image_size = image_size

        # build convs
        convs = OrderedDict(
            ex_conv0=_build_conv_lrelu(in_channels, base_channels),
            ex_conv1=_build_conv_lrelu(base_channels, 2 * base_channels),
        )
        channels = 2 * base_channels
        for i in range(2, 5):
            convs[f"ex_conv{i}"] = _build_conv_lrelu(channels, channels)
        self.convs = nn.Sequential(convs)

        # build fcs
        self.fcs = nn.Sequential(
            # in TF implementation, only handles square image 256x256.
            # HW: 256 -> 5 convs (2 ** 5) -> 256/32=8
            # C: 3 -> 5 convs -> 32
            # to handle non-square image, we use a common way to fix the dimension by average pooling with 8x8.
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            # 8 * 8 * 32 = 2048
            nn.Linear(pool_size * pool_size * channels, fc_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_hidden_dim, num_filter_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        # 1. rescale to low-res
        # to handle the non-square image, we scale image to 256x256 instead of resizing.
        # in TF implementation, they don't handle image size < 256.
        # so, we suppose the image size >= 256, too.
        scale_factor = self.image_size / min(h, w)
        x = F.interpolate(
            x, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

        # 2. predict filter params
        x = self.convs(x)
        x = self.fcs(x)
        return x
