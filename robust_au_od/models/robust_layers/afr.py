import logging
from typing import Union

import torch
import torch.nn as nn

from ...utils import is_debug_mode


def _check_nan(x):
    if not is_debug_mode():
        return

    logger = logging.getLogger("detectron2")
    if torch.any(torch.isnan(x)):
        logger.error("x has NaNs.", stack_info=True)
    if torch.any(torch.isinf(x)):
        logger.error("x has infs.", stack_info=True)


class AFR(nn.Module):
    """Anti-degradation Feature Restoration Module"""

    def __init__(
        self,
        embed_dims: int = 256,
        affine: bool = False,
    ):
        super().__init__()
        s_block = SpatialBlock(embed_dims, embed_dims, 3, affine=affine)
        self.sf_block = SpatialFusionBlock(s_block, embed_dims)

        # 0.262656M
        self.f_block = FrequencyBlock(embed_dims * 2)

        # 1.179904M
        self.conv = nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        _check_nan(x)
        x = self.sf_block(x)
        _check_nan(x)
        x = self.f_block(x)
        _check_nan(x)
        x = self.conv(x)
        _check_nan(x)
        return x


class SpatialAFRDebug(nn.Module):
    """Anti-degradation Feature Restoration Module (spatial only)"""

    def __init__(
        self,
        embed_dims: int = 256,
        affine: bool = False,
    ):
        super().__init__()
        s_block = SpatialBlockDebug(embed_dims, embed_dims, 3, affine=affine)
        self.sf_block = SpatialFusionBlock(s_block, embed_dims)

        self.conv = nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        x = self.sf_block(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class SpatialAFR(nn.Module):
    """Anti-degradation Feature Restoration Module (spatial only)"""

    def __init__(
        self,
        embed_dims: int = 256,
        affine: bool = False,
    ):
        super().__init__()
        s_block = SpatialBlock(embed_dims, embed_dims, 3, affine=affine)
        self.sf_block = SpatialFusionBlock(s_block, embed_dims)

        self.conv = nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        x = self.sf_block(x)
        x = self.conv(x)
        return x


class FrequencyAFR(nn.Module):
    """Anti-degradation Feature Restoration Module (frequency only)"""

    def __init__(self, embed_dims: int = 256):
        super().__init__()
        self.f_block = FrequencyBlock(embed_dims)

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, c, h, w]
        x = self.f_block(x)
        return x


class SpatialFusionBlock(nn.Module):
    def __init__(
        self,
        s_block: nn.Module,
        embed_dims: int,
    ):
        super().__init__()
        self.s_block = s_block
        # concat along channel dimension
        # 0.296528M
        self.ca_block = SEBlock(embed_dims * 2)

    def forward(self, x: torch.Tensor):
        x_in = self.s_block(x)
        _check_nan(x_in)
        x_all = torch.cat([x, x_in], dim=1)
        output = self.ca_block(x_all)
        return output


class SpatialBlockDebug(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        padding: Union[int, tuple[int, int]] = 1,
        bias: bool = False,
        affine: bool = False,
    ):
        super().__init__()
        self.IN = nn.InstanceNorm2d(in_channels, affine=affine)

        # 256 * 256 * 3 * 3 = 589.824K = 0.589824M
        # self.conv = nn.Conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     bias=bias,
        # )
        # self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        s_input = self.IN(x)
        _check_nan(s_input)
        # s_input = self.relu(s_input)
        # _check_nan(s_input)
        # out = self.conv(s_input)
        # _check_nan(out)
        return s_input


class SpatialBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        padding: Union[int, tuple[int, int]] = 1,
        bias: bool = False,
        affine: bool = False,
    ):
        super().__init__()
        self.IN = nn.InstanceNorm2d(in_channels, affine=affine)

        # 256 * 256 * 3 * 3 = 589.824K = 0.589824M
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        s_input = self.IN(x)
        _check_nan(s_input)
        s_input = self.relu(s_input)
        _check_nan(s_input)
        out = self.conv(s_input)
        _check_nan(out)
        return out


# 0.296528M
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        assert channels % reduction == 0, (
            "The channel size should be divisible by reduction ratio."
        )
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        batch_size, channels, _, _ = x.size()
        squeeze = self.squeeze(x).view(batch_size, channels)
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)
        return x * excitation


# 0.262656M
class FrequencyBlock(nn.Module):
    def __init__(self, embed_dims: int):
        super().__init__()
        self.conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

    def forward(self, x: torch.Tensor):
        _check_nan(x)
        fft_map = torch.fft.fft2(x, dim=(-2, -1))
        _check_nan(fft_map)

        magnitude_map = torch.abs(fft_map)
        phase_map = torch.angle(fft_map)
        _check_nan(phase_map)

        modified_magnitude = self.conv(magnitude_map)

        real_part = modified_magnitude * torch.cos(phase_map)
        imag_part = modified_magnitude * torch.sin(phase_map)
        modified_fft_map = torch.complex(real_part, imag_part)

        reconstructed_x = torch.real(torch.fft.ifft2(modified_fft_map, dim=(-2, -1)))

        return reconstructed_x
