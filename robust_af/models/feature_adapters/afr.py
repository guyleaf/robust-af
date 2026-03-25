import logging
from typing import Optional, Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ...utils import build_activation, is_debug_mode


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
        activation: Optional[str] = None,
        spatial_cfg: dict = dict(),
    ):
        super().__init__()
        s_block = SpatialBlock(embed_dims, embed_dims, **spatial_cfg)
        self.sf_block = SpatialFusionBlock(s_block, embed_dims)

        self.f_block = FrequencyBlock(embed_dims * 2)

        self.conv = nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1)
        if activation is not None:
            self.act = build_activation(activation)
        else:
            self.act = None

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        # _check_nan(x)
        x = self.sf_block(x)
        # _check_nan(x)
        x = self.f_block(x)
        # _check_nan(x)
        x = self.conv(x)
        # _check_nan(x)
        if self.act is not None:
            x = self.act(x)
            # _check_nan(x)
        return x


class SpatialAFR(nn.Module):
    """Anti-degradation Feature Restoration Module (spatial only)"""

    def __init__(
        self,
        embed_dims: int = 256,
        activation: Optional[str] = None,
        spatial_cfg: dict = dict(),
    ):
        super().__init__()
        s_block = SpatialBlock(embed_dims, embed_dims, **spatial_cfg)
        self.sf_block = SpatialFusionBlock(s_block, embed_dims)

        self.conv = nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1)
        if activation is not None:
            self.act = build_activation(activation)
        else:
            self.act = None

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        x = self.sf_block(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class FrequencyAFR(nn.Module):
    """Anti-degradation Feature Restoration Module (frequency only)"""

    def __init__(self, embed_dims: int = 256):
        super().__init__()
        self.f_block = FrequencyBlock(embed_dims)
        # TODO: do we need a layer for projection?

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, c, h, w]
        x = self.f_block(x)
        return x


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


class SpatialFusionBlock(nn.Module):
    def __init__(
        self,
        s_block: nn.Module,
        embed_dims: int,
    ):
        super().__init__()
        self.s_block = s_block
        self.ca_block = SEBlock(embed_dims * 2)

    def forward(self, x: torch.Tensor):
        x_in = self.s_block(x)
        # _check_nan(x_in)
        x_all = torch.cat([x, x_in], dim=1)
        output = self.ca_block(x_all)
        return output


class SpatialBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]] = 3,
        padding: Union[int, tuple[int, int]] = 1,
        bias: bool = False,
        affine: bool = False,
        conv: bool = True,
        activation: Optional[str] = "LeakyReLU",
        selector: bool = False,
    ):
        super().__init__()
        self.IN = nn.InstanceNorm2d(in_channels, affine=affine)

        if conv:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
        else:
            self.conv = None

        if activation is not None:
            self.act = build_activation(activation)
        else:
            self.act = None

        if selector:
            self.selector = SEBlock(out_channels)
        else:
            self.selector = None

    def forward(self, x: torch.Tensor):
        x = self.IN(x)
        # _check_nan(x)

        if self.conv is not None:
            x = self.conv(x)
            # _check_nan(x)
        if self.act is not None:
            x = self.act(x)
            # _check_nan(x)
        if self.selector is not None:
            x = self.selector(x)
            # _check_nan(x)
        return x


class FrequencyBlock(nn.Module):
    def __init__(self, embed_dims: int):
        super().__init__()
        self.conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # _check_nan(x)
        # disable amp to avoid RuntimeError: cuFFT only supports dimensions whose sizes are powers of two when computing in half precision
        with torch.autocast("cuda", enabled=False):
            fft_map = torch.fft.fft2(x.float(), dim=(-2, -1))
        # _check_nan(fft_map)

        magnitude_map = torch.abs(fft_map)
        phase_map = torch.angle(fft_map)
        # _check_nan(phase_map)

        modified_magnitude = self.conv(magnitude_map)

        real_part = modified_magnitude * torch.cos(phase_map)
        imag_part = modified_magnitude * torch.sin(phase_map)
        modified_fft_map = torch.complex(real_part, imag_part)

        with torch.autocast("cuda", enabled=False):
            modified_fft_map = modified_fft_map.to(torch.cfloat)
            reconstructed_x = torch.fft.ifft2(modified_fft_map, dim=(-2, -1))
        reconstructed_x = torch.real(reconstructed_x)

        return reconstructed_x


# ===================================== Modules for testing only =====================================


class SpatialBlockDebug(nn.Module):
    def __init__(self, embed_dims: int, affine: bool = False):
        super().__init__()
        self.IN = nn.InstanceNorm2d(embed_dims, affine=affine)

    def forward(self, x: torch.Tensor):
        s_input = self.IN(x)
        # _check_nan(s_input)
        return s_input


class SpatialAFRNonParameteric(nn.Module):
    """Anti-degradation Feature Restoration Module (spatial only)"""

    def __init__(
        self,
        embed_dims: int = 256,
    ):
        super().__init__()
        self.IN = nn.InstanceNorm2d(embed_dims)

    def forward(self, x: torch.Tensor, clear_x: torch.Tensor):
        x = self.IN(x)
        target_mean = clear_x.mean(dim=(2, 3), keepdim=True)
        target_std = clear_x.std(dim=(2, 3), keepdim=True)
        return x * target_std + target_mean


class SpatialAFRDebug(nn.Module):
    """Anti-degradation Feature Restoration Module (spatial only)"""

    def __init__(
        self,
        embed_dims: int = 256,
        affine: bool = False,
        activation: Optional[str] = None,
        se: bool = False,
    ):
        super().__init__()
        # s_block = SpatialBlockDebug(embed_dims, affine=affine)
        s_block = SpatialBlock(embed_dims, embed_dims, affine=affine, selector=se)
        self.sf_block = SpatialFusionBlock(s_block, embed_dims)

        self.conv = nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1)
        if activation is not None:
            self.act = build_activation(activation)
        else:
            self.act = None

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        x = self.sf_block(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SpatialAFRGroup(nn.Module):
    """Anti-degradation Feature Restoration Module (spatial only)"""

    def __init__(
        self,
        embed_dims: int = 256,
        affine: bool = False,
    ):
        super().__init__()
        # self.IN = nn.InstanceNorm2d(embed_dims, affine=affine)

        dw_channel = embed_dims * 4
        groups = 16

        # in
        self.expansion = nn.Conv2d(embed_dims, dw_channel, 1)

        # intra group
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, dw_channel),
            nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=groups),
            nn.GELU(),
        )
        self.intra_group_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, dw_channel, 1, groups=groups),
        )

        # inter group
        self.to_gw = Rearrange("b (g k) h w -> b g k h w", g=groups)
        self.inter_group_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, groups, 1),
            Rearrange("b g h w -> b g 1 h w"),
        )
        self.inverse_gw = Rearrange("b g k h w -> b (g k) h w")

        # out
        self.shrinkage = nn.Conv2d(dw_channel, embed_dims, 1)

        # self.conv = nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=3, padding=1)
        # if activation is not None:
        #     self.act = build_activation(activation)
        # else:
        #     self.act = None

    def forward(self, x: torch.Tensor):
        skip = x

        # x = self.IN(x)
        x = self.expansion(x)

        x = self.conv(x)
        x = x * self.intra_group_attn(x)

        iga = self.inter_group_attn(x)
        x = self.inverse_gw(self.to_gw(x) * iga)

        x = self.shrinkage(x)
        return skip + x


class SpatialAFRGroupRefined(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        affine: bool = False,
    ):
        super().__init__()
        self.IN = nn.InstanceNorm2d(embed_dims, affine=affine)

        num_groups = embed_dims
        group_channels = embed_dims * 2
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups, group_channels),
            nn.Conv2d(group_channels, group_channels, 3, padding=1, groups=num_groups),
            nn.GELU(),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(group_channels, group_channels, 1),
        )

        self.to_gw = Rearrange("b (g k) h w -> b g k h w", g=num_groups)
        self.inter_group_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(group_channels, num_groups, 1),
            Rearrange("b g h w -> b g 1 h w"),
        )
        self.inverse_gw = Rearrange("b g k h w -> b (g k) h w")

        self.projection = nn.Conv2d(group_channels, embed_dims, 1)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        skip = x

        # normalize feature maps to remove style information (degradations + other styles)
        x = self.IN(x)

        # make normalized and original feature map as one group by interleaving
        # [B, 2, C, H, W] -> [B, C, 2, H, W] -> [B, C*2, H, W]
        x = torch.stack([skip, x], dim=1).transpose(1, 2).reshape(B, -1, H, W)

        # refine feature map between normalized and original
        x = self.conv(x)
        x = x * self.attn(x)

        # refine feature map among batch samples
        iga = self.inter_group_attn(x)
        x = self.inverse_gw(self.to_gw(x) * iga)

        x = self.projection(x)
        return skip + x
