import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import is_debug_mode


def _check_nan(x):
    if not is_debug_mode():
        return

    logger = logging.getLogger("detectron2")
    if torch.any(torch.isnan(x)):
        logger.error("x has NaNs.", stack_info=True)
    if torch.any(torch.isinf(x)):
        logger.error("x has infs.", stack_info=True)


class AMFG(nn.Module):
    def __init__(self, embed_dims: int = 256, spatial_attention: int = 4):
        super().__init__()
        # 4.630608M
        self.dnc_block_combined = DNCBlock_combined(
            embed_dims, spatial_attention=spatial_attention
        )
        # 0.262656M
        self.fgm_block = FGMBlock(embed_dims * 2)
        # 1.179904M
        self.conv_layer = nn.Conv2d(
            embed_dims * 2, embed_dims, kernel_size=3, padding=1
        )

        # self.downsample_layer = nn.Sequential(
        #     nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
        #     LayerNorm2d(transformer_dim // 4),
        #     nn.GELU(),
        #     nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        # )

    def forward(self, x: torch.Tensor):
        # _check_nan(x)
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        x = self.dnc_block_combined(x)
        # _check_nan(x)
        x = self.fgm_block(x)
        # _check_nan(x)
        x = self.conv_layer(x)
        # _check_nan(x)
        return x


class SpatialAMFG(nn.Module):
    def __init__(self, embed_dims: int = 256, spatial_attention: int = 4):
        super().__init__()
        self.dnc_block_combined = DNCBlock_combined(
            embed_dims, spatial_attention=spatial_attention
        )
        self.conv_layer = nn.Conv2d(
            embed_dims * 2, embed_dims, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        x = self.dnc_block_combined(x)
        x = self.conv_layer(x)
        return x


class FrequencyAMFG(nn.Module):
    def __init__(self, embed_dims: int = 256):
        super().__init__()
        self.fgm_block = FGMBlock(embed_dims)

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w] -> [b, c, h, w]
        x = self.fgm_block(x)
        return x


# 4.630608M
class DNCBlock_combined(nn.Module):
    def __init__(self, embed_dims: int, spatial_attention: int = 4):
        super().__init__()
        # 4.33408M
        self.SEMBlock = SKDown(
            3,
            1,
            False,
            16,
            embed_dims,
            embed_dims,
            spatial_attention,
            first=False,
        )
        # concat along channel dimension
        # 0.296528M
        self.channel_attention = CABlock(embed_dims * 2)

    def forward(self, x):
        # _check_nan(x)
        x_in = self.SEMBlock(x)
        # _check_nan(x)
        # _check_nan(x_in)
        x_all = torch.cat([x, x_in], dim=1)
        # _check_nan(x_all)
        output = self.channel_attention(x_all)
        # _check_nan(output)
        return output


# 256 * 16 * 256 + (256 * 256 * 16 + 256 * 16) * 2 = 3.15392M
# spatial_attention=1, 256 * 16 + (16 * 256 + 256) * 2 = 12.8K
class Selector(nn.Module):
    def __init__(self, channel, reduction=16, spatial_attention=4):
        super().__init__()
        self.spatial_attention = spatial_attention
        self.in_channel = channel * (self.spatial_attention**2)
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (self.spatial_attention, self.spatial_attention)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.att_conv1 = nn.Linear(self.in_channel // reduction, self.in_channel)
        self.att_conv2 = nn.Linear(self.in_channel // reduction, self.in_channel)

    def forward(self, x):
        b, c, H, W = x.size()

        y = self.avg_pool(x).reshape(b, -1)
        y = self.fc(y)

        att1 = self.att_conv1(y).view(
            b, c, self.spatial_attention, self.spatial_attention
        )
        att2 = self.att_conv2(y).view(
            b, c, self.spatial_attention, self.spatial_attention
        )

        attention = torch.stack((att1, att2))
        attention = F.softmax(attention, dim=0)

        scale_factor = (
            int(H) / self.spatial_attention,
            int(W) / self.spatial_attention,
        )
        att1 = F.interpolate(
            attention[0],
            scale_factor=scale_factor,
            mode="nearest",
        )
        att2 = F.interpolate(
            attention[1],
            scale_factor=scale_factor,
            mode="nearest",
        )

        return att1, att2


# 4.33408M
class SelectiveConv(nn.Module):
    def __init__(
        self,
        kernel_size,
        padding,
        bias,
        reduction,
        in_channels,
        out_channels,
        spatial_attention,
        first=False,
    ):
        super().__init__()
        self.first = first
        # 256 * 256 * 3 * 3 = 589.824K = 0.589824M
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        # 256 * 256 * 3 * 3 = 589.824K = 0.589824M
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # 3.15392M
        self.selector = Selector(
            out_channels, reduction=reduction, spatial_attention=spatial_attention
        )

        self.IN = nn.InstanceNorm2d(in_channels)
        # 512
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        if self.first:
            f_input = x
            s_input = x
        else:
            # f_input = self.BN(x.clone())
            f_input = self.BN(x)
            # _check_nan(f_input)
            f_input = self.relu(f_input)

            # s_input = self.IN(x.clone())
            s_input = self.IN(x)
            # _check_nan(s_input)
            s_input = self.relu(s_input)

        # _check_nan(f_input)
        # _check_nan(s_input)

        out1 = self.conv1(f_input)
        out2 = self.conv2(s_input)

        # _check_nan(out1)
        # _check_nan(out2)

        out = out1 + out2

        # _check_nan(out)

        att1, att2 = self.selector(out)
        # _check_nan(att1)
        # _check_nan(att2)
        out = torch.mul(out1, att1) + torch.mul(out2, att2)

        return out


class SKDown(nn.Module):
    def __init__(
        self,
        kernel_size,
        padding,
        bias,
        reduction,
        in_channels,
        out_channels,
        spatial_attention,
        first=False,
    ):
        super().__init__()
        args = (
            kernel_size,
            padding,
            bias,
            reduction,
            in_channels,
            out_channels,
            spatial_attention,
        )
        self.maxpool_conv = nn.Sequential(SelectiveConv(*args, first=first))

    def forward(self, x):
        return self.maxpool_conv(x)


# 0.296528M
class CABlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = self.squeeze(x).view(batch_size, channels)
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)
        return x * excitation


# 0.262656M
class FGMBlock(nn.Module):
    def __init__(self, embed_dims: int):
        super().__init__()
        self.conv_layer = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

    def forward(self, x):
        # _check_nan(x)
        fft_map = torch.fft.fft2(x, dim=(-2, -1))
        # _check_nan(fft_map)

        # tmp = torch.where(fft_map == 0, 0, fft_map / (fft_map.abs().pow(2)))
        # if torch.isnan(tmp).any():
        #     logger = logging.getLogger("detectron2")
        #     logger.error("NaN occurred!")
        #     logger.error(fft_map)
        #     # if is_main_process():
        #     #     torch.save(fft_map, "fft_map.pt")
        #     #     torch.save(tmp, "tmp.pt")

        magnitude_map = torch.abs(fft_map)
        phase_map = torch.angle(fft_map)
        # _check_nan(phase_map)

        modified_magnitude = self.conv_layer(magnitude_map)

        real_part = modified_magnitude * torch.cos(phase_map)
        imag_part = modified_magnitude * torch.sin(phase_map)
        modified_fft_map = torch.complex(real_part, imag_part)

        reconstructed_x = torch.real(torch.fft.ifft2(modified_fft_map, dim=(-2, -1)))

        return reconstructed_x
