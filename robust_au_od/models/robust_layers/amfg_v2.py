import torch
import torch.nn as nn
import torch.nn.functional as F


class AMFG(nn.Module):
    def __init__(
        self, embed_dims: int = 256, spatial_attention: int = 4, selector: bool = False
    ):
        super().__init__()
        self.dnc_block_combined = DNCBlock_combined(
            embed_dims, spatial_attention=spatial_attention, selector=selector
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
        # x: [b, c, h, w] -> [b, 2*c, h, w]
        x = self.dnc_block_combined(x)
        x = self.fgm_block(x)
        x = self.conv_layer(x)
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


class DNCBlock_combined(nn.Module):
    def __init__(
        self, embed_dims: int, spatial_attention: int = 4, selector: bool = False
    ):
        super().__init__()
        self.SEMBlock = SKDown(
            3,
            1,
            False,
            16,
            embed_dims,
            embed_dims,
            spatial_attention,
            first=False,
            selector=selector,
        )
        # concat along channel dimension
        # 0.296528M
        self.channel_attention = CABlock(embed_dims * 2)

    def forward(self, x):
        x_in = self.SEMBlock(x)
        x_all = torch.cat([x, x_in], dim=1)
        output = self.channel_attention(x_all)
        return output


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
        self.att_conv = nn.Linear(self.in_channel // reduction, self.in_channel)

    def forward(self, x):
        b, c, H, W = x.size()

        y = self.avg_pool(x).reshape(b, -1)
        y = self.fc(y)

        attention = self.att_conv(y).view(
            b, c, self.spatial_attention, self.spatial_attention
        )
        attention = torch.sigmoid(attention)

        scale_factor = (
            int(H) / self.spatial_attention,
            int(W) / self.spatial_attention,
        )
        att = F.interpolate(
            attention,
            scale_factor=scale_factor,
            mode="nearest",
        )

        return att


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
        selector=False,
    ):
        super().__init__()
        self.first = first
        # 256 * 256 * 3 * 3 = 589.824K = 0.589824M
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # 3.15392M
        if selector:
            self.selector = Selector(
                out_channels, reduction=reduction, spatial_attention=spatial_attention
            )
        else:
            self.selector = None

        self.IN = nn.InstanceNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        if self.first:
            s_input = x
        else:
            # s_input = self.IN(x.clone())
            s_input = self.IN(x)
            s_input = self.relu(s_input)

        out = self.conv(s_input)
        if self.selector is not None:
            att = self.selector(out)
            out = torch.mul(out, att)

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
        selector=False,
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
        self.maxpool_conv = nn.Sequential(
            SelectiveConv(*args, first=first, selector=selector)
        )

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
        fft_map = torch.fft.fft2(x, dim=(-2, -1))

        magnitude_map = torch.abs(fft_map)
        phase_map = torch.angle(fft_map)

        modified_magnitude = self.conv_layer(magnitude_map)

        real_part = modified_magnitude * torch.cos(phase_map)
        imag_part = modified_magnitude * torch.sin(phase_map)
        modified_fft_map = torch.complex(real_part, imag_part)

        reconstructed_x = torch.real(torch.fft.ifft2(modified_fft_map, dim=(-2, -1)))

        return reconstructed_x
