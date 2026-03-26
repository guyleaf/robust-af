import math
from abc import ABCMeta
from typing import Optional, TypeVar, Union

import torch
import torch.nn as nn
import torchvision.transforms as T

from .utils import lerp, min_max_normalize, rgb2lum, tanhlr

V = TypeVar("V")
_range_t = tuple[V, V]


class GatedOp(nn.Module, metaclass=ABCMeta):
    # number of parameters of the filter
    _num_parameters: int = 0

    def __init__(self):
        super().__init__()
        assert isinstance(self._num_parameters, int)

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    def process(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return images

    def forward(
        self,
        images: torch.Tensor,
        latents: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        images = self.process(images, latents)
        if gate is not None:
            images = images * gate[:, None, None, None]
        return images


class WhiteBalanceOp(GatedOp):
    _num_parameters = 3

    # e^0.5 = 1.6487212707001282
    # log_wb_range = 0.5 => wb_range = 1.6487212707001282
    def __init__(self, embed_dims: int = 256, param_range: float = 1.6487212707001282):
        super().__init__()
        self._log_param_range = math.log(param_range)
        self.param_net = nn.Sequential(
            nn.Linear(embed_dims, self._num_parameters, bias=True)
        )

    def process(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # wb = self.wb_module(latent_out)
        # wb = torch.exp(self.tanhlr(wb, -log_wb_range, log_wb_range))

        # [b, 3]
        wb = self.param_net(latents)
        wb = torch.exp(tanhlr(wb, -self._log_param_range, self._log_param_range))

        # color_scaling = 1.0 / (
        #     1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] + 0.06 * wb[:, 2]
        # )
        # wb = color_scaling.unsqueeze(1) * wb

        luminance = 0.27 * wb[:, 0] + 0.67 * wb[:, 1] + 0.06 * wb[:, 2]
        luminance = luminance.clamp(min=1e-5)
        wb = wb / luminance[:, None]

        # wb_out = wb.unsqueeze(2).unsqueeze(3) * x
        # wb_out = (wb_out - wb_out.min()) / (wb_out.max() - wb_out.min())

        images = images * wb[..., None, None]
        images = min_max_normalize(images)
        return images


class GammaOp(GatedOp):
    _num_parameters = 1

    def __init__(self, embed_dims: int = 256, param_range: float = 2.5):
        super().__init__()
        self._log_param_range = math.log(param_range)
        self.param_net = nn.Sequential(
            nn.Linear(embed_dims, self._num_parameters, bias=True)
        )

    def process(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # gamma = self.gamma_module(latent_out).unsqueeze(2).unsqueeze(3)
        # gamma = torch.exp(self.tanhlr(gamma, -log_gamma, log_gamma))

        # [b, 1]
        gamma = self.param_net(latents)
        gamma = torch.exp(tanhlr(gamma, -self._log_param_range, self._log_param_range))

        # g = torch.pow(torch.maximum(x, torch.Tensor(1e-4)), gamma)
        # g = (g - g.min()) / (g.max() - g.min())

        images = images.clamp(min=1e-4)
        images = torch.pow(images, gamma[..., None, None])
        images = min_max_normalize(images)
        return images


class SharpOp(GatedOp):
    _num_parameters = 1

    def __init__(
        self,
        embed_dims: int = 256,
        param_range: _range_t[float] = (0.1, 1.0),
        kernel_size: int = 13,
        sigma: _range_t[float] = (0.1, 5.0),
    ):
        super().__init__()
        self._param_range = param_range
        self.gaussian_blur = T.GaussianBlur(kernel_size, sigma=sigma)
        self.param_net = nn.Sequential(
            nn.Linear(embed_dims, self._num_parameters, bias=True)
        )

    def process(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # y = self.sharpning_module(latent_out).unsqueeze(2).unsqueeze(3)
        # y = self.tanhlr(y, torch.Tensor(0.1), torch.Tensor(1.0))

        # [b, 1]
        w = self.param_net(latents)
        w = tanhlr(w, *self._param_range)

        # out_x = self.blur(x)
        # s = x + (y * (x - out_x))
        # s = (s - s.min()) / (s.max() - s.min())

        blurred_images = self.gaussian_blur(images)
        images = (images - blurred_images) * w[..., None, None] + images
        images = min_max_normalize(images)
        return images


class DefogOp(GatedOp):
    _num_parameters = 1

    def __init__(
        self, embed_dims: int = 256, param_range: _range_t[float] = (0.1, 1.0)
    ):
        super().__init__()
        self._param_range = param_range
        self.param_net = nn.Sequential(
            nn.Linear(embed_dims, self._num_parameters, bias=True)
        )

    def _estimate_dark_channel(self, images: torch.Tensor):
        # [b, c, h, w] -> [b, 1, h, w]
        return images.min(1, keepdim=True).values

    def _estimate_atmospheric_light(
        self, images: torch.Tensor, dark: torch.Tensor, top_k: int = 1000
    ):
        b, c, h, w = images.shape
        image_size = h * w

        flat_image = images.view(b, c, -1)
        flat_dark = dark.view(b, 1, -1)

        # numpx = int(max(math.floor(imsz / top_k), 1))
        # darkvec = dark.reshape(x.shape[0], imsz, 1)
        # imvec = x.reshape(x.shape[0], 3, imsz).transpose(1, 2)
        # indices = darkvec.argsort(1)
        # indices = indices[:, imsz - numpx : imsz]

        # select top-k pixels (0.1%)
        num_pixels = int(max(image_size // top_k, 1))
        # [b, 1, num_pixels]
        _, indices = flat_dark.topk(num_pixels)
        indices = indices.expand(-1, c, -1)

        # NOTE: in official implementation, it gather pixels from 1 to k-1.
        # we think it is a bug because it is divided by numpx.
        # so, we fix it here.
        # atmsum = torch.zeros([x.shape[0], 1, 3]).cuda()
        # for b in range(x.shape[0]):
        #     for ind in range(1, numpx):
        #         atmsum[b, :, :] = atmsum[b, :, :] + imvec[b, indices[b, ind], :]
        # a = atmsum / numpx
        # a = a.squeeze(1).unsqueeze(2).unsqueeze(3)

        selected_pixels = flat_image.gather(2, indices)
        # [b, c, num_pixels] -> [b, c, 1] -> [b, c, 1, 1]
        return selected_pixels.mean(2, keepdim=True)[..., None]

    def _estimate_dark_ica(self, images: torch.Tensor, A: torch.Tensor):
        # i = x / a
        # i = self.dark_channel(i)
        images = images / A.clamp(min=1e-6)
        # [b, c, h, w] -> [b, 1, h, w]
        return self._estimate_dark_channel(images)

    def process(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # omega = self.defogging_module(latent_out).unsqueeze(2).unsqueeze(3)
        # omega = self.tanhlr(omega, torch.Tensor(0.1), torch.Tensor(1.0))

        # [b, 1]
        omega = self.param_net(latents)
        omega = tanhlr(omega, *self._param_range)

        dark = self._estimate_dark_channel(images)
        A = self._estimate_atmospheric_light(images, dark)
        IcA = self._estimate_dark_ica(images, A)

        # t = 1.0 - (omega * i)
        # j = ((x - a) / (torch.maximum(t, torch.Tensor(0.01)))) + a
        # j = (j - j.min()) / (j.max() - j.min())

        trans_map = 1 - omega[..., None, None] * IcA
        trans_map = trans_map.clamp(min=1e-2)
        images = (images - A) / trans_map + A
        images = min_max_normalize(images)
        return images


class ContrastOp(GatedOp):
    _num_parameters = 1

    def __init__(self, embed_dims: int = 256):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(embed_dims, self._num_parameters, bias=True)
        )

    def process(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # alpha = torch.tanh(self.contrast_module(latent_out))
        # [b, 1]
        alpha = torch.tanh(self.param_net(latents))

        # luminance = torch.minimum(
        #     torch.maximum(self.rgb2lum(x), torch.Tensor(0.0)), torch.Tensor(1.0)
        # ).unsqueeze(1)
        # contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        # contrast_image = x / (luminance + 1e-6) * contrast_lum
        # contrast_image = self.lerp(x, contrast_image, alpha)
        # contrast_image = (contrast_image - contrast_image.min()) / (
        #     contrast_image.max() - contrast_image.min()
        # )

        luminance = torch.clamp(rgb2lum(images), min=0, max=1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = images / luminance.clamp(min=1e-6) * contrast_lum
        images = lerp(images, contrast_image, alpha[..., None, None])
        images = min_max_normalize(images)
        return images


class ToneOp(GatedOp):
    def __init__(
        self,
        embed_dims: int = 256,
        param_range: _range_t[float] = (0.5, 2.0),
        curve_steps: int = 8,
    ):
        self._num_parameters = self._curve_steps = curve_steps
        super().__init__()
        self._param_range = param_range
        self.param_net = nn.Sequential(
            nn.Linear(embed_dims, self._num_parameters, bias=True)
        )

    def process(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # tone_curve = self.tone_module(latent_out).reshape(-1, 1, curve_steps)
        # tone_curve = self.tanhlr(tone_curve, 0.5, 2)

        # [b, curve_steps]
        tone_curve = self.param_net(latents)
        tone_curve = tanhlr(tone_curve, *self._param_range)

        # NOTE: scale images to [0, curve_steps] instead of multiple divisions
        # total_image = x * 0
        # for i in range(curve_steps):
        #     total_image += torch.clamp(
        #         x - 1.0 * i / curve_steps, 0, 1.0 / curve_steps
        #     ) * tone_curve[:, :, i].unsqueeze(2).unsqueeze(3)

        scaled_images = images * self._curve_steps
        total_images = torch.zeros_like(scaled_images)
        for i in range(self._curve_steps):
            # value range: [0, 1] * tone_curve -> [0, tone_curve]
            total_images = total_images + (
                torch.clamp(scaled_images - i, min=0, max=1)
                * tone_curve[:, i, None, None, None]
            )

        # tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30
        # total_image *= curve_steps / tone_curve_sum.unsqueeze(2).unsqueeze(3)
        # total_image = (total_image - total_image.min()) / (
        #     total_image.max() - total_image.min()
        # )

        # value range: [0, tone_curve] -> normalized by sum of tone_curve -> [0, 1]
        # [b, curve_steps] -> [b, 1]
        tone_curve_sum = tone_curve.sum(dim=1, keepdim=True).clamp(min=1e-30)
        # [b, c, h, w] / [b, 1, 1, 1] -> [b, c, h, w]
        images = total_images / tone_curve_sum[..., None, None]
        images = min_max_normalize(images)
        return images


IdentityOp = GatedOp

GDIP_OPS = dict(
    wb=WhiteBalanceOp,
    gamma=GammaOp,
    identity=IdentityOp,
    sharp=SharpOp,
    defog=DefogOp,
    contrast=ContrastOp,
    tone=ToneOp,
)


class GatedDIP(GatedOp):
    def __init__(
        self,
        embed_dims: int = 256,
        param_range: _range_t[float] = (1e-2, 1.0),
        op_cfgs: dict = {},
        ignored_ops: Union[list[str], set[str]] = [],
    ):
        super().__init__()
        self._param_range = param_range
        self._num_parameters = 0

        # build ops
        ops = nn.ModuleList()
        ignored_ops = set(ignored_ops)
        for name, op_cls in GDIP_OPS.items():
            if name in ignored_ops:
                continue

            cfg = op_cfgs.get(name, {})
            if op_cls is IdentityOp:
                instance = op_cls(**cfg)
            else:
                instance = op_cls(embed_dims=embed_dims, **cfg)

            ops.append(instance)
            self._num_parameters += instance.num_parameters

        self.ops = ops
        num_ops = len(self.ops)
        self.gate = nn.Sequential(nn.Linear(embed_dims, num_ops, bias=True))

    def process(self, images: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # [b, num_ops]
        gates = tanhlr(self.gate(latents), *self._param_range)

        out_images = torch.zeros_like(images)
        for i, op in enumerate(self.ops):
            out_images = out_images + op(images, latents, gates[:, i])

        out_images = min_max_normalize(out_images)
        return out_images
