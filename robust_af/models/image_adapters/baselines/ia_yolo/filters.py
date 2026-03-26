import math
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Optional, TypeVar

import torch
import torch.nn.functional as F

from .utils import get_tanh, lerp, rgb2lum

T = TypeVar("T")
_range_t = tuple[T, T]


class Filter(metaclass=ABCMeta):
    # number of parameters of the filter
    _num_parameters: int = 0

    def __init__(self):
        assert isinstance(self._num_parameters, int)

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    def refine_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        return parameters

    @abstractmethod
    def process(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        parameters = self.refine_parameters(parameters)
        return self.process(images, parameters)


class UsmFilter(Filter):
    _num_parameters = 1

    def __init__(
        self,
        coef_range: _range_t[float] = (0, 5),
        radius: int = 12,
        sigma: Optional[float] = 5,
    ):
        super().__init__()
        assert sigma is not None, "Currently, only support UsmFilter with fixed sigma."
        self._parameter_mapper = get_tanh(*coef_range)
        self._radius = radius

        # if None, sigma is a learnable parameter, like UsmFilter_sigma in TF implementation.
        if sigma is None:
            self._kernels = None
            self._num_parameters = UsmFilter._num_parameters + 1
        else:
            sigma_tensor = torch.tensor([[sigma]])
            self._kernels = self._make_gaussian_2d_kernels(sigma_tensor, radius)

    def _make_gaussian_2d_kernels(self, sigma: torch.Tensor, radius: int):
        # [K]
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=sigma.device)
        # ([K] -> [1, K]) / [B, 1] -> [B, K]
        kernel = torch.exp(-0.5 * (x[None] / sigma) ** 2)
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        # [B, K, K]
        return torch.einsum("bi,bj->bij", kernel, kernel)

    def refine_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        # TODO: sigma range mapping is not implemented yet.
        # Need to determine a proper range for learnable sigma.
        return self._parameter_mapper(parameters)

    def process(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = images.shape
        if self._kernels is None:
            sigma, parameters = parameters[:, :1], parameters[:, 1:]
            # [B, K, K]
            kernels = self._make_gaussian_2d_kernels(sigma, self._radius)
        else:
            # move to GPU only once
            if self._kernels.device != parameters.device:
                self._kernels = self._kernels.to(parameters.device)
            # [1, K, K]
            kernels = self._kernels

        # make kernel format match with conv2d
        # kernel_i = tf.tile(kernel_i[:, :, tf.newaxis, tf.newaxis], [1, 1, 1, 1])
        # [B, K, K] -> [B, c, 1, K, K]
        kernels = kernels[:, None, None].expand(-1, c, -1, -1, -1)

        # pad_w = (25 - 1) // 2
        # padded = tf.pad(
        #     img, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode="REFLECT"
        # )
        # outputs = []
        # for channel_idx in range(3):
        #     data_c = padded[:, :, :, channel_idx : (channel_idx + 1)]
        #     data_c = tf.nn.conv2d(data_c, kernel_i, [1, 1, 1, 1], "VALID")
        #     outputs.append(data_c)
        # output = tf.concat(outputs, axis=3)

        # apply gaussian kernel
        kernel_size = kernels.size(-1)
        padding = (kernel_size - 1) // 2
        padded_images = F.pad(images, [padding] * 4, mode="reflect")
        if kernels.size(0) == 1:  # batch-wise shortcut
            kernel = kernels[0]
            blurred_images = F.conv2d(padded_images, kernel, stride=1, groups=c)
        else:
            blurred_images = []
            for i, padded_image in enumerate(padded_images):
                kernel = kernels[i]
                padded_image = F.conv2d(padded_image[None], kernel, stride=1, groups=c)
                blurred_images.append(padded_image.squeeze(0))
            blurred_images = torch.stack(blurred_images)

        images = (images - blurred_images) * parameters[..., None, None] + images
        # images = (images - blurred_images) * 2.5 + images
        return images


class DefogFilter(Filter):
    _num_parameters = 1

    def __init__(self, coef_range: _range_t[float] = (0.1, 1.0)):
        super().__init__()
        self._parameter_mapper = get_tanh(*coef_range)

    def _estimate_dark_channel(self, images: torch.Tensor):
        # [b, c, h, w] -> [b, 1, h, w]
        return images.min(1, keepdim=True).values

    def _estimate_atmospheric_light(self, images: torch.Tensor, dark: torch.Tensor):
        b, c, h, w = images.shape
        image_size = h * w

        flat_image = images.view(b, c, -1)
        flat_dark = dark.view(b, 1, -1)

        # NOTE: we don't know why the authors mention picking top 1000 pixels in the paper.
        # but here, they pick 0.1% of pixels.
        # in dark channel prior, they use 0.1%. so, we keep it.
        # numpx = int(max(math.floor(image_size / 1000), 1))
        # indices = darkvec.argsort(0)
        # indices = indices[(imsz - numpx) : imsz]

        # select top-k pixels (0.1%)
        num_pixels = int(max(image_size // 1000, 1))
        # [b, 1, num_pixels]
        _, indices = flat_dark.topk(num_pixels)
        indices = indices.expand(-1, c, -1)

        # NOTE: in TF implementation, it gather pixels from 1 to k-1.
        # we think it is a bug because it is divided by numpx.
        # so, we fix it here.
        # atmsum = np.zeros([1, 3])
        # for ind in range(1, numpx):
        #     atmsum = atmsum + imvec[indices[ind]]
        # A = atmsum / numpx

        selected_pixels = flat_image.gather(2, indices)
        # [b, c, num_pixels] -> [b, c, 1] -> [b, c, 1, 1]
        return selected_pixels.mean(2, keepdim=True)[..., None]

    def _estimate_dark_ica(self, images: torch.Tensor, A: torch.Tensor):
        # im3 = np.empty(im.shape, im.dtype)
        # for ind in range(0, 3):
        #     im3[:, :, ind] = im[:, :, ind] / A[0, ind]
        images = images / A.clamp(min=1e-6)
        # [b, c, h, w] -> [b, 1, h, w]
        return self._estimate_dark_channel(images)

    def refine_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        return self._parameter_mapper(parameters)

    def process(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        dark = self._estimate_dark_channel(images)
        A = self._estimate_atmospheric_light(images, dark)
        IcA = self._estimate_dark_ica(images, A)

        trans_map = 1 - parameters[..., None, None] * IcA
        # trans_map = 1 - 0.5 * IcA
        trans_map = trans_map.clamp(min=1e-2)
        return (images - A) / trans_map + A


class GammaFilter(Filter):
    _num_parameters = 1

    def __init__(self, coef_range: float = 3):
        super().__init__()
        log_coef_range = math.log(coef_range)
        self._parameter_mapper = get_tanh(-log_coef_range, log_coef_range)

    def refine_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._parameter_mapper(parameters))

    def process(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        images = images.clamp(min=1e-4)
        return torch.pow(images, parameters[..., None, None])


class ImprovedWhiteBalanceFilter(Filter):
    _num_parameters = 3

    # e^0.5 = 1.6487212707001282
    # log_wb_range = 0.5 => wb_range = 1.6487212707001282
    def __init__(self, coef_range: float = 1.6487212707001282):
        super().__init__()
        log_coef_range = math.log(coef_range)
        self._parameter_mapper = get_tanh(-log_coef_range, log_coef_range)

        self._mask = torch.tensor([[0, 1, 1]], dtype=torch.float32)
        # self._mask = torch.tensor([[1, 0, 1]], dtype=torch.float32)

    def refine_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        if self._mask.device != parameters.device:
            self._mask = self._mask.to(parameters.device)

        parameters = parameters * self._mask
        color_scaling = torch.exp(self._parameter_mapper(parameters))
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        luminance = (
            0.27 * color_scaling[:, 0]
            + 0.67 * color_scaling[:, 1]
            + 0.06 * color_scaling[:, 2]
        )
        luminance = luminance.clamp(min=1e-5)
        return color_scaling / luminance[:, None]

    def process(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return images * parameters[..., None, None]


class ToneFilter(Filter):
    def __init__(self, coef_range: _range_t[float] = (0.5, 2), curve_steps: int = 8):
        self._num_parameters = self._curve_steps = curve_steps
        super().__init__()
        self._parameter_mapper = get_tanh(*coef_range)

    def refine_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        return self._parameter_mapper(parameters)

    def process(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        # NOTE: scale images to [0, curve_steps] instead of multiple divisions
        scaled_images = images * self._curve_steps
        total_images = torch.zeros_like(scaled_images)
        for i in range(self._curve_steps):
            # value range: [0, 1] * tone_curve -> [0, tone_curve]
            total_images = total_images + (
                torch.clamp(scaled_images - i, min=0, max=1)
                * parameters[:, i, None, None, None]
            )

        # value range: [0, tone_curve] -> normalized by sum of tone_curve -> [0, 1]
        # [b, curve_steps] -> [b, 1]
        tone_curve_sum = parameters.sum(dim=1, keepdim=True).clamp(min=1e-30)
        # [b, c, h, w] / [b, 1, 1, 1] -> [b, c, h, w]
        return total_images / tone_curve_sum[..., None, None]


class ContrastFilter(Filter):
    _num_parameters = 1

    def __init__(self):
        super().__init__()

    def refine_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        # return torch.sigmoid(parameters)
        return torch.tanh(parameters)

    def process(self, images: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        luminance = torch.clamp(rgb2lum(images), min=0, max=1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = images / luminance.clamp(min=1e-6) * contrast_lum
        return lerp(images, contrast_image, parameters[..., None, None])
        # return lerp(images, contrast_image, 0.5)


# NOTICE: order matter
# The insertion order follows the config of TF implementation
FILTERS = OrderedDict(
    defog=DefogFilter,
    wb=ImprovedWhiteBalanceFilter,
    gamma=GammaFilter,
    tone=ToneFilter,
    contrast=ContrastFilter,
    usm=UsmFilter,
)
