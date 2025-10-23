from typing import Any, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FrozenBatchNorm(nn.modules.batchnorm._NormBase):
    """
    Frozen version of nn.modules.batchnorm._BatchNorm

    Some implementations will check if it is a instance of _BatchNorm and convert it to other BatchNorm, e.g. SyncBatchNorm.
    So, in order not to be replaced by them, we implement a froze version without training part.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.requires_grad_(False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in eval mode when buffers are None.
        """
        bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class FrozenBatchNormConverterMixin:
    __target__: Type[nn.modules.batchnorm._BatchNorm] = None
    __arguments__: list[str] = ["eps", "momentum", "affine", "track_running_stats"]

    @classmethod
    def make_arguments(cls, module: nn.Module):
        kwargs = {k: getattr(module, k) for k in cls.__arguments__}
        return kwargs

    @classmethod
    def convert_to_frozen_batchnorm(cls, module: nn.Module):
        """
        Convert all `__target__` in module into frozen version.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is `__target__`, returns a new module.
            Otherwise, in-place convert module and return it.

        Modified from https://github.com/facebookresearch/detectron2/blob/a1ce2f956a1d2212ad672e3c47d53405c2fe4312/detectron2/layers/batch_norm.py#L102
        """
        assert cls.__target__ is not None and issubclass(
            cls.__target__, nn.modules.batchnorm._BatchNorm
        )
        res = module
        if isinstance(module, cls.__target__):
            kwargs = cls.make_arguments(module)
            res = cls(module.num_features, **kwargs)
            res.load_state_dict(module.state_dict())
        else:
            for name, child in module.named_children():
                new_child = cls.convert_to_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    @classmethod
    def convert_to_batchnorm(cls, module: nn.Module) -> nn.Module:
        """
        Convert all frozen version of `__target__` in module into `__target__`.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is frozen version of `__target__`, returns a new module.
            Otherwise, in-place convert module and return it.

        Modified from https://github.com/facebookresearch/detectron2/blob/a1ce2f956a1d2212ad672e3c47d53405c2fe4312/detectron2/layers/batch_norm.py#L136
        """
        assert cls.__target__ is not None and issubclass(
            cls.__target__, nn.modules.batchnorm._BatchNorm
        )
        res = module
        if isinstance(module, cls):
            kwargs = cls.make_arguments(module)
            res = cls.__target__(module.num_features, **kwargs)
            res.load_state_dict(module.state_dict())
        else:
            for name, child in module.named_children():
                new_child = cls.convert_to_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class FrozenBatchNorm2d(_FrozenBatchNorm, FrozenBatchNormConverterMixin):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Differences from common implementation:
    1. keep all settings to support converting back
    2. follow track_running_stats to use batch statistics or running states instead of acting like an identity

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """

    __target__ = nn.BatchNorm2d

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class FrozenSyncBatchNorm(_FrozenBatchNorm, FrozenBatchNormConverterMixin):
    """
    SyncBatchNorm where the batch statistics and the affine parameters are fixed

    Differences from common implementation:
    1. keep all settings to support converting back
    2. follow track_running_stats to use batch statistics or running states instead of acting like an identity

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world
    """

    __target__ = nn.SyncBatchNorm
    __arguments__: list[str] = [
        "eps",
        "momentum",
        "affine",
        "track_running_stats",
        "process_group",
    ]

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Optional[Any] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype,
            **factory_kwargs,
        )
        self.process_group = process_group

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() < 2:
            raise ValueError(f"expected at least 2D input (got {input.dim()}D input)")


if __name__ == "__main__":
    assert not issubclass(
        FrozenBatchNorm2d, (nn.BatchNorm2d, nn.modules.batchnorm._BatchNorm)
    )
    instance = FrozenBatchNorm2d(100)
    assert isinstance(instance, (FrozenBatchNorm2d, _FrozenBatchNorm))
    assert not isinstance(instance, (nn.BatchNorm2d, nn.modules.batchnorm._BatchNorm))
