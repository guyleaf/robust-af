from typing import List, Union

import torch.nn as nn

from ..models import FrozenBatchNorm2d, FrozenSyncBatchNorm


def convert_to_frozen_batchnorm_2d(module: nn.Module):
    module = FrozenBatchNorm2d.convert_to_frozen_batchnorm(module)
    module = FrozenSyncBatchNorm.convert_to_frozen_batchnorm(module)
    return module


def convert_to_batchnorm_2d(module: nn.Module):
    module = FrozenBatchNorm2d.convert_to_batchnorm(module)
    module = FrozenSyncBatchNorm.convert_to_batchnorm(module)
    return module


def unfreeze_modules_and_parameters(
    modules_and_parameters: List[Union[nn.Module, nn.Parameter]],
) -> List[Union[nn.Module, nn.Parameter]]:
    """Unfreeze specific parameters or modules.

    Args:
        modules_and_parameters (List[Union[nn.Module, nn.Parameter]]): parameters or modules.
    """
    new_instances = []
    for i, p in enumerate(modules_and_parameters):
        if isinstance(p, nn.Module):
            p = convert_to_batchnorm_2d(p)
        p.requires_grad_(True)
        new_instances.append(p)
    return new_instances


def freeze_all(model: nn.Module) -> nn.Module:
    # NOTE: if we use model.requires_grad_ directly, the requires_grad_ method in submodules will be ignored.

    # freeze top-level parameters
    for param in model.parameters(False):
        param.requires_grad_(False)

    # freeze all submodules
    for _, module in model.named_children():
        module.requires_grad_(False)

    model = convert_to_frozen_batchnorm_2d(model)
    return model
