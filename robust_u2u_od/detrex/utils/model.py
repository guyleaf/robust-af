from typing import List, Union

import torch.nn as nn
from detectron2.layers.batch_norm import FrozenBatchNorm2d


def unfreeze_modules_and_parameters(
    modules_and_parameters: List[Union[nn.Module, nn.Parameter]],
):
    """Unfreeze specific parameters or modules.

    Args:
        modules_and_parameters (List[Union[nn.Module, nn.Parameter]]): parameters or modules.

    Returns:
        List[nn.BatchNorm2d]: list of `nn.BatchNorm2d` (no matter it originates from FrozenBatchNorm2d or nn.BatchNorm2d)
    """
    new_bns = []
    for i, p in enumerate(modules_and_parameters):
        p.requires_grad_(True)
        if isinstance(p, nn.Module):
            p = FrozenBatchNorm2d.convert_frozenbatchnorm2d_to_batchnorm2d(p)

        # NOTE: only consider 2d ver.
        if isinstance(p, nn.BatchNorm2d):
            new_bns.append(p)
    return new_bns


def freeze_all(model: nn.Module) -> nn.Module:
    # freeze top-level parameters
    for param in model.parameters(False):
        param.requires_grad_(False)

    # freeze all submodules
    for _, module in model.named_children():
        module.requires_grad_(False)

    # using FrozenBatchNorm2d can avoid setting eval mode for every mode switch
    model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    return model
