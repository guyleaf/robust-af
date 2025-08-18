from typing import List, Union

import torch.nn as nn
from huggingface_hub import repo_exists
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id


def is_huggingface_hub_model(model: str):
    try:
        validate_repo_id(model)
        return repo_exists(model)
    except HFValidationError:
        return False


def unfreeze_modules_and_parameters(
    modules_and_parameters: List[Union[nn.Module, nn.Parameter]],
):
    """Unfreeze specific parameters or modules.

    Args:
        modules_and_parameters (List[Union[nn.Module, nn.Parameter]]): parameters or modules.
    """
    for i, p in enumerate(modules_and_parameters):
        p.requires_grad_(True)


def freeze_all(model: nn.Module) -> nn.Module:
    # NOTE: if we use model.requires_grad_ directly, the requires_grad_ method in submodules will be ignored.

    # freeze top-level parameters
    for param in model.parameters(False):
        param.requires_grad_(False)

    # freeze all submodules
    for _, module in model.named_children():
        module.requires_grad_(False)
    return model


def update_batch_norm_mode(module: nn.Module, train: bool):
    for m in module.modules():
        if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        m.train(train)
