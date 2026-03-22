import typing
from collections import defaultdict
from typing import List, Union

import tabulate
import torch.nn as nn


def convert_to_frozen_batchnorm_2d(module: nn.Module):
    from ..models import FrozenBatchNorm2d, FrozenSyncBatchNorm

    module = FrozenBatchNorm2d.convert_to_frozen_batchnorm(module)
    module = FrozenSyncBatchNorm.convert_to_frozen_batchnorm(module)
    return module


def convert_to_batchnorm_2d(module: nn.Module):
    from ..models import FrozenBatchNorm2d, FrozenSyncBatchNorm

    module = FrozenBatchNorm2d.convert_to_batchnorm(module)
    module = FrozenSyncBatchNorm.convert_to_batchnorm(module)
    return module


def build_activation(name: str, **kwargs) -> nn.Module:
    kwargs_ = {"inplace": True, **kwargs}
    act = getattr(nn, name)
    try:
        return act(**kwargs_)
    except Exception:
        return act(**kwargs)


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


def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.

    Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    """
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r


def parameter_count_table(model: nn.Module, max_depth: int = 3) -> str:
    """
    Format the parameter count of the model (and its submodules or parameters)
    in a nice table. It looks like this:

    ::

        | name                            | #elements or shape   |
        |:--------------------------------|:---------------------|
        | model                           | 37.9M                |
        |  backbone                       |  31.5M               |
        |   backbone.fpn_lateral3         |   0.1M               |
        |    backbone.fpn_lateral3.weight |    (256, 512, 1, 1)  |
        |    backbone.fpn_lateral3.bias   |    (256,)            |
        |   backbone.fpn_output3          |   0.6M               |
        |    backbone.fpn_output3.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output3.bias    |    (256,)            |
        |   backbone.fpn_lateral4         |   0.3M               |
        |    backbone.fpn_lateral4.weight |    (256, 1024, 1, 1) |
        |    backbone.fpn_lateral4.bias   |    (256,)            |
        |   backbone.fpn_output4          |   0.6M               |
        |    backbone.fpn_output4.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output4.bias    |    (256,)            |
        |   backbone.fpn_lateral5         |   0.5M               |
        |    backbone.fpn_lateral5.weight |    (256, 2048, 1, 1) |
        |    backbone.fpn_lateral5.bias   |    (256,)            |
        |   backbone.fpn_output5          |   0.6M               |
        |    backbone.fpn_output5.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output5.bias    |    (256,)            |
        |   backbone.top_block            |   5.3M               |
        |    backbone.top_block.p6        |    4.7M              |
        |    backbone.top_block.p7        |    0.6M              |
        |   backbone.bottom_up            |   23.5M              |
        |    backbone.bottom_up.stem      |    9.4K              |
        |    backbone.bottom_up.res2      |    0.2M              |
        |    backbone.bottom_up.res3      |    1.2M              |
        |    backbone.bottom_up.res4      |    7.1M              |
        |    backbone.bottom_up.res5      |    14.9M             |
        |    ......                       |    .....             |

    Args:
        model: a torch module
        max_depth (int): maximum depth to recursively print submodules or
            parameters

    Returns:
        str: the table to be printed

    Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    """
    count: typing.DefaultDict[str, int] = parameter_count(model)
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }

    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        if x > 1e8:
            return "{:.1f}G".format(x / 1e9)
        if x > 1e5:
            return "{:.1f}M".format(x / 1e6)
        if x > 1e2:
            return "{:.1f}K".format(x / 1e3)
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append((indent + name, indent + str(param_shape[name])))
                else:
                    table.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop(""))))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(
        table, headers=["name", "#elements or shape"], tablefmt="pipe"
    )
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab
