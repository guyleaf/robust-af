import argparse

from rich import print
from rtdetrv2.core import YAMLConfig

import robust_au_od.rtdetrv2  # noqa: F401
from robust_au_od.utils.misc import format_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    cfg = YAMLConfig(args.config_file)
    print(cfg.model)

    trainable_modules = []
    for name, module in cfg.model.named_modules():
        if name.count(".") <= 3 and any(
            param.requires_grad for param in module.parameters()
        ):
            trainable_modules.append(name)

    # calculate number of parameters
    total_parameters = 0
    total_trainable_parameters = 0
    trainable_parameters = []
    for name, param in cfg.model.named_parameters():
        size = param.numel()
        total_parameters += size
        if param.requires_grad:
            total_trainable_parameters += size
            trainable_parameters.append(name)

    print("\nTrainable (partial) modules:")
    print(trainable_modules)

    print("\nTrainable Parameters:")
    print(trainable_parameters)

    print(
        f"\nTotal parameters (trainable/total): {format_size(total_trainable_parameters)}/{format_size(total_parameters)}"
    )
