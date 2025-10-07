import argparse

from detectron2.config import LazyConfig, instantiate
from rich import print

from robust_u2u_od.utils.misc import format_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    cfg = LazyConfig.load(args.config_file)
    print(cfg.model)
    model = instantiate(cfg.model)
    # print(model)

    trainable_modules = []
    for name, module in model.named_modules():
        if name.count(".") <= 3 and any(
            param.requires_grad for param in module.parameters()
        ):
            trainable_modules.append(name)

    # calculate number of parameters
    total_parameters = 0
    total_trainable_parameters = 0
    trainable_parameters = []
    for name, param in model.named_parameters():
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
