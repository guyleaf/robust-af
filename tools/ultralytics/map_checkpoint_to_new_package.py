import argparse
from pathlib import Path

import torch
import torch.nn as nn
from rich.progress import track


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("src", type=str, help="the source folder to be scanned")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="the target folder to save the mapped checkpoints. in-place operation by default.",
    )
    parser.add_argument(
        "--pattern", type=str, default="*.pt", help="the search pattern for checkpoints"
    )
    parser.add_argument(
        "--old", type=str, default="robust_au_od", help="old package name"
    )
    parser.add_argument("--new", type=str, default="robust_af", help="new package name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    src = Path(args.src)
    target = Path(args.target or src)

    for checkpoint in track(src.rglob(args.pattern)):
        content = torch.load(checkpoint)
        if "model" not in content:
            continue
        model: nn.Module = content["model"]

        old_module_name = type(model).__module__
        new_module_name = old_module_name.replace(args.old, args.new, 1)
        model.__module__ = new_module_name

        out_file = target / checkpoint.relative_to(src)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # out_file.unlink(missing_ok=True)
        torch.save(content, out_file)
