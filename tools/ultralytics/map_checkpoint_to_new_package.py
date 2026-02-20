import argparse
from pathlib import Path

import torch
from rich.progress import track

from robust_af.utils import setup_environment


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    src = Path(args.src)
    target = Path(args.target or src)

    setup_environment()

    for checkpoint in track(src.rglob(args.pattern)):
        content = torch.load(checkpoint)
        out_file = target / checkpoint.relative_to(src)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(content, out_file)
