import argparse
import json
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
from rich.console import Console
from rich.progress import track

from robust_af.transforms import DEGRADATION_TRANSFORMS

CONSOLE = Console()


def visualize_image_adapter(dump_dir: Path, out_dir: Path, args: argparse.Namespace):
    degradations: list[str] = args.degradations
    for name in track(degradations, description="Visualizing...", console=CONSOLE):
        out_image_dir = out_dir / name
        out_image_dir.mkdir(parents=True, exist_ok=True)
        degradation_dir = dump_dir / name
        images_file = degradation_dir / "images.pt"
        images: dict[int, dict[str, torch.Tensor]] = torch.load(
            images_file, weights_only=True
        )

        image_ids = list(images.keys())
        for image_id in image_ids[:: args.vis_period]:
            image_name = f"{image_id:05d}.jpg"
            image = images[image_id]

            for k, image in images[image_id].items():
                image_name = f"{image_id:05d}_{k}.jpg"

                # [H, W, 3]
                # value range: [0, 1]
                np_image: np.ndarray = image.permute(1, 2, 0).numpy()
                np_image = (np_image * 255).clip(min=0, max=255).astype(np.uint8)
                image = Image.fromarray(np_image)
                image.save(out_image_dir / image_name)

                if args.show:
                    image.show(title=image_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize images from image adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dump_dir", help="Path of dump directory")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory of t-SNE results. If None, use dump_dir.",
    )
    parser.add_argument(
        "--vis-period",
        type=int,
        default=10,
        help="Visualize image per iterations.",
    )

    parser.add_argument(
        "--degradations",
        type=str,
        nargs="+",
        default=list(DEGRADATION_TRANSFORMS.keys()),
        help="Visualized degradations. If it is not existed in dump_dir, ignore it.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the result in a graphical window."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dump_dir = Path(args.dump_dir)

    # load metadata
    with open(dump_dir / "metadata.json", "r") as f:
        metadata: dict = json.load(f)

    # filter degradations if exists
    target_degradations = set(args.degradations)
    args.degradations = list(
        filter(lambda name: name in target_degradations, metadata["degradations"])
    )

    # determine out_dir
    if args.out_dir is None:
        out_dir = dump_dir
    else:
        out_dir = Path(args.out_dir)
    out_dir = out_dir / f"vis_image_adapter_{len(args.degradations)}"
    args.out_dir = out_dir.as_posix()

    out_dir.mkdir(parents=True, exist_ok=True)
    # save args
    with open(out_dir / "args.json", "w") as f:
        content = vars(args)
        json.dump(content, f, indent=4)

    visualize_image_adapter(dump_dir, out_dir, args)
