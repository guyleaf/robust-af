#!/usr/bin/env python3
"""Browse and visualize a COCO dataset with bounding boxes using matplotlib."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from rich.progress import track

from robust_af.transforms import DEGRADATION_TRANSFORMS, apply_degradation
from robust_af.utils.random import seed_everything

_DEGRADATION_LIST = [d for d in DEGRADATION_TRANSFORMS if d != "identity"]


def get_color_map(num_classes: int) -> list[tuple[float, float, float]]:
    rng = random.Random(42)
    return [(rng.random(), rng.random(), rng.random()) for _ in range(num_classes)]


def visualize_sample(
    coco: COCO,
    img_id: int,
    degradation: str,
    image_dir: Path,
    color_map: list[tuple[float, float, float]],
    show_labels: bool = True,
    ax: plt.Axes | None = None,
) -> None:
    img_info = coco.imgs[img_id]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    img_path = image_dir / img_info["file_name"]
    image = np.array(Image.open(img_path).convert("RGB"))

    image = apply_degradation(degradation, image)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.imshow(image)
    ax.set_title(degradation, fontsize=90)
    ax.axis("off")

    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(coco.dataset["categories"])}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.dataset["categories"]}

    for ann in anns:
        x, y, w, h = ann["bbox"]
        cat_id = ann["category_id"]
        color = color_map[cat_id_to_idx.get(cat_id, 0)]

        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        if show_labels:
            label = cat_id_to_name.get(cat_id, str(cat_id))
            ax.text(
                x,
                y - 4,
                label,
                fontsize=8,
                color="white",
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
                clip_on=True,
            )


def browse(
    image_dir: str | Path,
    annotation_file: str | Path,
    degradations: list[str],
    num_images: int = 3,
    image_ids: list[int] | None = None,
    shuffle: bool = False,
    seed: int | None = 2026,
    show_labels: bool = True,
    out_dir: str | Path | None = "results",
    show: bool = False,
) -> None:
    seed_everything(seed)
    image_dir = Path(image_dir)
    coco = COCO(annotation_file)

    all_ids = list(coco.imgs.keys())
    if image_ids:
        ids = [i for i in image_ids if i in coco.imgs]
    elif shuffle:
        ids = random.sample(all_ids, min(num_images, len(all_ids)))
    else:
        ids = all_ids[:num_images]

    num_classes = len(coco.dataset.get("categories", []))
    color_map = get_color_map(max(num_classes, 1))

    cols = min(5, len(degradations))
    rows = (len(degradations) + cols - 1) // cols
    for img_id in track(ids):
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(13 * cols, 9 * rows),
            constrained_layout=True,
        )
        axes = np.array(axes).flatten() if len(degradations) > 1 else [axes]

        for i, degradation in enumerate(degradations):
            visualize_sample(
                coco, img_id, degradation, image_dir, color_map, show_labels, axes[i]
            )
        for j in range(len(degradations), len(axes)):
            axes[j].axis("off")

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{img_id:05d}.png"
            fig.savefig(out_file, bbox_inches="tight")
            print(f"Saved as {out_file}")

        if show:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse COCO dataset augmented online with degradatations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", help="Path to COCO dataset")
    parser.add_argument(
        "--out-dir", default="images", help="Save visualization to the directory"
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=["val.json", "test.json"],
        help="Browse these annotations",
    )
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=3,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--ids", type=int, nargs="*", metavar="ID", help="Specific image IDs to display"
    )
    parser.add_argument(
        "--degradations",
        type=str,
        nargs="+",
        default=_DEGRADATION_LIST,
        help="Apply specific degradations for each image",
    )

    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly sample images",
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Random seed for deterministic shuffle"
    )
    parser.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw category labels or not",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the visualization interactively",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = Path(args.root_dir)
    image_dir = root_dir / "images"
    annotation_dir = root_dir / "annotations"

    kwargs = dict(
        degradations=args.degradations,
        num_images=args.num_images,
        image_ids=args.ids,
        shuffle=args.shuffle,
        seed=args.seed,
        show_labels=args.labels,
        show=args.show,
    )
    out_dir = Path(args.out_dir)
    for annotation in args.annotations:
        annotation: str
        annotation_file = annotation_dir / annotation
        if annotation_file.exists():
            dataset_name = root_dir.stem.lower().replace("-", "_")
            _out_dir = out_dir / f"{dataset_name}_{annotation_file.stem}"
            browse(image_dir, annotation_file, out_dir=_out_dir, **kwargs)
