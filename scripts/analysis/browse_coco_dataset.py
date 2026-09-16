#!/usr/bin/env python3
"""Browse and visualize a COCO dataset with bounding boxes using matplotlib."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO


def get_color_map(num_classes: int) -> list[tuple[float, float, float]]:
    rng = random.Random(42)
    return [(rng.random(), rng.random(), rng.random()) for _ in range(num_classes)]


def draw(coco: COCO, img_id: int, image_dir: Path):
    img_info = coco.imgs[img_id]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    img_path = image_dir / img_info["file_name"]
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.dataset["categories"]}

    font = ImageFont.load_default()
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cat_id = ann["category_id"]

        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

        text = cat_id_to_name[cat_id]
        bbox = draw.textbbox((x + 2, y), text, font=font)
        draw.rectangle(bbox, fill="blue")
        draw.text((x + 2, y), text=text, font=font, fill="white")

    return image


def visualize_sample(
    coco: COCO,
    img_id: int,
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

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.imshow(image)
    # ax.set_title(f"{img_info['file_name']} ({len(anns)} annotations)", fontsize=10)
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
    num_images: int = 9,
    image_ids: list[int] | None = None,
    shuffle: bool = False,
    seed: int | None = 2026,
    show_labels: bool = True,
    out_file: str | Path | None = "result.png",
    show: bool = False,
    single: bool = False,
) -> None:
    image_dir = Path(image_dir)
    coco = COCO(annotation_file)

    all_ids = list(coco.imgs.keys())
    if image_ids:
        ids = [i for i in image_ids if i in coco.imgs]
    elif shuffle:
        ids = random.Random(seed).sample(all_ids, min(num_images, len(all_ids)))
    else:
        ids = all_ids[:num_images]

    if single:
        for i, img_id in enumerate(ids):
            image = draw(coco, img_id, image_dir)
            if out_file is not None:
                out_dir = out_file.with_name(out_file.stem)
                out_dir.mkdir(parents=True, exist_ok=True)
                image.save(out_dir / f"{img_id:05d}.png")

            if show:
                image.show(title=f"image_id: {img_id}")
        if out_file is not None:
            print(f"Saved to {out_file.parent} folder")
    else:
        num_classes = len(coco.dataset.get("categories", []))
        color_map = get_color_map(max(num_classes, 1))

        cols = min(4, len(ids))
        rows = (len(ids) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(13 * cols, 9 * rows),
            constrained_layout=True,
            # rgidspec_kw={"wspace": 0, "hspace": 0},
        )
        axes = np.array(axes).flatten() if len(ids) > 1 else [axes]

        for i, img_id in enumerate(ids):
            visualize_sample(coco, img_id, image_dir, color_map, show_labels, axes[i])

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        if out_file is not None:
            out_file = Path(out_file)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_file, bbox_inches="tight")
            print(f"Saved to {out_file}")

        if show:
            plt.show()

        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse COCO dataset with bounding boxes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", help="Path to COCO dataset")
    parser.add_argument(
        "--out-dir", default="images", help="Save visualization to the directory"
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=["val.json", "test.json", "degraded_val.json", "degraded_test.json"],
        help="Browse these annotations",
    )
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        default=12,
        help="Number of images to display (default: 9)",
    )
    parser.add_argument(
        "--ids", type=int, nargs="*", metavar="ID", help="Specific image IDs to display"
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
    parser.add_argument(
        "--single",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save images as files instead of one graph",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = Path(args.root_dir)
    image_dir = root_dir / "images"
    annotation_dir = root_dir / "annotations"

    kwargs = dict(
        num_images=args.num_images,
        image_ids=args.ids,
        shuffle=args.shuffle,
        seed=args.seed,
        show_labels=args.labels,
        show=args.show,
        single=args.single,
    )
    out_dir = Path(args.out_dir)
    for annotation in args.annotations:
        annotation: str
        annotation_file = annotation_dir / annotation
        if annotation_file.exists():
            dataset_name = root_dir.stem.lower().replace("-", "_")
            out_file = out_dir / f"{dataset_name}_{annotation_file.stem}.png"
            browse(image_dir, annotation_file, out_file=out_file, **kwargs)
