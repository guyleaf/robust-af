import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import cv2
from rich.progress import track

from robust_af.utils import (
    collect_images,
    format_coco_annotation,
    format_coco_image,
    split_into_train_val,
)

_COCO_FILE = {
    "info": {
        "year": 2022,
        "version": 1,
        "description": "Sim2Air dataset (S-Eagle-B, S-Eagle-T, S-UAV-B, UAV-Eagle)",
        "contributor": "Antonella Barisic and Frano Petric and Stjepan Bogdan",
        "url": "https://github.com/larics/synthetic-UAV",
        "date_created": "2026-03-08",
    },
    "licenses": [
        {"id": 1, "name": "MIT License", "url": "https://opensource.org/license/mit"}
    ],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": "UAV"}],
}


def prepare_sim2air_subset(
    subset: str,
    image_files: list[Path],
    images_dir: Path,
    labels_dir: Path,
    out_images_dir: Path,
    out_annotations_dir: Path,
    annotation: bool = False,
):
    metadata = deepcopy(_COCO_FILE)
    images: list = metadata["images"]
    annotations: list = metadata["annotations"]

    annotation_id = 1

    for image_id, image_file in enumerate(
        track(image_files, description=f"{subset.capitalize()} images"), start=1
    ):
        # copy image to images directory
        if not annotation:
            shutil.copy2(image_file, out_images_dir)

        # get the size of the image
        image = cv2.imread(image_file.as_posix(), cv2.IMREAD_COLOR)
        image_h, image_w, _ = image.shape

        # read the annotation file of the frame
        label_file = labels_dir / image_file.with_suffix(".txt").relative_to(images_dir)
        with open(label_file, "r") as f:
            original_annotations = f.readlines()

        # convert YOLO to COCO format
        for object in track(original_annotations, description="Object", transient=True):
            # object_id center_x center_y w h (0-1 scale)
            _, x, y, w, h = map(float, object.split(" "))

            x = int(round((x - w / 2) * image_w))
            y = int(round((y - h / 2) * image_h))
            w = int(round(w * image_w))
            h = int(round(h * image_h))

            annotation_info = format_coco_annotation(
                annotation_id, image_id, 1, x, y, w, h
            )
            annotations.append(annotation_info)
            annotation_id += 1

        image_info = format_coco_image(image_id, image_file.name, image_h, image_w)
        images.append(image_info)

    metadata_file = out_annotations_dir / "test.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def prepare_sim2air_dataset(
    root_dir: Path, out_dir: Path, args: argparse.Namespace
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # make dir for annotations
    target_annotations_dir = out_dir / "annotations"
    target_annotations_dir.mkdir(parents=True, exist_ok=True)

    # make dir for images
    target_images_dir = out_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    images_dir = root_dir / "images"
    labels_dir = root_dir / "labels"
    image_files = collect_images(images_dir)

    if args.split:
        # split images into train, val subsets
        split_image_files = split_into_train_val(image_files, args.val_ratio, args.seed)
    else:
        split_image_files = {"test": image_files}

    for subset, image_files in split_image_files.items():
        prepare_sim2air_subset(
            subset,
            image_files,
            images_dir,
            labels_dir,
            target_images_dir,
            target_annotations_dir,
            annotation=args.annotation,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Sim2Air annotations from YOLO to COCO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="Root directory to read annotations and images"
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory to write annotations and images"
    )
    parser.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split into train and val subsets. Otherwise, make it as a test dataset.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Ratio of the validation subset"
    )
    parser.add_argument(
        "--seed", type=int, default=777, help="The seed of random sequence"
    )
    parser.add_argument(
        "--annotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Processing annotations only",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    prepare_sim2air_dataset(Path(args.root_dir), Path(args.out_dir), args)
