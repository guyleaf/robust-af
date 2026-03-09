import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from rich.progress import track

from robust_af.utils import (
    format_coco_annotation,
    format_coco_image,
    split_into_train_val,
)

_COCO_FILE = {
    "info": {
        "year": 2021,
        "version": 1,
        "description": "Det-Fly dataset",
        "url": "https://github.com/Jake-WU/Det-Fly",
        "date_created": "2023-03-28",
    },
    "licenses": [
        {"id": 1, "name": "MIT License", "url": "https://opensource.org/license/mit"}
    ],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": "UAV"}],
}


def prepare_det_fly_dataset(
    root_dir: str, out_dir: str, args: argparse.Namespace
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    images_dir = os.path.join(root_dir, "images")
    annotations_dir = os.path.join(root_dir, "annotations")

    # collect all images
    image_files = []
    for root, _, files in os.walk(images_dir, topdown=False):
        for file in files:
            image_files.append(Path(os.path.join(root, file)))
    # 2024.09.14 better reproducibility
    image_files.sort()

    # split images into train, val subsets
    split_image_files = split_into_train_val(image_files, args.val_ratio, args.seed)

    # make dir for annotations
    target_annotations_dir = os.path.join(out_dir, "annotations")
    os.makedirs(target_annotations_dir, exist_ok=True)

    # make dir for images
    target_images_dir = os.path.join(out_dir, "images")
    os.makedirs(target_images_dir, exist_ok=True)

    # TODO: Make this faster
    for subset, image_files in split_image_files.items():
        metadata = deepcopy(_COCO_FILE)
        images: list = metadata["images"]
        annotations: list = metadata["annotations"]

        annotation_id = 1

        for image_id, image_file in enumerate(
            track(image_files, description=f"{subset.capitalize()} image"), start=1
        ):
            image_file: Path

            image_name_with_extension = os.path.basename(image_file)
            image_name = os.path.splitext(image_name_with_extension)[0]

            # read the annotation file of the image
            annotation_file = os.path.join(
                annotations_dir, image_file.parent.name, f"{image_name}.xml"
            )
            original_annotation = ET.parse(annotation_file).getroot()

            image_w = int(original_annotation.find("size/width").text)
            image_h = int(original_annotation.find("size/height").text)

            images.append(
                format_coco_image(image_id, image_name_with_extension, image_h, image_w)
            )

            if not args.annotation:
                # copy image to target directory
                shutil.copy2(image_file, target_images_dir)

            for object in original_annotation.iter("object"):
                attributes = dict(
                    truncated=bool(int(object.find("truncated").text)),
                    difficult=bool(int(object.find("difficult").text)),
                )
                x1 = int(object.find("bndbox/xmin").text)
                y1 = int(object.find("bndbox/ymin").text)
                x2 = int(object.find("bndbox/xmax").text)
                y2 = int(object.find("bndbox/ymax").text)

                annotation_info = format_coco_annotation(
                    annotation_id, image_id, 1, x1, y1, x2 - x1, y2 - y1, attributes
                )
                annotations.append(annotation_info)

                annotation_id += 1

        # {subset}.json
        metadata_file = os.path.join(target_annotations_dir, f"{subset}.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split Det-Fly images into subsets and Convert Det-Fly annotations to COCO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="Root directory to read annotations and images"
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory to write annotations and images"
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
    prepare_det_fly_dataset(args.root_dir, args.out_dir, args)
