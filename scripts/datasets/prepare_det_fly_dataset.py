import argparse
import datetime
import json
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from rich.progress import track

from robust_af.utils import (
    collect_images,
    format_coco_annotation,
    format_coco_image,
    split_into_train_val,
)

_COCO_FILE = {
    "info": {
        "year": 2021,
        "version": 2,
        "description": "Det-Fly dataset",
        "url": "https://github.com/Jake-WU/Det-Fly",
        "date_created": datetime.date.today().isoformat(),
    },
    "licenses": [
        {"id": 1, "name": "MIT License", "url": "https://opensource.org/license/mit"}
    ],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": "UAV"}],
}


def prepare_det_fly_subset(
    subset: str,
    image_files: list[Path],
    images_dir: Path,
    annotations_dir: Path,
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
        rel_image_file = image_file.relative_to(images_dir)
        # copy image to target directory
        if not annotation:
            out_image_file = out_images_dir / rel_image_file
            out_image_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_file, out_image_file)

        # read the annotation file of the image
        annotation_file = annotations_dir / rel_image_file.with_suffix(".xml")
        original_annotation = ET.parse(annotation_file).getroot()

        # get the size of the image
        image_w = int(original_annotation.find("size/width").text)
        image_h = int(original_annotation.find("size/height").text)

        # convert to COCO format
        for object in track(
            original_annotation.iter("object"), description="Object", transient=True
        ):
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

        image_info = format_coco_image(
            image_id, rel_image_file.as_posix(), image_h, image_w, license=1
        )
        images.append(image_info)

    # {subset}.json
    metadata_file = out_annotations_dir / f"{subset}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def prepare_det_fly_dataset(
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
    annotations_dir = root_dir / "annotations"
    image_files = collect_images(images_dir)

    # split images into train, val subsets
    split_image_files = split_into_train_val(image_files, args.val_ratio, args.seed)
    # test subset contains all images for generalization test
    split_image_files["test"] = image_files

    for subset, image_files in split_image_files.items():
        prepare_det_fly_subset(
            subset,
            image_files,
            images_dir,
            annotations_dir,
            target_images_dir,
            target_annotations_dir,
            annotation=args.annotation or subset == "test",
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split and Convert Det-Fly annotations to COCO format.",
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
    prepare_det_fly_dataset(Path(args.root_dir), Path(args.out_dir), args)
