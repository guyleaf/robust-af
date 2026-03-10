import argparse
import datetime
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

from rich.progress import track

from robust_af.utils import collect_images, format_coco_annotation, format_coco_image

_COCO_FILE = {
    "info": {
        "year": 2020,
        "version": 2,
        "description": "Multirotor Aerial Vehicle VID (MAV-VID) dataset",
        "contributor": "alejodosr (https://alejandrorodriguezramos.me)",
        "url": "https://bitbucket.org/alejodosr/mav-vid-dataset",
        "date_created": datetime.date.today().isoformat(),
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": "UAV"}],
}


def prepare_mav_vid_subset(
    subset: str,
    image_files: list[Path],
    images_dir: Path,
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
        rel_out_image_file = subset / rel_image_file
        # copy image to video directory
        if not annotation:
            out_image_file = out_images_dir / rel_out_image_file
            out_image_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_file, out_image_file)

        # get the size of the image
        with open(image_file.with_suffix(".shape"), "r") as f:
            image_h, image_w, _ = map(int, f.readline().split(" "))

        # read the annotation file of the video
        with open(image_file.with_suffix(".txt"), "r") as f:
            original_annotations = f.readlines()

        # convert YOLO to COCO format
        for object in track(original_annotations, description="Object", transient=True):
            # object_id center_x center_y w h (0-1 scale)
            _, x, y, w, h = map(float, object.split(" "))

            # convert YOLO to COCO format
            x = int(round((x - w / 2) * image_w))
            y = int(round((y - h / 2) * image_h))
            w = int(round(w * image_w))
            h = int(round(h * image_h))

            annotation_info = format_coco_annotation(
                annotation_id, image_id, 1, x, y, w, h
            )
            annotations.append(annotation_info)
            annotation_id += 1

        image_info = format_coco_image(
            image_id, rel_out_image_file.as_posix(), image_h, image_w
        )
        images.append(image_info)

    # {subset}.json
    metadata_file = out_annotations_dir / f"{subset}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def prepare_mav_vid_dataset(
    root_dir: Path, out_dir: Path, args: argparse.Namespace
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # make dir for annotations
    target_annotations_dir = out_dir / "annotations"
    target_annotations_dir.mkdir(parents=True, exist_ok=True)

    # make dir for images
    target_images_dir = out_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    for subset in ["train", "val"]:
        images_dir = os.path.join(root_dir, subset, "img")
        image_files = collect_images(images_dir)

        prepare_mav_vid_subset(
            subset,
            image_files,
            images_dir,
            target_images_dir,
            target_annotations_dir,
            annotation=args.annotation,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove unnecessary files and Convert MAV-VID annotations to COCO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="Root directory to read annotations and images"
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory to write annotations and images"
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
    prepare_mav_vid_dataset(Path(args.root_dir), Path(args.out_dir), args)
