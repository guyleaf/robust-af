import argparse
import datetime
import json
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import cv2
from rich.progress import track

from robust_af.utils import (
    collect_images,
    format_coco_annotation,
    format_coco_frame,
    format_coco_image,
    format_coco_video,
    split_into_train_val,
)

_DETECTION_COCO_FILE = {
    "info": {
        "year": 2022,
        "version": 2,
        "description": "DUT Anti-UAV dataset - Detection",
        "url": "https://github.com/wangdongdut/DUT-Anti-UAV",
        "date_created": datetime.date.today().isoformat(),
    },
    "licenses": [
        {
            "id": 1,
            "name": "Apache License, Version 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        }
    ],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": "UAV"}],
}

_TRACKING_COCO_FILE = {
    "info": {
        "year": 2022,
        "version": 2,
        "description": "DUT Anti-UAV dataset - Tracking",
        "url": "https://github.com/wangdongdut/DUT-Anti-UAV",
        "date_created": datetime.date.today().isoformat(),
    },
    "licenses": [
        {
            "id": 1,
            "name": "Apache License, Version 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        }
    ],
    "images": [],
    "videos": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": "UAV"}],
}


def prepare_detection_subset(
    subset: str,
    image_files: list[Path],
    images_dir: Path,
    annotations_dir: Path,
    out_images_dir: Path,
    out_annotations_dir: Path,
    annotation: bool = False,
):
    if not annotation:
        # copy images to target directory
        shutil.copytree(images_dir, out_images_dir / subset)

    metadata = deepcopy(_DETECTION_COCO_FILE)
    images: list = metadata["images"]
    annotations: list = metadata["annotations"]

    annotation_id = 1

    for image_id, image_file in enumerate(
        track(image_files, description=f"{subset.capitalize()} images"), start=1
    ):
        rel_image_file = image_file.relative_to(images_dir)
        rel_out_image_file = subset / rel_image_file

        # read the annotation file of the image
        annotation_file = annotations_dir / rel_image_file.with_suffix(".xml")
        original_annotation = ET.parse(annotation_file).getroot()

        # get the size of the image
        image_w = int(original_annotation.find("size/width").text)
        image_h = int(original_annotation.find("size/height").text)

        # convert to COCO format
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

        image_info = format_coco_image(
            image_id, rel_out_image_file.as_posix(), image_h, image_w, license=1
        )
        images.append(image_info)

    # {subset}.json
    metadata_file = out_annotations_dir / f"{subset}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def prepare_tracking_subset(
    subset: str,
    video_dirs: list[Path],
    videos_dir: Path,
    annotations_dir: Path,
    out_images_dir: Path,
    out_annotations_dir: Path,
    annotation: bool = False,
):
    metadata = deepcopy(_TRACKING_COCO_FILE)
    images: list = metadata["images"]
    videos: list = metadata["videos"]
    annotations: list = metadata["annotations"]

    image_id = 1
    annotation_id = 1

    for video_id, video_dir in enumerate(
        track(video_dirs, description=f"{subset.capitalize()} videos"), start=1
    ):
        rel_video_dir = video_dir.relative_to(videos_dir)

        if not annotation:
            # copy video folder to target directory
            out_video_dir = out_images_dir / rel_video_dir
            shutil.copytree(video_dir, out_video_dir)
            ignored_file = out_video_dir / f"{rel_video_dir.stem}_gt_first.txt"
            ignored_file.unlink()

        # 2024.09.14 better reproducibility
        # frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        frame_files = collect_images(video_dir)

        # get the size of the frame
        frame = cv2.imread(frame_files[0].as_posix(), cv2.IMREAD_COLOR)
        video_h, video_w, _ = frame.shape

        # read the annotation file of the video
        annotation_file = rel_video_dir.with_name(f"{rel_video_dir.stem}_gt.txt")
        annotation_file = annotations_dir / annotation_file
        with open(annotation_file, "r") as f:
            original_annotations = f.readlines()

        if len(frame_files) != len(original_annotations):
            raise RuntimeError("The number of frames is not matched with annotations")

        # convert to COCO format
        for frame_id, frame_file in enumerate(
            track(frame_files, description="Frame", transient=True), start=1
        ):
            frame_no = frame_id - 1
            rel_frame_file = frame_file.relative_to(videos_dir)

            # x y w h
            x, y, w, h = map(int, original_annotations[frame_no].lstrip().split(" "))
            if x >= 0 and y >= 0 and w * h > 0:
                annotation_info = format_coco_annotation(
                    annotation_id, image_id, 1, x, y, w, h
                )
                annotations.append(annotation_info)
                annotation_id += 1

            frame_name = rel_frame_file.as_posix()
            frame_info = format_coco_frame(
                image_id, frame_name, video_h, video_w, video_id, frame_id, license=1
            )
            images.append(frame_info)
            image_id += 1

        videos.append(format_coco_video(video_id, rel_video_dir.as_posix()))

    # {subset}.json
    metadata_file = out_annotations_dir / f"{subset}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def prepare_detection_dataset(
    root_dir: Path, out_dir: Path, args: argparse.Namespace
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # make dir for annotations
    target_annotations_dir = out_dir / "annotations"
    target_annotations_dir.mkdir(parents=True, exist_ok=True)

    # make dir for images
    target_images_dir = out_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    for subset in ["train", "val", "test"]:
        images_dir = root_dir / subset / "img"
        annotations_dir = root_dir / subset / "xml"

        # 2024.09.14 better reproducibility
        # image_files = sorted(os.listdir(images_dir))
        image_files = collect_images(images_dir)
        prepare_detection_subset(
            subset,
            image_files,
            images_dir,
            annotations_dir,
            target_images_dir,
            target_annotations_dir,
            annotation=args.annotation,
        )


def prepare_tracking_dataset(
    root_dir: Path, out_dir: Path, args: argparse.Namespace
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # make dir for annotations
    target_annotations_dir = out_dir / "annotations"
    target_annotations_dir.mkdir(parents=True, exist_ok=True)

    # make dir for images
    target_images_dir = out_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = root_dir / "data"
    annotations_dir = root_dir / "labels"
    # 2024.09.14 better reproducibility
    # video_dirs = [
    #     os.path.join(videos_dir, video_dir)
    #     for video_dir in sorted(os.listdir(videos_dir))
    # ]
    video_dirs = sorted(videos_dir.iterdir())

    # split videos into train, val subsets
    split_video_dirs = split_into_train_val(video_dirs, args.val_ratio, args.seed)
    # test subset contains all images for generalization test
    split_video_dirs["test"] = video_dirs

    for subset, video_dirs in split_video_dirs.items():
        prepare_tracking_subset(
            subset,
            video_dirs,
            videos_dir,
            annotations_dir,
            target_images_dir,
            target_annotations_dir,
            annotation=args.annotation or subset == "test",
        )


def prepare_dut_anti_uav_dataset(
    root_dir: Path, out_dir: Path, args: argparse.Namespace
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_fn_map = {
        "detection": prepare_detection_dataset,
        "tracking": prepare_tracking_dataset,
    }

    for dataset_name, fn in dataset_fn_map.items():
        dataset_root_dir = root_dir / dataset_name
        dataset_out_dir = out_dir / dataset_name
        if dataset_root_dir.is_dir():
            print(f"Preparing {dataset_name.capitalize()} Dataset")
            fn(dataset_root_dir, dataset_out_dir, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DUT Anti-UAV annotations to COCO format.",
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
    prepare_dut_anti_uav_dataset(Path(args.root_dir), Path(args.out_dir), args)
