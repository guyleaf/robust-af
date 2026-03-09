import argparse
import glob
import json
import os
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy

import cv2
from rich.progress import track

from robust_af.utils import (
    format_coco_annotation,
    format_coco_frame,
    format_coco_image,
    format_coco_video,
    split_into_train_val,
)

_DETECTION_COCO_FILE = {
    "info": {
        "year": 2022,
        "version": 1,
        "description": "DUT Anti-UAV dataset - Detection",
        "url": "https://github.com/wangdongdut/DUT-Anti-UAV",
        "date_created": "2023-03-28",
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
        "version": 1,
        "description": "DUT Anti-UAV dataset - Tracking",
        "url": "https://github.com/wangdongdut/DUT-Anti-UAV",
        "date_created": "2023-03-28",
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


def prepare_detection_dataset(
    root_dir: str, out_dir: str, args: argparse.Namespace
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    subsets = ["train", "val", "test"]

    # make dir for annotations
    target_annotations_dir = os.path.join(out_dir, "annotations")
    os.makedirs(target_annotations_dir, exist_ok=True)

    # make dir for images
    target_images_dir = os.path.join(out_dir, "images")
    os.makedirs(target_images_dir, exist_ok=True)

    for subset in subsets:
        metadata = deepcopy(_DETECTION_COCO_FILE)
        images: list = metadata["images"]
        annotations: list = metadata["annotations"]

        annotation_id = 1

        images_dir = os.path.join(root_dir, subset, "img")
        annotations_dir = os.path.join(root_dir, subset, "xml")
        # 2024.09.14 better reproducibility
        image_files = sorted(os.listdir(images_dir))

        if not args.annotation:
            # copy images to target directory
            shutil.copytree(images_dir, os.path.join(target_images_dir, subset))

        for image_id, image_file in enumerate(
            track(image_files, description=f"{subset.capitalize()} image"), start=1
        ):
            image_name = os.path.splitext(image_file)[0]

            # read the annotation file of the image
            annotation_file = os.path.join(annotations_dir, f"{image_name}.xml")
            original_annotation = ET.parse(annotation_file).getroot()

            image_w = int(original_annotation.find("size/width").text)
            image_h = int(original_annotation.find("size/height").text)

            images.append(
                format_coco_image(image_id, f"{subset}/{image_file}", image_h, image_w)
            )

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


def prepare_tracking_dataset(
    root_dir: str, out_dir: str, args: argparse.Namespace
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # make dir for annotations
    target_annotations_dir = os.path.join(out_dir, "annotations")
    os.makedirs(target_annotations_dir, exist_ok=True)

    # make dir for images
    target_images_dir = os.path.join(out_dir, "images")
    os.makedirs(target_images_dir, exist_ok=True)

    videos_dir = os.path.join(root_dir, "data")
    annotations_dir = os.path.join(root_dir, "labels")
    video_dirs = [
        os.path.join(videos_dir, video_dir)
        # 2024.09.14 better reproducibility
        for video_dir in sorted(os.listdir(videos_dir))
    ]

    split_video_dirs = split_into_train_val(video_dirs, args.val_ratio, args.seed)

    for subset, video_dirs in split_video_dirs.items():
        metadata = deepcopy(_TRACKING_COCO_FILE)
        images: list = metadata["images"]
        videos: list = metadata["videos"]
        annotations: list = metadata["annotations"]

        image_id = 1
        annotation_id = 1

        for video_id, video_dir in enumerate(
            track(video_dirs, description=f"{subset.capitalize()} video"), start=1
        ):
            video_name = os.path.splitext(os.path.basename(video_dir))[0]

            # read the annotation file of the video
            with open(os.path.join(annotations_dir, f"{video_name}_gt.txt"), "r") as f:
                original_annotations = f.readlines()

            videos.append(format_coco_video(video_id, video_name))

            if not args.annotation:
                # copy video folder to target directory
                shutil.copytree(
                    video_dir,
                    os.path.join(target_images_dir, video_name),
                    ignore=lambda _1, _2, video_name=video_name: [
                        f"{video_name}_gt_first.txt"
                    ],
                    dirs_exist_ok=True,
                )

            # 2024.09.14 better reproducibility
            frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
            frame = cv2.imread(frame_files[0], cv2.IMREAD_COLOR)
            video_h, video_w, _ = frame.shape

            for frame_id, frame_file in enumerate(
                track(frame_files, description="Frame", transient=True)
            ):
                # x y w h
                x, y, w, h = map(
                    int, original_annotations[frame_id].lstrip().split(" ")
                )
                if x >= 0 and y >= 0 and w * h > 0:
                    annotation_info = format_coco_annotation(
                        annotation_id, image_id, 1, x, y, w, h
                    )
                    annotations.append(annotation_info)
                    annotation_id += 1

                frame_file = os.path.basename(frame_file)
                frame_file = f"{video_name}/{frame_file}"
                frame_info = format_coco_frame(
                    image_id, frame_file, video_h, video_w, video_id, frame_id
                )
                images.append(frame_info)
                image_id += 1

        # {subset}.json
        metadata_file = os.path.join(target_annotations_dir, f"{subset}.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)


def prepare_dut_anti_uav_dataset(
    root_dir: str, out_dir: str, args: argparse.Namespace
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    detection_root_dir = os.path.join(root_dir, "detection")
    detection_out_dir = os.path.join(out_dir, "detection")
    if os.path.isdir(detection_root_dir):
        print("Preparing Detection Dataset")
        prepare_detection_dataset(detection_root_dir, detection_out_dir, args)

    tracking_root_dir = os.path.join(root_dir, "tracking")
    tracking_out_dir = os.path.join(out_dir, "tracking")
    if os.path.isdir(tracking_root_dir):
        print("Preparing Tracking Dataset")
        prepare_tracking_dataset(tracking_root_dir, tracking_out_dir, args)


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
    prepare_dut_anti_uav_dataset(args.root_dir, args.out_dir, args)
