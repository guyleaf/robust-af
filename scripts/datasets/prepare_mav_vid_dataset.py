import argparse
import glob
import json
import os
import shutil
from copy import deepcopy

from rich.progress import track

from robust_af.utils import format_coco_annotation, format_coco_frame, format_coco_video

_COCO_FILE = {
    "info": {
        "year": 2020,
        "version": 1,
        "description": "Multirotor Aerial Vehicle VID (MAV-VID) dataset",
        "contributor": "alejodosr (https://alejandrorodriguezramos.me)",
        "url": "https://bitbucket.org/alejodosr/mav-vid-dataset",
        "date_created": "2023-03-29",
    },
    "licenses": [
        {"id": 1, "name": "MIT License", "url": "https://opensource.org/license/mit"}
    ],
    "images": [],
    "videos": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": "UAV"}],
}


def sort_images_by_video_name_then_image_name(image_file: str) -> tuple[int, int]:
    image_name, _ = os.path.splitext(os.path.basename(image_file))
    video_name, new_image_name = image_name.split("_")
    return (int(video_name), int(new_image_name))


def prepare_mav_vid_dataset(
    root_dir: str, out_dir: str, args: argparse.Namespace
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # make dir for annotations
    target_annotations_dir = os.path.join(out_dir, "annotations")
    os.makedirs(target_annotations_dir, exist_ok=True)

    # make dir for images
    target_images_dir = os.path.join(out_dir, "images")
    os.makedirs(target_images_dir, exist_ok=True)

    subsets = ["train", "val"]

    for subset in subsets:
        metadata = deepcopy(_COCO_FILE)
        images: list = metadata["images"]
        # videos: list = metadata["videos"]
        annotations: list = metadata["annotations"]

        video_id = 1
        videos = {}
        frame_ids = {}
        annotation_id = 1

        # make dir for {subset}
        target_dir = os.path.join(target_images_dir, subset)
        os.makedirs(target_dir, exist_ok=True)

        images_dir = os.path.join(root_dir, subset, "img")
        image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        image_files = sorted(image_files, key=sort_images_by_video_name_then_image_name)

        for image_id, image_file in enumerate(
            track(image_files, description=f"{subset.capitalize()} image"), start=1
        ):
            image_name, extension = os.path.splitext(os.path.basename(image_file))
            video_name, new_image_name = image_name.split("_")
            video_dir = os.path.join(target_dir, video_name)
            os.makedirs(video_dir, exist_ok=True)
            current_video_id = int(video_name)

            # read the annotation file of the video
            with open(os.path.join(images_dir, f"{image_name}.txt"), "r") as f:
                original_annotations = f.readlines()

            # avoid duplicates
            if current_video_id not in videos:
                videos[current_video_id] = format_coco_video(video_id, video_name)
                video_id += 1

            # copy image to video directory
            new_image_file = os.path.join(video_dir, new_image_name + extension)
            if not args.annotation:
                shutil.copy2(image_file, new_image_file)

            # read the annotation file of the video
            with open(os.path.join(images_dir, f"{image_name}.shape"), "r") as f:
                image_h, image_w, _ = map(int, f.readline().split(" "))

            for object in track(
                original_annotations, description="Object", transient=True
            ):
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

            frame_id = frame_ids.get(current_video_id, 1)

            frame_file = os.path.basename(new_image_file)
            frame_file = f"{subset}/{video_name}/{frame_file}"
            image_info = format_coco_frame(
                image_id,
                frame_file,
                image_h,
                image_w,
                current_video_id,
                frame_id,
            )
            images.append(image_info)

            frame_ids[current_video_id] = frame_id + 1

        metadata["videos"] = list(videos.values())

        # {subset}.json
        metadata_file = os.path.join(target_annotations_dir, f"{subset}.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)


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
    prepare_mav_vid_dataset(args.root_dir, args.out_dir, args)
