import argparse
import datetime
import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import ffmpegio
from rich.progress import track

from robust_af.utils import (
    VideoSamplingMethod,
    choose_ffmpeg_vf,
    collect_images,
    collect_videos,
    extract_video_frames,
    format_coco_annotation,
    format_coco_frame,
    format_coco_video,
)

_COCO_FILE = {
    "info": {
        "year": 2023,
        "version": 2,
        "description": "Drove-vs-bird 2023 Challenge dataset",
        "url": "https://wosdetc2023.wordpress.com/drone-vs-bird-detection-challenge/",
        "date_created": datetime.date.today().isoformat(),
    },
    "licenses": [],
    "images": [],
    "videos": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "drone", "supercategory": "UAV"}],
}

# matched with https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark/tree/main/datasets/drone-vs-bird
_TRAIN_VIDEOS = [
    "00_01_52_to_00_01_58",
    "00_06_10_to_00_06_27",
    "00_09_30_to_00_10_09",
    "00_10_09_to_00_10_40",
    "2019_08_19_C0001_5319_phantom",
    "2019_08_19_GP015869_1520_inspire",
    "2019_09_02_C0002_3700_mavic",
    "2019_10_16_C0003_1700_matrice",
    "2019_10_16_C0003_4613_mavic",
    "2019_10_16_C0003_5043_mavic",
    "2019_11_14_C0001_3922_matrice",
    "GOPR5842_002",
    "GOPR5842_005",
    "GOPR5842_007",
    "GOPR5843_002",
    "GOPR5843_005",
    "GOPR5844_002",
    "GOPR5844_004",
    "GOPR5845_001",
    "GOPR5845_004",
    "GOPR5846_002",
    "GOPR5846_005",
    "GOPR5848_002",
    "custom_fixed_wing_1",
    "custom_fixed_wing_2",
    "dji_matrice_210_hillside",
    "dji_matrice_210_mountain",
    "dji_matrice_210_sky",
    "dji_mavick_distant_hillside",
    "dji_mavick_hillside_off_focus",
    "dji_mavick_mountain",
    "dji_mavick_mountain_cruise",
    "dji_pantom_landing_custom_fixed_takeoff",
    "dji_phantom_4_hillside_cross",
    "dji_phantom_4_long_takeoff",
    "dji_phantom_4_mountain_hover",
    "dji_phantom_4_swarm_noon",
    "fixed_wing_over_hill_1",
    "fixed_wing_over_hill_2",
    "gopro_000",
    "gopro_001",
    "gopro_002",
    "gopro_003",
    "gopro_004",
    "gopro_006",
    "gopro_007",
    "gopro_008",
    "matrice_600_2",
    "matrice_600_3",
    "off_focus_parrot_birds",
    "parot_disco_takeoff",
    "parrot_clear_birds",
    "parrot_clear_birds_med_range",
    "parrot_disco_distant_cross",
    "parrot_disco_distant_cross_3",
    "parrot_disco_long_session",
    "parrot_disco_midrange_cross",
    "parrot_disco_zoomin_zoomout",
    "swarm_dji_phantom",
    "two_parrot_disco_1",
    # ignore it due to inconsistent annotations
    # "2019_10_16_C0003_3633_inspire",
]

# matched with https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark/tree/main/datasets/drone-vs-bird
_VALIDATION_VIDEOS = [
    "00_02_45_to_00_03_10_cut",
    "2019_08_19_GOPR5869_1530_phantom",
    "2019_09_02_C0002_2527_inspire",
    "2019_09_02_GOPR5871_1058_solo",
    "GOPR5847_003",
    "GOPR5847_004",
    "GOPR5848_004",
    "distant_parrot_2",
    "distant_parrot_with_birds",
    "dji_matrice_210_off_focus",
    "dji_mavick_close_buildings",
    "dji_phantom_mountain_cross",
    "gopro_005",
    "swarm_dji_phantom4_2",
    "two_distant_phantom",
    "two_uavs_plus_airplane",
]

_TEST_VIDEOS = _TRAIN_VIDEOS + _VALIDATION_VIDEOS


def filter_video_with_video_list(video_list: list[str]):
    video_set = set(video_list)

    def filter_(video_file: Path):
        return video_file.stem in video_set

    return filter_


def split_videos_into_subsets(video_files: list[Path]):
    assert len(set(_TRAIN_VIDEOS) & set(_VALIDATION_VIDEOS)) == 0

    # split videos by pre-defined list
    split_video_files = {
        "train": list(filter(filter_video_with_video_list(_TRAIN_VIDEOS), video_files)),
        "val": list(
            filter(filter_video_with_video_list(_VALIDATION_VIDEOS), video_files)
        ),
        "test": list(filter(filter_video_with_video_list(_TEST_VIDEOS), video_files)),
    }
    assert len(split_video_files["train"]) == len(_TRAIN_VIDEOS) and len(
        split_video_files["val"]
    ) == len(_VALIDATION_VIDEOS)

    for subset, video_files in split_video_files.items():
        print(f"Total number of {subset} videos: ", len(video_files))
    return split_video_files


def prepare_dds_subset(
    subset: str,
    video_files: list[Path],
    videos_dir: Path,
    annotations_dir: Path,
    out_images_dir: Path,
    out_annotations_dir: Path,
    annotation: bool = False,
):
    metadata = deepcopy(_COCO_FILE)
    images: list = metadata["images"]
    videos: list = metadata["videos"]
    annotations: list = metadata["annotations"]

    image_id = 1
    annotation_id = 1

    for video_id, video_file in enumerate(
        track(video_files, description=f"{subset.capitalize()} videos"), start=1
    ):
        rel_video_file = video_file.relative_to(videos_dir)
        rel_video_dir = rel_video_file.with_suffix("")

        # make dir for {video_file}
        out_video_dir = out_images_dir / rel_video_dir
        out_video_dir.mkdir(parents=True, exist_ok=True)

        if not annotation:
            vf = choose_ffmpeg_vf(
                sample_method=args.sample_method,
                sample_interval=args.sample_interval,
                subset=subset,
            )
            extract_video_frames(video_file.as_posix(), out_video_dir.as_posix(), vf=vf)

        # get the size of the image
        video_info = ffmpegio.probe.video_streams_basic(video_file)[0]
        video_h, video_w = (video_info["height"], video_info["width"])

        # read the annotation file of the video
        annotation_file = annotations_dir / rel_video_file.with_suffix(".txt")
        with open(annotation_file, "r") as f:
            original_annotations = f.readlines()

        # 2024.09.14 better reproducibility
        frame_files = collect_images(out_video_dir)
        if len(frame_files) == 0:
            raise RuntimeError(f"No image files found for video {out_video_dir}.")

        for frame_id, frame_file in enumerate(
            track(frame_files, description="Frame", transient=True), start=1
        ):
            rel_frame_file = frame_file.relative_to(out_images_dir)
            # based on output image, %5d.png
            frame_no = int(frame_file.stem)

            # frame_no number_of_objects x y w h cls x y w h cls...
            original_annotation = original_annotations[frame_no].split(" ")
            if frame_no != int(original_annotation[0]):
                raise RuntimeError(
                    f"The annotation of frame no {frame_no} is not found."
                )

            number_of_objects = int(original_annotation[1])
            for i in range(number_of_objects):
                # 2 head columns + 5 columns per object
                start = 5 * i + 2
                end = 5 * (i + 1) + 2
                # ignore cls & handle negative values
                x, y, w, h = map(
                    partial(max, 0), map(int, original_annotation[start : end - 1])
                )
                if w * h > 0:
                    annotation_info = format_coco_annotation(
                        annotation_id, image_id, 1, x, y, w, h
                    )
                    annotations.append(annotation_info)
                    annotation_id += 1

            frame_info = format_coco_frame(
                image_id,
                rel_frame_file.as_posix(),
                video_h,
                video_w,
                video_id,
                frame_id,
            )
            images.append(frame_info)
            image_id += 1

        videos.append(format_coco_video(video_id, rel_video_dir.as_posix()))

    # {subset}.json
    metadata_file = out_annotations_dir / f"{subset}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def prepare_dds_dataset(
    root_dir: Path, out_dir: Path, args: argparse.Namespace
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # make dir for annotations
    target_annotations_dir = out_dir / "annotations"
    target_annotations_dir.mkdir(parents=True, exist_ok=True)

    # make dir for images
    target_images_dir = out_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = root_dir / "videos"
    annotations_dir = root_dir / "annotations"
    video_files = collect_videos(videos_dir)

    # split videos by pre-defined list
    split_video_files = split_videos_into_subsets(video_files)

    for subset, video_files in split_video_files.items():
        prepare_dds_subset(
            subset,
            video_files,
            videos_dir,
            annotations_dir,
            target_images_dir,
            target_annotations_dir,
            annotation=args.annotation or subset == "test",
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Drone-vs-Bird videos to images and Drone-vs-Bird annotations to COCO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="Root directory to read annotations and videos"
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory to write annotations and images"
    )
    parser.add_argument(
        "--sample-method",
        type=VideoSamplingMethod,
        choices=list(VideoSamplingMethod),
        default=VideoSamplingMethod.SIMPLE,
        help="Method for frame sampling.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Sample at every n frames. 1 => all frames will be extracted. If sample_method is 'i_frame', then this option will be ignored.",
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
    prepare_dds_dataset(Path(args.root_dir), Path(args.out_dir), args)
