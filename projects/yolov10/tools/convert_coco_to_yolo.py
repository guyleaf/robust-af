import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from rich import print
from rich.progress import track
from ultralytics.data.converter import convert_coco


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO dataset to YOLO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", type=str, help="Root folder of COCO dataset")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="images",
        help="Relative path of image folder",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="annotations",
        help="Relative path of annotation folder",
    )
    parser.add_argument(
        "--use-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include segmentation masks in the output",
    )
    parser.add_argument(
        "--use-keypoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include keypoint annotations in the output",
    )
    parser.add_argument(
        "--cls91to80",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs",
    )
    args = parser.parse_args()
    assert os.path.isdir(args.root_dir)
    return args


if __name__ == "__main__":
    args = vars(parse_args())
    root_dir = Path(args.pop("root_dir"))
    image_dir = str(args.pop("image_dir"))
    annotation_dir = root_dir / str(args.pop("annotation_dir"))
    label_dir = root_dir / "labels"

    if label_dir.exists():
        shutil.rmtree(label_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # avoid increment_path() renaming the folder
        shutil.rmtree(tmpdir)
        convert_coco(labels_dir=annotation_dir, save_dir=tmpdir, **args)
        tmpdir = tmpdir / "labels"

        # combine train/val/test folder created by convert_coco()
        # make labels folder follow the paths in images folder to keep compatibility
        subsets = list(annotation_dir.glob("*.json"))
        for subset in track(subsets, description="Combining..."):
            src = tmpdir / subset.stem
            dst = label_dir
            shutil.copytree(src, dst, dirs_exist_ok=True)

        # use the .txt files to split into subsets
        for subset in track(subsets, description="Spliting..."):
            with open(subset) as f:
                content = json.load(f)
            with open(root_dir / subset.with_suffix(".txt").name, "w") as f:
                f.writelines(
                    f"./{image_dir}/" + image["file_name"] + "\n"
                    for image in content["images"]
                )

    print(f"[green]Final Results saved to {label_dir}.")
