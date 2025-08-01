import argparse
import os
import shutil
import tempfile
from pathlib import Path

from rich import print
from ultralytics.data.converter import convert_coco


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO dataset to YOLO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", type=str, help="Root folder of COCO dataset")
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
    annotation_dir = root_dir / str(args.pop("annotation_dir"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # avoid increment_path() rename the folder
        shutil.rmtree(tmpdir)
        convert_coco(labels_dir=annotation_dir, save_dir=tmpdir, **args)
        src = Path(tmpdir) / "labels"
        dst = root_dir / "labels"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[green]Final Results saved to {dst}.")
