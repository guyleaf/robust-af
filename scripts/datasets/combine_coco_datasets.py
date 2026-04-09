import argparse
import asyncio
import datetime
import json
import os
import posixpath
import shutil
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional

from pycocotools.coco import COCO
from rich import print
from rich.progress import track

from robust_af.utils import format_coco_image

_COCO_FILE = {
    "info": {
        "year": datetime.date.today().year,
        "version": 1,
        "description": "Combined dataset ({})",
        "url": "",
        "date_created": datetime.date.today().isoformat(),
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [],
}


def load_annotations(
    datasets: dict[str, Path], annotation_file: str
) -> dict[str, Optional[COCO]]:
    dataset_to_coco: dict[str, Optional[COCO]] = {}
    for name, root_dir in datasets.items():
        path = root_dir / "annotations" / annotation_file
        if path.exists():
            dataset_to_coco[name] = COCO(path)
        else:
            warnings.warn(
                f"The anntation {annotation_file} is not found in {name} dataset."
            )
            dataset_to_coco[name] = None
    return dataset_to_coco


def validate_and_get_categories(
    dataset_to_coco: dict[str, Optional[COCO]],
) -> list[dict]:
    categories: Optional[list[dict]] = None
    for name, coco in dataset_to_coco.items():
        if coco is None:
            continue

        # sort the categories by id
        _categories = sorted(
            coco.dataset["categories"], key=lambda category: category["id"]
        )

        if categories is not None:
            assert all(
                [
                    category_1 == category_2
                    for category_1, category_2 in zip(categories, _categories)
                ]
            ), f"Found inconsistent categories in {name} dataset."
        else:
            categories = _categories
    assert categories is not None, "The categories are not found."
    return categories


async def combine_datasets(
    datasets: dict[str, Path],
    annotation_files: list[str],
    target_images_dir: Path,
    target_annotations_dir: Path,
    annotation_only: bool = False,
):
    copy_tasks = []
    for annotation_file in annotation_files:
        # load annotation of datasets
        dataset_to_coco = load_annotations(datasets, annotation_file)

        coco = deepcopy(_COCO_FILE)
        coco["categories"] = validate_and_get_categories(dataset_to_coco)

        image_id = 1
        annotation_id = 1
        coco_images = []
        coco_annotations = []
        for dataset_name, coco_ in track(
            dataset_to_coco.items(), description=f"Processing {annotation_file}..."
        ):
            if coco_ is None:
                continue

            for image in coco_.dataset["images"]:
                _annotations = coco_.loadAnns(coco_.getAnnIds(imgIds=image["id"]))

                rel_src = image["file_name"]
                rel_dst = posixpath.join(dataset_name, rel_src)

                # copy image to destination asynchronously
                if not annotation_only:
                    src = datasets[dataset_name] / "images" / rel_src
                    dst = target_images_dir / rel_dst
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    copy_tasks.append(
                        asyncio.create_task(asyncio.to_thread(shutil.copy2, src, dst))
                    )

                # reindex image
                # some datasets are coming from videos (get_frame_format)
                # so, here we use format function to get image
                image = format_coco_image(
                    image_id,
                    rel_dst,
                    image["height"],
                    image["width"],
                    license=image["license"],
                    caption=image.get("caption", None),
                    weight=image.get("weight", None),
                )

                # reindex annotations
                for _annotation in _annotations:
                    _annotation = deepcopy(_annotation)
                    _annotation["id"] = annotation_id
                    _annotation["image_id"] = image_id

                    coco_annotations.append(_annotation)
                    annotation_id += 1

                coco_images.append(image)
                image_id += 1

        # save annotation
        coco["images"] = coco_images
        coco["annotations"] = coco_annotations
        out_file = target_annotations_dir / annotation_file
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(coco, f)

        print("Total number of images:", image_id - 1)
        print("Total number of annotation:", annotation_id - 1)
        print()

    if len(copy_tasks) != 0:
        print("Waiting all copy tasks to be finished...")
        dones: set[asyncio.Future]
        pendings: set[asyncio.Future]
        dones, pendings = await asyncio.wait(
            copy_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        # cancel all pending tasks if raising exception
        for pending in pendings:
            pending.cancel()

        for done in dones:
            ex = done.exception()
            if ex is not None:
                raise ex


def combine_coco_datasets(
    root_dirs: list[str],
    out_dir: str,
    dataset_names: list[str],
    annotation_files: list[str],
    annotation_only: bool = False,
):
    target_images_dir = Path(os.path.join(out_dir, "images"))
    target_annotations_dir = Path(os.path.join(out_dir, "annotations"))
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_annotations_dir.mkdir(parents=True, exist_ok=True)

    # make a map from dataset name to root_dir
    datasets = {
        dataset_name: Path(root_dir)
        for root_dir, dataset_name in zip(root_dirs, dataset_names)
    }

    _COCO_FILE["info"]["description"] = _COCO_FILE["info"]["description"].format(
        ", ".join(dataset_names)
    )

    # combine coco annotations
    asyncio.run(
        combine_datasets(
            datasets,
            annotation_files,
            target_images_dir,
            target_annotations_dir,
            annotation_only=annotation_only,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine COCO datasets (only works in annotations with the same categories)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dirs", type=str, nargs="+", help="Root folders of COCO datasets"
    )
    parser.add_argument("out_dir", type=str, help="Output folder of combined dataset")
    parser.add_argument(
        "--dataset-names",
        type=str,
        nargs="*",
        default=[],
        help="Name of datasets (optional)",
    )
    parser.add_argument(
        "--annotation-files",
        type=str,
        nargs="+",
        default=["train.json", "val.json", "test.json"],
    )
    parser.add_argument(
        "--annotation-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="processing annotations only",
    )
    args = parser.parse_args()

    assert len(args.root_dirs) > 1, "The number of datasets should be at least two."

    number_of_datasets = len(args.dataset_names)
    if number_of_datasets > 0:
        assert number_of_datasets == len(args.root_dirs), (
            "The number of root dirs and dataset names should be the same."
        )
    else:
        args.dataset_names = [os.path.basename(root_dir) for root_dir in args.root_dirs]
    return args


if __name__ == "__main__":
    args = parse_args()
    combine_coco_datasets(**vars(args))
