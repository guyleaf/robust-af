import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Union

import numpy as np
import PIL.Image as Image
import torch
from pycocotools.coco import COCO
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

from robust_af.transforms import apply_random_degradation
from robust_af.utils import seed_everything

CONSOLE = Console()


@dataclass
class ImageMetadata:
    id: int
    name: str


class DegradedCOCODataset(Dataset):
    def __init__(
        self,
        images_dir: Union[str, Path],
        annotation_file: Union[str, Path, COCO],
        transform: Callable[..., tuple[np.ndarray, str]],
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.transform = transform

        self._loaded = False
        if isinstance(annotation_file, COCO):
            self.full_init()

    def _load_image(self, id: int) -> np.ndarray:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return np.asarray(Image.open(self.images_dir / path).convert("RGB"))

    def full_init(self):
        if self._loaded:
            return

        if isinstance(self.annotation_file, COCO):
            self.coco = self.annotation_file
        else:
            self.coco = COCO(self.annotation_file)
        self.ids = sorted(self.coco.imgs.keys())
        self._loaded = True

    def __getitem__(self, index: int) -> tuple[np.ndarray, ImageMetadata]:
        self.full_init()
        id = self.ids[index]
        image = self._load_image(id)
        image, name = self.transform(image)
        return image, ImageMetadata(id, name)

    def __len__(self) -> int:
        self.full_init()
        return len(self.ids)


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


# ====================================================================


def empty_collate_fn(batch):
    return batch


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    CONSOLE.print(f"Initialized worker process: id={worker_id}, seed={seed}")


def process_coco_dataset(
    executor: ThreadPoolExecutor,
    images_dir: Path,
    annotation_file: Path,
    tgt_images_dir: Path,
    tgt_annotation_file: Path,
    transform_fn: Callable[..., tuple[np.ndarray, str]],
    num_workers: int = 8,
):
    dataset = DegradedCOCODataset(images_dir, annotation_file, transform_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=empty_collate_fn,
        worker_init_fn=worker_init_fn,
        # avoid potential deadlocks between parent and workers (OpenCV, Numpy, using locks etc.)
        # due to copy-on-write (fork)
        multiprocessing_context="spawn",
    )

    tasks = []
    coco = COCO(annotation_file)
    for batch_inputs in track(
        dataloader,
        description=f"Submitting {annotation_file.stem} subset to saving workers...",
        console=CONSOLE,
    ):
        batch_inputs: list[tuple[np.ndarray, ImageMetadata]]
        image, metadata = batch_inputs[0]

        # determine saved place & update coco metadata
        coco_metadata = coco.loadImgs(metadata.id)[0]
        file_name = Path(tgt_images_dir.stem) / coco_metadata["file_name"]
        # use png format to avoid potential lossy compression
        file_name = file_name.with_suffix(".png")
        coco_metadata["file_name"] = file_name.as_posix()
        coco_metadata["degradation"] = metadata.name

        # submit a saving task to worker
        out_path = images_dir / file_name
        tasks.append(executor.submit(save_image, out_path, image))

    # save degraded version of annotation
    with open(tgt_annotation_file, "w") as f:
        json.dump(coco.dataset, f)

    with CONSOLE.status("[yellow]Waiting for all workers to finish..."):
        wait(tasks)
        for task in tasks:
            task.result()  # raises if error


def make_degraded_coco_dataset(
    root_dir: Union[str, Path],
    tgt_images_name: str = "degraded",
    annotation_files: list[str] = ["val.json", "test.json"],
    seed: int = 2025,
    num_workers: int = 8,
    identity: bool = False,
):
    seed_everything(seed % 2**32)
    CONSOLE.print(f"Initialized main process: seed={seed}")

    root_dir = Path(root_dir)
    images_dir = root_dir / "images"
    annotations_dir = root_dir / "annotations"

    tgt_images_dir = images_dir / tgt_images_name
    transform_fn = partial(apply_random_degradation, identity=identity)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for annotation_file in annotation_files:
            annotation_path = annotations_dir / annotation_file
            if not annotation_path.exists():
                continue

            tgt_annotation_path = annotation_path.with_stem(
                f"{tgt_images_dir.stem}_{annotation_path.stem}"
            )
            process_coco_dataset(
                executor,
                images_dir,
                annotation_path,
                tgt_images_dir,
                tgt_annotation_path,
                transform_fn,
                num_workers=num_workers,
            )

    CONSOLE.print(f"[green]Saved degraded images in {tgt_images_dir}.")
    CONSOLE.print(
        "[green]Saved degraded annotations in {}.".format(
            annotations_dir / f"{tgt_images_name}_xxx.json"
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a degraded version of COCO dataset (offline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", type=str, help="Root folder of COCO dataset")
    parser.add_argument(
        "--tgt-images-name",
        type=str,
        default="degraded",
        help="Name of images folder for storing degraded images",
    )
    parser.add_argument(
        "--annotation-files",
        type=str,
        nargs="+",
        default=["val.json", "test.json"],
        help="Apply degradation augmentation to specific subsets",
    )
    parser.add_argument(
        "--identity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a skip connection to the augmentation list. In common case, only apply offline augmentation to val, test subsets. So, the identity is not added by default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Reproducibility for degradation augmentation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of workers(processes) for loading and processing images.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    make_degraded_coco_dataset(**vars(args))
