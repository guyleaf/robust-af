import glob
import inspect
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

from torch.utils.data import Dataset
from ultralytics.data import YOLODataset as ORIGINAL_YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.dataset import DATASET_CACHE_VERSION, load_dataset_cache_file
from ultralytics.data.utils import HELP_URL, IMG_FORMATS, get_hash, img2label_paths
from ultralytics.utils import LOCAL_RANK, LOGGER, TQDM

from ...utils import RandomContext
from .augment import (
    Degradation,
    Identity,
    MultiBranch,
    robust_v8_transforms,
    v8_transforms,
)


class YOLODataset(ORIGINAL_YOLODataset):
    def _filter_images_with_degradations(
        self, metadata: list[tuple[str, str]], degradations: Union[set[str], list[str]]
    ):
        degradations = set(degradations)
        num_before = len(metadata)

        # read COCO annotation to get info?
        metadata = [(f, d) for f, d in metadata if d in degradations]

        num_after = len(metadata)
        LOGGER.info(
            "Removed {} images with not in specific degradations. {} images left.".format(
                num_before - num_after, num_after
            )
        )
        return metadata

    def _get_cache_path(self):
        # split cache file by degradations to avoid getting wrong evaluation
        # while running multiple experiments at the same time
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        degradations = self.data.get("degradations", None)
        if degradations is not None:
            cache_path = cache_path.with_stem(
                f"{cache_path.stem}_degraded_" + "_".join(degradations)
            )
        return cache_path

    def get_img_files(self, img_path: Union[str, list[str]]):
        """Read and Filter image files."""
        try:
            metadata = []  # image metadata
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    metadata = [
                        (f, "identity")
                        for f in glob.iglob(str(p / "**" / "*.*"), recursive=True)
                    ]
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        for x in t:
                            x = x.split(" ")
                            metadata.append(
                                (
                                    # local to global path
                                    x[0].replace("./", parent)
                                    if x[0].startswith("./")
                                    else x[0],
                                    x[1] if len(x) > 1 else "identity",
                                    # ignore other metadata if exists
                                )
                            )
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")

            metadata = sorted(
                (
                    (f.replace("/", os.sep), d)
                    for f, d in metadata
                    if f.split(".")[-1].lower() in IMG_FORMATS
                ),
                key=lambda x: x[0],
            )
            assert metadata, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(
                f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}"
            ) from e

        degradations = self.data.get("degradations", None)
        if degradations is not None:
            metadata = self._filter_images_with_degradations(metadata, degradations)

        if self.fraction < 1:
            num_elements_to_select = round(len(metadata) * self.fraction)
            metadata = random.sample(metadata, num_elements_to_select)
        return [f for f, _ in metadata]

    def get_labels(self):
        """
        Returns dictionary of labels for YOLO training.
        Fix cache_path to avoid using the same cache file in per-degradation experiemnts.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = self._get_cache_path()
        try:
            cache, exists = (
                load_dataset_cache_file(cache_path),
                True,
            )  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(
                self.label_files + self.im_files
            )  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop(
            "results"
        )  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(
                f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = (
            (len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels
        )
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(
                f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        return labels

    def _build_degradation_transform(self, hyp: SimpleNamespace):
        if hyp.degradation["enabled"]:
            LOGGER.info("Degradation transform enabled!")
            sig = inspect.signature(Degradation)
            kwargs = {k: v for k, v in hyp.degradation.items() if k in sig.parameters}
            return Degradation(**kwargs)
        else:
            return Identity()

    def build_transforms(self, hyp: Optional[SimpleNamespace] = None):
        """Builds and appends transforms to the list."""
        assert hyp is not None
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(
                self, self.imgsz, hyp, degraded=hyp.degraded_augs
            )
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )

        if self.augment:
            return transforms
        if hyp.degradation["always"]:
            transforms = Compose([self._build_degradation_transform(hyp), transforms])
        return transforms


class RobustYOLODataset(YOLODataset):
    """Generate a degraded-clear pair from a single image."""

    def build_transforms(self, hyp: Optional[SimpleNamespace] = None):
        """Builds and appends transforms to the list."""
        assert hyp is not None
        degradation_transform = self._build_degradation_transform(hyp)

        formatter = Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            return_obb=self.use_obb,
            batch_idx=True,
            mask_ratio=hyp.mask_ratio,
            mask_overlap=hyp.overlap_mask,
            bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
        )
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transform, robust_transform = robust_v8_transforms(
                self, self.imgsz, hyp, degradation_transform
            )
            transform.append(formatter)
            robust_transform.append(formatter)
        else:
            # support calculating loss in validation/testing
            transform = Compose(
                [
                    LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
                    formatter,
                ]
            )
            robust_transform = Compose([degradation_transform, *transform.tolist()])

        transforms = Compose(
            [
                MultiBranch(
                    "degraded",
                    reproduce_randomness=True,
                    clear=transform,
                    degraded=robust_transform,
                )
            ]
        )
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]):
        """Collates data samples into batches."""
        new_batch = YOLODataset.collate_fn(batch)
        if "clear" in batch[0]:
            clear_batch = [sample["clear"] for sample in batch]
            clear_batch = YOLODataset.collate_fn(clear_batch)
            new_batch["clear"] = clear_batch
        return new_batch


class RobustPairedYOLODataset(Dataset):
    """Make a degraded-clear pair from two YOLODatasets."""

    def __init__(
        self, dataset: ORIGINAL_YOLODataset, robust_dataset: ORIGINAL_YOLODataset
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.robust_dataset = robust_dataset
        assert isinstance(dataset, ORIGINAL_YOLODataset) and isinstance(
            robust_dataset, ORIGINAL_YOLODataset
        )
        assert len(dataset) == len(robust_dataset), "Two datasets should be paired."

    def __getitem__(self, index: int):
        with RandomContext():
            label = self.robust_dataset[index]
        label["clear"] = self.dataset[index]
        return label

    def __len__(self):
        return len(self.robust_dataset)

    # NO TEST: because we only use it in val/test evaluation
    def close_mosaic(self, hyp: SimpleNamespace):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        self.robust_dataset.close_mosaic(hyp)
        self.dataset.close_mosaic(hyp)

    @staticmethod
    def collate_fn(batch: list[dict]):
        """Collates data samples into batches."""
        return RobustYOLODataset.collate_fn(batch)
