from copy import deepcopy
from types import SimpleNamespace
from typing import Optional

from ultralytics.data import YOLODataset as ORIGINAL_YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.utils import LOGGER

from .augment import Degradation, Identity, MultiBranch, robust_v8_transforms


class YOLODataset(ORIGINAL_YOLODataset):
    def _build_degradation_transform(self, hyp: SimpleNamespace):
        if hyp.degradation["enabled"]:
            LOGGER.info("Degradation transform enabled!")
            return Degradation(
                seed=hyp.degradation["seed"],
                identity=hyp.degradation["identity"],
                ignored_degradations=hyp.degradation["ignored_degradations"],
            )
        else:
            return Identity()

    def build_transforms(self, hyp: Optional[SimpleNamespace] = None):
        """Builds and appends transforms to the list."""
        assert hyp is not None
        transforms = super().build_transforms(hyp)
        if self.augment:
            return transforms
        # testing only
        transforms = Compose(
            [
                self._build_degradation_transform(hyp),
                transforms,
            ]
        )
        return transforms


class RobustYOLODataset(YOLODataset):
    def _build_degradation_transform(self, hyp: SimpleNamespace):
        # use diff seed in training stage to avoid using the same random sequence in val/test stage
        if hyp.degradation["seed"] is not None and self.augment:
            hyp = deepcopy(hyp)
            hyp.degradation["seed"] += 666
        return super()._build_degradation_transform(hyp)

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
