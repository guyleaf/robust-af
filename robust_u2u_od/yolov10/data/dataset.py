import torch
from ultralytics.data import YOLODataset as ORIGINAL_YOLODataset
from ultralytics.data.augment import Compose
from ultralytics.utils import LOGGER

from .augment import Degradation


class YOLODataset(ORIGINAL_YOLODataset):
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        transforms = super().build_transforms(hyp)
        if hyp.degradation["enabled"]:
            transforms = Compose(
                [
                    Degradation(
                        self,
                        transforms,
                        seed=hyp.degradation["seed"],
                        identity=hyp.degradation["identity"],
                        ignored_degradations=hyp.degradation["ignored_degradations"],
                    )
                ]
            )
            LOGGER.info("Degradation transform enabled!")
        return transforms

    # TODO: support mosaic & mixup with multiple degraded images?

    @staticmethod
    def collate_fn(batch: list[dict]):
        """Collates data samples into batches."""
        new_batch = ORIGINAL_YOLODataset.collate_fn(batch)
        if "clear_img" in batch[0]:
            clear_imgs = [sample["clear_img"] for sample in batch]
            new_batch["clear_img"] = torch.stack(clear_imgs, 0)
        return new_batch
