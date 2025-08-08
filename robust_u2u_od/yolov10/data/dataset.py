from ultralytics.data import YOLODataset as ORIGINAL_YOLODataset
from ultralytics.data.augment import Compose
from ultralytics.utils import LOGGER

from .augment import Degradation


class YOLODataset(ORIGINAL_YOLODataset):
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        transforms = super().build_transforms(hyp)
        if "degradation" in hyp and hyp.degradation.enabled:
            transforms = Compose(
                [
                    Degradation(
                        self,
                        transforms,
                        seed=hyp.degradation.seed,
                        identity=hyp.degradation.identity,
                        ignored_degradations=hyp.degradation.ignored_degradations,
                    )
                ]
            )
            LOGGER.info("Degradation transform enabled!")
        return transforms

    # TODO: support mosaic & mixup with multiple degraded images?
