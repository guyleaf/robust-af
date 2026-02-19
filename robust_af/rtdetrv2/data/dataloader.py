import random

import torch
import torch.nn.functional as F
from rtdetrv2.core import register
from rtdetrv2.data import BatchImageCollateFuncion


@register()
class RobustBatchImageCollateFuncion(BatchImageCollateFuncion):
    def _apply_batch_random_scale(self, images: torch.Tensor, targets: list[dict]):
        sz = random.choice(self.scales)

        B, N = images.shape[:2]
        # B, N, C, H, W => B * N, C, H, W
        images = images.reshape(-1, *images.shape[-3:])
        images = F.interpolate(images, size=sz)
        # B * N, C, H, W => B, N, C, H, W
        images = images.reshape(B, N, *images.shape[-3:])

        if "masks" in targets[0]:
            for tg in targets:
                tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")
            raise NotImplementedError("")
        return images, targets

    def __call__(self, items: list):
        images = torch.stack([x[0] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            images, targets = self._apply_batch_random_scale(images, targets)
        return images, targets
