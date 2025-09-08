from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from rtdetrv2.core import register
from rtdetrv2.data import CocoDetection as ORIGINAL_CocoDetection
from rtdetrv2.data import Compose, EmptyTransform

from ....utils import RandomContext


@register()
class RobustCocoDetection(ORIGINAL_CocoDetection):
    __inject__ = ["transforms", "robust_transforms"]

    def __init__(
        self,
        img_folder: Union[str, Path],
        ann_file: str,
        transforms: Optional[nn.Module],
        robust_transforms: nn.Module,
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
    ):
        super().__init__(
            img_folder, ann_file, transforms, return_masks, remap_mscoco_category
        )

        if self._transforms is None:
            self._transforms = EmptyTransform()
        self._robust_transforms = Compose([robust_transforms, self._transforms])

    def __getitem__(self, idx: int):
        img, target = self.load_item(idx)
        with RandomContext():
            clear_img, _, _ = self._transforms(img.copy(), deepcopy(target), self)
        img, target, _ = self._robust_transforms(img, target, self)
        return torch.stack([img, clear_img]), target
