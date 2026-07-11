import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from rtdetrv2.core import register
from rtdetrv2.data import CocoDetection as ORIGINAL_CocoDetection

from ....utils import RandomContext


@register()
class CocoDetectionv2(ORIGINAL_CocoDetection):
    def __init__(
        self,
        *args,
        degradations: Optional[Union[list[str], set[str]]] = None,
        image_ids: Optional[Union[list[str], set[str]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if degradations is not None:
            degradations = set(degradations)

            def _filter(id: int):
                return self.coco.imgs[id].get("degradation", "identity") in degradations

            # only return images with specific degradations
            self.ids = list(filter(_filter, self.ids))
            print(f"Specified degradations: {', '.join(degradations)}")

        if image_ids is not None:
            image_ids = set(image_ids)
            assert len(image_ids - self.coco.imgs.keys()) == 0, "Unknown image ids."

            def _filter(id: int):
                return id in image_ids

            # only return specific images
            self.ids = list(filter(_filter, self.ids))

    def load_item(self, idx: int):
        image, target = super().load_item(idx)

        metadata = self.coco.load_imgs([target["image_id"].item()])[0]
        target["image_path"] = os.path.join(self.root, metadata["file_name"])
        target["degradation"] = metadata.get("degradation", "identity")
        return image, target


@register()
class RobustCocoDetection(CocoDetectionv2):
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
        self._robust_transforms = robust_transforms

    def __getitem__(self, idx: int):
        img, target = self.load_item(idx)
        clear_img, clear_target = img.copy(), deepcopy(target)

        img, target, _ = self._robust_transforms(img, target, self)
        if self._transforms is not None:
            with RandomContext():
                clear_img, clear_target, _ = self._transforms(
                    clear_img, clear_target, self
                )
            img, target, _ = self._transforms(img, target, self)
        return torch.stack([img, clear_img]), target
