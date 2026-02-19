import copy
import logging
from typing import Union

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class RobustDetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the format used by Robust DETR for pair training.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Make a copy for clear version
    3. Applies transforms (shared) to the clear image and annotations
    4. Applies transforms (shared) and robust transforms to the image and annotations
    5. Prepare image and annotations to Tensor and `Instances`

    Modifed from DatasetMapper and DetrDatasetMapper.

    Args:
        is_train: whether it's used in training or inference
        augmentations: a list of augmentations or deterministic transforms to apply
        image_format: an image format supported by :func:`detection_utils.read_image`.
        use_instance_mask: whether to process instance segmentation annotations, if available
        instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
            masks into this format.
        recompute_boxes: whether to overwrite bounding box annotations
            by computing tight bounding boxes from instance mask annotations.
        identity: whether add identity (skip connection) to degradations.
        ignored_degradations: a list of ignored degradations which won't be used.
    """

    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: list[Union[T.Transform, T.Augmentation]],
        robust_augmentations: list[Union[T.Transform, T.Augmentation]],
        image_format: str = "RGB",
        use_instance_mask: bool = False,
        instance_mask_format: str = "polygon",
        recompute_boxes: bool = False,
    ):
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        self.augmentations = T.AugmentationList(augmentations)
        self.robust_augmentations = T.AugmentationList(robust_augmentations)
        self.image_format = image_format
        self.is_train = is_train

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[RobustDetrDatasetMapper] Augmentations used in {mode}: {self.augmentations}"
        )

    def _apply_augs(self, augmentations: T.AugmentationList, image: np.ndarray):
        aug_input = T.AugInput(image)
        transforms = augmentations(aug_input)
        image = aug_input.image
        return image, transforms

    def __call__(self, dataset_dict: dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # 1.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        ori_image = utils.read_image(
            dataset_dict["file_name"], format=self.image_format
        )
        utils.check_image_size(dataset_dict, ori_image)

        # 2. prepare clear version
        clear_dataset_dict = dataset_dict["clear"] = copy.deepcopy(dataset_dict)
        # we rely on degraded annotations. so, we don't need this.
        clear_dataset_dict.pop("annotations", None)

        # 3. augment clear image
        image, transforms = self._apply_augs(self.augmentations, ori_image)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        clear_dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # 4. augment image
        image, _ = self._apply_augs(self.robust_augmentations, ori_image)
        assert image.shape[:2] == ori_image.shape[:2]
        image = transforms.apply_image(image)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # 5.
        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            image_shape = image.shape[:2]  # h, w
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)

            dataset_dict["instances"] = instances
            clear_dataset_dict["instances"] = copy.deepcopy(instances)
        return dataset_dict
