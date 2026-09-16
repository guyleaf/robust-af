import logging
from typing import Optional, Union

import torch.utils.data as torchdata
from detectron2.data import (
    MetadataCatalog,
    print_instances_class_histogram,
)
from detectron2.data import (
    get_detection_dataset_dicts as detectron2_get_detection_dataset_dicts,
)


def filter_images_with_image_ids(
    dataset_dicts: list[dict], image_ids: Union[set[int], list[int]]
):
    image_ids = set(image_ids)
    dataset_dicts = [x for x in dataset_dicts if x["image_id"] in image_ids]

    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(f"Select {num_after} images with specific image ids.")
    return dataset_dicts


def filter_images_with_degradations(
    dataset_dicts: list[dict], degradations: Union[set[str], list[str]]
):
    degradations = set(degradations)
    num_before = len(dataset_dicts)

    dataset_dicts = [x for x in dataset_dicts if x["degradation"] in degradations]

    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with not in specific degradations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def get_detection_dataset_dicts(
    names: Union[str, list[str]],
    filter_empty: bool = True,
    min_keypoints: int = 0,
    proposal_files: list[str] = None,
    check_consistency: bool = True,
    degradations: Optional[Union[set[str], list[str]]] = None,
    image_ids: Optional[Union[set[int], list[int]]] = None,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names

    dataset_dicts = detectron2_get_detection_dataset_dicts(
        names,
        filter_empty=filter_empty,
        min_keypoints=min_keypoints,
        proposal_files=proposal_files,
        check_consistency=check_consistency,
    )
    if isinstance(dataset_dicts, torchdata.Dataset):
        return dataset_dicts

    has_instances = "annotations" in dataset_dicts[0]
    if degradations is not None:
        dataset_dicts = filter_images_with_degradations(dataset_dicts, degradations)
    if image_ids is not None:
        dataset_dicts = filter_images_with_image_ids(dataset_dicts, image_ids)

    if check_consistency and has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts
