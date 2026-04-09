from pathlib import Path
from typing import Optional

import torch

from .io import dump_json


def dump_coco_images(
    folder: Path,
    images: dict[int, torch.Tensor],
    file_name: str = "images.pt",
):
    # validate features follow [C, H, W]
    for image_id, image in images.items():
        assert isinstance(image_id, int) and image_id > 0
        assert isinstance(image, torch.Tensor) and image.dim() == 3

    folder.mkdir(parents=True, exist_ok=True)
    torch.save(images, folder / file_name)


def dump_coco_features(
    folder: Path,
    features: dict[int, dict[str, torch.Tensor]],
    file_name: str = "features.pt",
):
    # validate features follow [C, H, W]
    for image_id, image_features in features.items():
        assert isinstance(image_id, int) and image_id > 0
        for name, feature in image_features.items():
            assert isinstance(name, str)
            assert isinstance(feature, torch.Tensor) and feature.dim() == 3

    folder.mkdir(parents=True, exist_ok=True)
    torch.save(features, folder / file_name)


def dump_coco_results(
    folder: Path, predictions: list[dict], file_name: str = "coco_results.json"
):
    # TODO: use dataclass to format structure?
    # validate prediction format follow COCO results
    for pred in predictions:
        assert isinstance(pred["image_id"], int) and pred["image_id"] > 0
        assert isinstance(pred["category_id"], int) and pred["category_id"] > 0
        assert isinstance(pred["bbox"], list) and all(
            isinstance(item, float) for item in pred["bbox"]
        )
        assert isinstance(pred["score"], float) and pred["score"] >= 0

    folder.mkdir(parents=True, exist_ok=True)
    dump_json(folder / file_name, predictions)


def dump_metadata(folder: Path, metadata: dict, file_name: str = "metadata.json"):
    dump_json(folder / file_name, metadata)


def dump_results(
    folder: Path,
    predictions: list[dict],
    backbone_features: dict[int, dict[str, torch.Tensor]],
    images: Optional[dict[int, torch.Tensor]] = None,
    features: Optional[dict[int, dict[str, torch.Tensor]]] = None,
    metadata: dict = {},
):
    for pred in predictions:
        image_id = pred["image_id"]
        assert image_id in backbone_features
        if images is not None:
            assert image_id in images
        if features is not None:
            assert image_id in features

    dump_coco_features(folder, backbone_features, file_name="backbone_features.pt")
    if images is not None:
        dump_coco_images(folder, images)
    if features is not None:
        dump_coco_features(folder, features)
    dump_coco_results(folder, predictions)
    dump_metadata(folder, metadata)
