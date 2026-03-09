import os
from typing import Optional, TypeVar

from sklearn.model_selection import train_test_split

T = TypeVar("T")


def filter_video_with_video_list(video_list: list[str]):
    def filter_(video_name: str):
        video_name = os.path.basename(video_name)
        video_name = os.path.splitext(video_name)[0]
        return video_name in video_list

    return filter_


def split_into_train_val_test(
    x: list[T], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[T]]:
    if test_ratio < 1.0:
        train, test = train_test_split(x, test_size=test_ratio, random_state=seed)
        ratio_remaining = 1.0 - test_ratio
        val_ratio = val_ratio / ratio_remaining
        if val_ratio < 1.0:
            train, val = train_test_split(train, test_size=val_ratio, random_state=seed)
        else:
            val = train
            train = []
    else:
        test = x
        train = val = []

    return {"train": train, "val": val, "test": test}


def split_into_train_val(
    x: list[T], val_ratio: float, seed: int, labels: Optional[list] = None
) -> dict[str, list[T]]:
    if val_ratio < 1.0:
        train, val = train_test_split(
            x, test_size=val_ratio, random_state=seed, stratify=labels
        )
    else:
        val = x
        train = []

    return {"train": train, "val": val}


def format_coco_image(
    id: int,
    file_name: str,
    h: int,
    w: int,
    license: Optional[int] = None,
    caption: Optional[str] = None,
    weight: Optional[float] = None,
):
    image = dict(id=id, file_name=file_name, height=h, width=w, license=license)
    if caption is not None:
        weight = weight if weight is not None else 1.0
        image["caption"] = caption
        image["weight"] = weight
    return image


def format_coco_frame(
    id: int,
    file_name: str,
    h: int,
    w: int,
    video_id: int,
    frame_id: int,
    license: Optional[int] = None,
):
    return dict(
        id=id,
        file_name=file_name,
        height=h,
        width=w,
        license=license,
        video_id=video_id,
        frame_id=frame_id,
    )


def format_coco_video(id: int, name: str):
    return dict(id=id, name=name)


def format_coco_annotation(
    id: int,
    image_id: int,
    category_id: int,
    x: int,
    y: int,
    w: int,
    h: int,
    attributes: dict = {},
):
    result = dict(
        id=id,
        image_id=image_id,
        category_id=category_id,
        area=w * h,
        bbox=[x, y, w, h],
        iscrowd=0,
        attributes=attributes,
    )
    return result
