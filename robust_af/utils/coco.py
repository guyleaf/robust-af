from typing import Optional


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
