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
