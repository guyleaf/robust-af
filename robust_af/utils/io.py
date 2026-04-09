import json
import mimetypes
from pathlib import Path
from typing import Callable, Union

_MIME_CHECKER = mimetypes.MimeTypes()


def is_image(path: Path):
    mime_type = _MIME_CHECKER.guess_type(path)[0]
    return mime_type is not None and mime_type.startswith("image")


def is_video(path: Path):
    mime_type = _MIME_CHECKER.guess_type(path)[0]
    return mime_type is not None and mime_type.startswith("video")


def collect_files(path: Union[str, Path], filter_fn: Callable[[Path], bool]):
    path = Path(path)
    if path.is_dir():
        images = filter(filter_fn, path.rglob("*.*"))
    else:
        images = [path]
    return sorted(images)


def collect_images(
    path: Union[str, Path], filter_fn: Callable[[Path], bool] = is_image
) -> list[Path]:
    """Collect images from path folder in ascending order"""
    return collect_files(path, filter_fn)


def collect_videos(
    path: Union[str, Path], filter_fn: Callable[[Path], bool] = is_video
) -> list[Path]:
    """Collect videos from path folder in ascending order"""
    return collect_files(path, filter_fn)


def dump_json(path: Path, obj, **kwargs):
    with open(path, "w") as f:
        json.dump(obj, f, **kwargs)
