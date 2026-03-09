import mimetypes
import os
import tempfile
from pathlib import Path
from subprocess import PIPE
from typing import Callable, Union

import ffmpegio
from packaging.version import Version, parse

from .enum import VideoSamplingMethod

_MIME_CHECKER = mimetypes.MimeTypes()


def is_image(path: Path):
    mime_type = _MIME_CHECKER.guess_type(path)[0]
    return mime_type is not None and mime_type.startswith("image")


def collect_images(
    path: Union[str, Path], filter_fn: Callable[[Path], bool] = is_image
) -> list[Path]:
    """Collect images from path folder in ascending order"""
    path = Path(path)
    if path.is_dir():
        images = filter(filter_fn, path.rglob("*.*"))
    else:
        images = [path]

    # if as_posix:
    #     images = map(methodcaller("as_posix"), images)

    return sorted(images)


def get_ffmpeg_version():
    version = ffmpegio.ffmpeg_info()["version"]
    return parse(version.split("-")[0])


def choose_ffmpeg_vf(
    sample_method: VideoSamplingMethod = VideoSamplingMethod.SIMPLE,
    sample_interval: int = 5,
    subset: str = "train",
) -> str:
    sample_interval = max(sample_interval, 1)
    vf = "select"
    # if subset != "train":
    #     return vf

    if sample_method == VideoSamplingMethod.I_FRAME:
        vf += "='eq(pict_type\\,I)'"
    elif sample_interval > 1:
        if sample_method == VideoSamplingMethod.THUMBNAIL:
            vf = f"thumbnail={sample_interval}"
        else:
            vf += f"='not(mod(n\\,{sample_interval}))'"
    return vf


def extract_video_frames(
    file_name: str,
    out_dir: str,
    output_format: str = "%05d.png",
    **kwargs,
):
    _kwargs = dict(
        show_log=True,
        hwaccel_in="cuda",
        an=None,
        sn=None,
        frame_pts=True,
    )
    ffmpeg_version = get_ffmpeg_version()
    if ffmpeg_version >= Version("5.1"):
        _kwargs["fps_mode"] = "passthrough"
    else:
        _kwargs["vsync"] = "passthrough"

    # get pts of the fist and last frame
    nb_frames = count_video_frames(file_name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        _out_dir = os.path.join(tmp_dir, output_format)
        ffmpegio.transcode(
            file_name, _out_dir, vf=f"select='eq(n,0)+eq(n,{nb_frames - 1})'", **_kwargs
        )
        first_frame_i, last_frame_i = tuple(
            sorted(
                map(lambda frame: int(os.path.splitext(frame)[0]), os.listdir(tmp_dir))
            )
        )

    _kwargs.update(kwargs)
    _out_dir = os.path.join(out_dir, output_format)
    ffmpegio.transcode(file_name, _out_dir, **_kwargs)

    # WORKAROUND: some videos don't have pts timestamp
    # rename frames to start from zero
    # there are special cases which the first frame is not zero and the last frame is correct
    first_offset = first_frame_i
    last_offset = last_frame_i - (nb_frames - 1)
    if first_offset == 0 or last_offset == 0:
        return

    # if the pts of last frame is bigger than nb_frames
    # calibrate it to start from zero
    frames = os.listdir(out_dir)
    for frame in frames:
        frame_i = int(os.path.splitext(frame)[0])
        frame_i -= first_offset

        old_frame = os.path.join(out_dir, frame)
        new_frame = os.path.join(out_dir, output_format % frame_i)
        os.rename(old_frame, new_frame)


def count_video_frames(file_name: str) -> int:
    out = ffmpegio.probe.ffprobe(
        [
            "-v",
            "error",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            "-count_frames",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=nb_read_frames",
            file_name,
        ],
        stdout=PIPE,
        universal_newlines=True,
    ).stdout
    return int(out)
