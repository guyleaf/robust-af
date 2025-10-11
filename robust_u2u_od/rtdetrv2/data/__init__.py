# ruff: noqa: F401
from . import dataset, transforms
from .dataloader import RobustBatchImageCollateFuncion

__all__ = list(globals().keys())

from PIL import Image as _Image

# Limit to around a gigabyte for a 24-bit (3 bpp) image
# Sometimes RandomZoomOut + RandomIoUCrop will exceed 2 * MAX_IMAGE_PIXELS.
_Image.MAX_IMAGE_PIXELS = int(1024 * 1024 * 1024 // 3)
