from pathlib import Path
from typing import Union

import PIL.Image as Image
import torch
from torchvision.transforms.v2 import functional as F


def save_image(image: torch.Tensor, out_file: Union[str, Path]):
    image = F.to_dtype_image(image, dtype=torch.uint8, scale=True)

    im = Image.fromarray(image.permute(1, 2, 0).cpu().numpy(), mode="RGB")
    im.save(out_file)
