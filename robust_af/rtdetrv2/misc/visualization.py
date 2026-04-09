from typing import Optional

import PIL.Image as Image
import torch
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes


def show_prediction(
    image: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    label2name: Optional[dict[int, str]] = None,
    out_file: Optional[str] = None,
    show: bool = False,
):
    if label2name is None:
        label2name = {}

    image = F.to_dtype_image(image, dtype=torch.uint8, scale=True)
    labels = prediction["labels"].tolist()
    scores = prediction["scores"].tolist()

    texts = []
    for label, score in zip(labels, scores):
        text: str = label2name.get(label, str(label))
        text = f"{text}: {score:.2f}"
        texts.append(text)

    annotated_image = draw_bounding_boxes(
        image, prediction["boxes"], labels=texts, width=3
    )

    im = Image.fromarray(annotated_image.permute(1, 2, 0).cpu().numpy(), mode="RGB")
    if out_file is not None:
        im.save(out_file)
    if show:
        im.show()
    return annotated_image
