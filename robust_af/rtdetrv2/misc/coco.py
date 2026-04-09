import torch
from torchvision.ops import box_convert


def convert_prediction_to_coco(prediction: dict[str, torch.Tensor], image_id: int):
    """Converts bounding box predictions to COCO detection format.

    Modified from rtdetrv2.data.CocoEvaluator.prepare_for_coco_detection.

    Args:
        predictions (Dict): Dictionary mapping image ids to prediction dicts. Each prediction dict must contain:
            - "boxes": Tensor of shape [N, 4] (x1, y1, x2, y2)
            - "scores": torch.Tensor of shape [N] of length N
            - "labels": torch.Tensor of shape [N] of length N

    Returns:
        List[Dict]: List of detection results in COCO format.
    """
    boxes = prediction["boxes"]
    boxes = box_convert(boxes, "xyxy", "xywh").tolist()
    scores = prediction["scores"].tolist()
    labels = prediction["labels"].tolist()

    coco_results = [
        {
            "image_id": image_id,
            "category_id": int(labels[k]),
            "bbox": box,
            "score": scores[k],
        }
        for k, box in enumerate(boxes)
    ]
    return coco_results
