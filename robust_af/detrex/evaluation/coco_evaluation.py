from detectron2.evaluation import COCOEvaluator as ORIGINAL_COCOEvaluator


class COCOEvaluator(ORIGINAL_COCOEvaluator):
    """
    Differences from original COCO evaluator:
        1. Support returning COCOeval instance after evaluate() for further analysis, such as t-SNE vis with failed cases.

    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def _eval_predictions(self, predictions: list[dict], img_ids: list[int] = None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        if img_ids is None:
            # only evaluate predictions on their images instead of whole dataset
            img_ids = list(set(x["image_id"] for x in predictions))
        super()._eval_predictions(predictions, img_ids=img_ids)
