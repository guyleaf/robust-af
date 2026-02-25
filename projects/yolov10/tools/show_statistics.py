import argparse

from robust_af.yolov10.models import RobustYOLOv10, YOLOv10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get the statistics of model, such as FLOPs, Parameters...",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model", type=str, help="Config or Checkpoint (.pt) of the model"
    )
    parser.add_argument(
        "--robust",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the robust version of the model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.robust:
        model_cls = RobustYOLOv10
    else:
        model_cls = YOLOv10

    model = model_cls(args.model)
    model.model.model[-1].export = True
    model.model.model[-1].format = "onnx"
    del model.model.model[-1].cv2
    del model.model.model[-1].cv3
    model.fuse()
