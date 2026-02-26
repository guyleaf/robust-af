import argparse

from robust_af.utils import parameter_count_table
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
    parser.add_argument(
        "--detailed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the details of parameter counts",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.robust:
        model_cls = RobustYOLOv10
    else:
        model_cls = YOLOv10

    # copied from yolov10/flops.py
    model = model_cls(args.model)
    model.model.model[-1].export = True
    model.model.model[-1].format = "onnx"
    del model.model.model[-1].cv2
    del model.model.model[-1].cv3
    model.fuse()

    # NOTE: BNs from base model are frozen in robust version. So, is_fused() == True is normal.
    # print again to avoid missing printing after fusing.
    model.info(detailed=args.detailed)

    if args.detailed:
        print("Parameter Count:\n" + parameter_count_table(model.model, max_depth=5))
