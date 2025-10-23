import argparse

from rich import print

from robust_au_od.yolov10.models import RobustYOLOv10, YOLOv10
from robust_au_od.yolov10.utils import is_huggingface_hub_model, load_global_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("cfg", type=str, help="Path to the config.")
    parser.add_argument(
        "--robust",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the robust version of YOLOv10.",
    )
    # parser.add_argument(
    #     "opts",
    #     help="""
    #         Modify config options at the end of the command. For Yacs configs, use
    #         space-separated "PATH.KEY VALUE" pairs.
    #         For python-based LazyConfig, use "path.key=value".
    #     """.strip(),
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = load_global_cfg(args.cfg)

    mode = cfg.get("mode")
    task = cfg.get("task")
    model = cfg.get("model")
    pretrained = cfg.get("pretrained")

    # TODO: move robust arg to config
    if args.robust:
        model_cls = RobustYOLOv10
    else:
        model_cls = YOLOv10

    if is_huggingface_hub_model(model):
        model = model_cls.from_pretrained(model, task=task)
    else:
        model = model_cls(model, task=task)

    if isinstance(pretrained, str):
        model.load(pretrained)

    # Run command in python
    getattr(model, mode)(**cfg)

    print("[bold bright_green]Success!")
