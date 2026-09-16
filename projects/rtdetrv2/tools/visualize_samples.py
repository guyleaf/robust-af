import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from rtdetrv2.core import YAMLConfig, yaml_utils

from robust_af.rtdetrv2.apis import Inferencer


def draw(
    image: Image.Image,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    thrh: float = 0.5,
    label2name: Optional[dict[int, str]] = None,
):
    if label2name is None:
        label2name = {}

    draw = ImageDraw.Draw(image)

    if scores is not None:
        labels = labels[scores > thrh]
        boxes = boxes[scores > thrh]
        scores = scores[scores > thrh]
    else:
        scores = torch.full_like(labels, -1)

    font = ImageFont.load_default()
    for label, box, score in zip(labels, boxes, scores):
        box = box.tolist()
        label = label.item()
        score = score.item()
        label = label2name.get(label, label)

        draw.rectangle(box, outline="red", width=2)

        text = f"{label}"
        if score != -1:
            text += f" {score:.2f}"

        bbox = draw.textbbox((box[0] + 2, box[1]), text, font=font)
        draw.rectangle(bbox, fill="blue")
        draw.text((box[0] + 2, box[1]), text=text, font=font, fill="white")

    return image


def infer_dataset(cfg: YAMLConfig, args: argparse.Namespace):
    inferencer = Inferencer(cfg)
    generator = inferencer(
        cfg,
        # shuffle=args.shuffle,
        # seed=cfg.seed,
        max_num_samples=args.max_num_samples,
        image_ids=args.image_ids,
    )

    out_dir: Path = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pred, sample in generator:
        image_path = Path(sample["image_path"])
        image_id: int = sample["image_id"].item()

        image = Image.open(image_path).convert("RGB")
        image = draw(
            image, **pred, thrh=args.score_threshold, label2name=sample["category2name"]
        )
        out_file = out_dir / f"{image_id:05d}.png"
        image.save(out_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize predictions for demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config_file", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory of t-SNE results. If None, use work_dir.",
    )
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=100,
        help="The maximum number of samples.",
    )
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specify image ids for inference.",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Score threshold to filter invalid predictions.",
    )

    # parser.add_argument(
    #     "--seed", type=int, default=2025, help="Make randomness deterministic"
    # )
    # parser.add_argument(
    #     "--shuffle",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="Shuffle dataset before dumping.",
    # )
    parser.add_argument(
        "-u", "--update", nargs="+", default=[], help="update yaml config"
    )

    args = parser.parse_args()
    assert 1 >= args.score_threshold > 0
    if args.image_ids is not None:
        args.max_num_samples = max(args.max_num_samples, len(args.image_ids))
    return args


if __name__ == "__main__":
    args = parse_args()
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict["resume"] = args.checkpoint
    # update_dict["seed"] = args.seed
    update_dict["print_method"] = "rich"

    cfg = YAMLConfig(args.config_file, **update_dict)

    out_dir = Path(cfg.output_dir)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir) / out_dir.name

    out_dir = out_dir / "vis"
    cfg.output_dir = args.out_dir = out_dir.resolve().as_posix()

    cfg.save()
    infer_dataset(cfg, args)
