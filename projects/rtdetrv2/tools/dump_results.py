import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import track
from rtdetrv2.core import BaseConfig, YAMLConfig, yaml_utils
from rtdetrv2.data.dataset._dataset import DetDataset
from rtdetrv2.misc import setup_seed
from torch.utils.data import DataLoader, Subset

from robust_af.rtdetrv2.apis import Inferencer
from robust_af.rtdetrv2.data.transforms import Degradation
from robust_af.rtdetrv2.misc import convert_prediction_to_coco, show_prediction
from robust_af.transforms import DEGRADATION_TRANSFORMS
from robust_af.utils import dump_json, dump_results

CONSOLE = Console()

TEST_TRANSFORMS = dict(
    type="Compose",
    ops=[
        dict(type=Degradation.__name__),
        dict(type="Resize", size=[640, 640]),
        dict(type="ConvertPILImage", dtype="float32", scale=True),
    ],
)


class DumpInferencer(Inferencer):
    """An inference for dumping features and predictions to COCO format"""

    def __init__(self, cfg: Union[str, Path, BaseConfig]):
        super().__init__(cfg, deploy=False)

        self.images: Optional[torch.Tensor] = None
        self.features: Optional[dict[str, torch.Tensor]] = None
        self.backbone_features: Optional[dict[str, torch.Tensor]] = None

        if (
            hasattr(self.model, "with_robust_image_module")
            and self.model.with_robust_image_module
        ):
            self.model.robust_image_module.register_forward_hook(self._save_images)
        if hasattr(self.model, "with_robust_module") and self.model.with_robust_module:
            self.model.robust_module.register_forward_hook(self._save_features)
        self.model.backbone.register_forward_hook(self._save_backbone_features)

    def _save_images(
        self,
        module: nn.Module,
        args: tuple,
        output: torch.Tensor,
    ):
        # [1, c, h, w]
        self.images = output

    def _save_features(
        self, module: nn.Module, args: tuple, output: list[torch.Tensor]
    ):
        # [1, c, h, w]
        self.features = {f"backbone_{i}": item for i, item in enumerate(output)}

    def _save_backbone_features(
        self, module: nn.Module, args: tuple, output: list[torch.Tensor]
    ):
        # [1, c, h, w]
        self.backbone_features = {
            f"backbone_{i}": item for i, item in enumerate(output)
        }

    def prepare_dataloaders(
        self,
        cfg: YAMLConfig,
        degradations: list[str],
        max_num_samples: int,
        name: str = "val_dataloader",
        shuffle: bool = False,
        seed: int = 2026,
    ):
        cfg = deepcopy(cfg)
        dataloader_cfg = cfg.yaml_cfg[name]
        dataloader_cfg["total_batch_size"] = 1

        dataloaders: dict[str, DataLoader] = {}
        for degradation in degradations:
            transforms = deepcopy(TEST_TRANSFORMS)

            # specify name of degradation
            aug = transforms["ops"][0]
            assert aug["type"] == Degradation.__name__
            aug["name"] = degradation
            aug["seed"] = seed

            dataloader_cfg["dataset"]["transforms"] = transforms
            # make shuffle determinitic
            setup_seed(seed)
            dataloaders[degradation] = self.prepare_dataloader(
                cfg, max_num_samples, name=name, shuffle=shuffle
            )

        # validate consistency across degradations
        indices = dataloaders[degradations[0]].dataset.indices
        for dataloader in dataloaders.values():
            assert dataloader.dataset.indices == indices
        return dataloaders

    def __call__(
        self,
        cfg: Union[str, Path, YAMLConfig],
        degradations: list[str],
        shuffle: bool = False,
        seed: int = 2026,
        max_num_samples: int = 100,
    ):
        """Infer images from dataloader

        Args:
            cfg (Union[str, Path, YAMLConfig]): Config. If it is a file path, load it and use the val dataloader.
            shuffle (bool, optional): Shuffle dataset before inference. Defaults to False.
            seed (int): The seed for randomness. Defaults to 2026.
            max_num_samples (int, optional): Total number of images to be inferred. Defaults to 100.

        Yields:
            predictions (dict[str, torch.Tensor]): The predictions of the sample. Labels are dataset categories.
            target (dict): The metadata of the sample.
        """
        # build dataloader
        if not isinstance(cfg, YAMLConfig):
            cfg = YAMLConfig(str(cfg))
        dataloaders = self.prepare_dataloaders(
            cfg, degradations, max_num_samples, shuffle=shuffle, seed=seed
        )

        # avoid any randomness
        setup_seed(seed)
        for name, dataloader in dataloaders.items():
            dataset = dataloader.dataset
            if isinstance(dataset, Subset):
                dataset = dataset.dataset
            assert isinstance(dataset, DetDataset)
            label2category = dataset.label2category
            category2name = dataset.category2name

            num_samples = len(dataloader.dataset)
            counter = 0
            for samples, targets in track(
                dataloader, description=f"Inference ({name})", console=CONSOLE
            ):
                targets: list[dict]
                preds = self.forward(samples, targets, label2category=label2category)

                for i, (pred, target) in enumerate(zip(preds, targets)):
                    target["image"] = samples[i]
                    target["degradation"] = name
                    target["category2name"] = category2name

                    target["backbone_features"] = {
                        k: v[i] for k, v in self.backbone_features.items()
                    }
                    if self.images is not None:
                        target["images"] = self.images[i]
                    if self.features is not None:
                        target["features"] = {k: v[i] for k, v in self.features.items()}

                    counter += 1
                    last_sample = counter == num_samples
                    yield pred, target, counter, last_sample


def main(cfg: YAMLConfig, args: argparse.Namespace):
    inferencer = DumpInferencer(cfg)
    generator = inferencer(
        cfg,
        args.degradations,
        shuffle=args.shuffle,
        seed=cfg.seed,
        max_num_samples=args.max_num_samples,
    )

    images = {}
    features = {}
    backbone_features = {}
    predictions = []
    out_dir: Path = Path(cfg.output_dir)
    vis_image_ids = set(args.vis_image_ids)
    for pred, sample, counter, last_sample in generator:
        image_id: int = sample["image_id"].item()
        name: str = sample["degradation"]
        images_i: Optional[torch.Tensor] = sample.get("images")
        features_i: Optional[dict[str, torch.Tensor]] = sample.get("features")
        backbone_features_i: dict[str, torch.Tensor] = sample["backbone_features"]

        backbone_features[image_id] = {
            k: v.cpu() for k, v in backbone_features_i.items()
        }
        if images_i is not None:
            images[image_id] = images_i.cpu()
        if features_i is not None:
            features[image_id] = {k: v.cpu() for k, v in features_i.items()}

        # convert pred to coco format
        coco_preds = convert_prediction_to_coco(pred, image_id)
        predictions.extend(coco_preds)

        # save results per degradation
        if last_sample:
            assert len(images) in (0, counter)
            assert len(features) in (0, counter)

            kwargs = dict(
                images=None if len(images) == 0 else images,
                features=None if len(features) == 0 else features,
                metadata=dict(degradation=name, num_samples=counter),
            )
            dump_results(out_dir / name, predictions, backbone_features, **kwargs)

            images = {}
            features = {}
            predictions = []

        if image_id in vis_image_ids:
            image_dir = out_dir / name / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            out_file = image_dir / f"{image_id:05d}.jpg"
            show_prediction(
                sample["image"],
                pred,
                label2name=sample["category2name"],
                out_file=out_file.as_posix(),
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump results for analysis",
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
        help="The maximum number of samples per degradation. "
        "Higher number need longer time to calculate.",
    )
    parser.add_argument(
        "--vis-image-ids",
        type=int,
        nargs="*",
        default=[],
        help="Specify image ids for dumping the input image for easier analysis.",
    )

    parser.add_argument(
        "--degradations",
        type=str,
        nargs="+",
        # default=[
        #     "snow",
        #     "fog",
        #     "rain",
        #     "iso_noise",
        #     "gaussian_noise",
        #     "color_jitter",
        #     "motion_blur",
        #     "identity",
        # ],
        default=list(DEGRADATION_TRANSFORMS.keys()),
        help="Evaluated degradations",
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Make randomness deterministic"
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle dataset before dumping.",
    )
    parser.add_argument(
        "-u", "--update", nargs="+", default=[], help="update yaml config"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict["resume"] = args.checkpoint
    update_dict["seed"] = args.seed
    update_dict["print_method"] = "rich"
    cfg = YAMLConfig(args.config_file, **update_dict)

    out_dir = Path(cfg.output_dir)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir) / out_dir.name

    suffix = f"{len(args.degradations)}_degrads_max_num_{args.max_num_samples}"
    if args.shuffle:
        suffix += "_shuffle"

    out_dir = out_dir / f"dump_{suffix}"
    cfg.output_dir = args.out_dir = out_dir.resolve().as_posix()

    annotation_file = cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"]
    args.annotation_file = Path(annotation_file).resolve().as_posix()

    dump_json(out_dir / "metadata.json", vars(args), indent=4)
    cfg.save()

    main(cfg, args)
