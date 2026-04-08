import argparse
import json
import logging
import os.path as osp
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import detectron2.data.transforms as T
import torch
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import Instances
from detectron2.utils.env import seed_all_rng
from detectron2.utils.visualizer import Visualizer
from omegaconf import DictConfig
from rich.progress import track
from torch.utils.data import DataLoader

from robust_af.detrex.apis import Inferencer
from robust_af.detrex.data.dataset_mappers import DetrDatasetMapper
from robust_af.detrex.data.transforms import Degradation
from robust_af.detrex.engine import default_setup
from robust_af.detrex.utils import convert_instances_to_coco
from robust_af.transforms import DEGRADATION_TRANSFORMS
from robust_af.utils import dump_results

LOGGER = logging.getLogger("detectron2")

TEST_MAPPER = L(DetrDatasetMapper)(
    augmentations=[
        L(Degradation)(),
        L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
    ],
    is_train=False,
    use_instance_mask=False,
    image_format="RGB",
)


class DumpInferencer(Inferencer):
    """An inference for dumping features and predictions to COCO format

    References:
        1. detectron2.evaluation.inference_on_dataset
        2. base/train_net.py
    """

    def __init__(self, cfg: Union[str, Path, DictConfig]) -> None:
        super().__init__(cfg)

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
        self, module: nn.Module, args: tuple, output: dict[str, torch.Tensor]
    ):
        # [1, c, h, w]
        self.features = output

    def _save_backbone_features(
        self, module: nn.Module, args: tuple, output: dict[str, torch.Tensor]
    ):
        # [1, c, h, w]
        self.backbone_features = output

    def prepare_dataloaders(
        self,
        cfg: DictConfig,
        degradations: list[str],
        max_num_samples: int,
        shuffle: bool = False,
        seed: int = 2026,
    ) -> dict[str, DataLoader]:
        cfg = deepcopy(cfg)
        cfg.batch_size = 1

        dataloaders: dict[str, DataLoader] = {}
        for name in degradations:
            mapper = deepcopy(TEST_MAPPER)

            # specify name of degradation
            aug = mapper.augmentations[0]
            assert aug["_target_"] is Degradation
            aug.name = name
            aug.seed = seed

            cfg.mapper = mapper
            # make shuffle determinitic
            seed_all_rng(seed)
            dataloaders[name] = self.prepare_dataloader(
                cfg, max_num_samples, shuffle=shuffle
            )

        # validate consistency across degradations
        indices = dataloaders[degradations[0]].dataset._dataset.indices
        for dataloader in dataloaders.values():
            assert dataloader.dataset._dataset.indices == indices
        return dataloaders

    def __call__(
        self,
        dataloader_cfg: Union[str, Path, DictConfig],
        degradations: list[str],
        shuffle: bool = False,
        seed: int = 2026,
        max_num_samples: int = 100,
    ):
        # build dataloader
        if not isinstance(dataloader_cfg, DictConfig):
            dataloader_cfg = LazyConfig.load(dataloader_cfg).dataloader.test
        dataloaders = self.prepare_dataloaders(
            dataloader_cfg, degradations, max_num_samples, shuffle=shuffle, seed=seed
        )

        # avoid any randomness
        seed_all_rng(seed)
        for name, dataloader in dataloaders.items():
            num_samples = len(dataloader.dataset)
            counter = 0
            for batch_inputs in track(dataloader, description=f"Inference ({name})"):
                batch_inputs: list[dict]
                batch_preds = self.forward(batch_inputs)

                for i, preds in enumerate(batch_preds):
                    sample = batch_inputs[i]
                    sample["degradation"] = name

                    sample["backbone_features"] = {
                        k: v[i] for k, v in self.backbone_features.items()
                    }
                    if self.images is not None:
                        sample["images"] = self.images[i]
                    if self.features is not None:
                        sample["features"] = {k: v[i] for k, v in self.features.items()}

                    counter += 1
                    last_sample = counter == num_samples
                    yield preds, sample, counter, last_sample


def main(cfg: DictConfig, args: argparse.Namespace):
    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
    inferencer = DumpInferencer(cfg)
    generator = inferencer(
        cfg.dataloader.test,
        args.degradations,
        shuffle=args.shuffle,
        seed=cfg.train.seed,
        max_num_samples=args.max_num_samples,
    )

    images = {}
    features = {}
    backbone_features = {}
    predictions = []
    out_dir: Path = Path(cfg.train.output_dir)
    vis_image_ids = set(args.vis_image_ids)
    for preds, sample, counter, last_sample in generator:
        image_id: int = sample["image_id"]
        name: str = sample["degradation"]
        images_i: Optional[torch.Tensor] = sample.get("images")
        features_i: Optional[dict[str, torch.Tensor]] = sample.get("features")
        backbone_features_i: dict[str, torch.Tensor] = sample["backbone_features"]
        instances: Instances = preds["instances"]

        backbone_features[image_id] = {
            k: v.cpu() for k, v in backbone_features_i.items()
        }
        if images_i is not None:
            images[image_id] = images_i.cpu()
        if features_i is not None:
            features[image_id] = {k: v.cpu() for k, v in features_i.items()}

        # convert preds to coco format
        coco_instances = convert_instances_to_coco(
            cfg.dataloader.test.dataset.names, image_id, instances
        )
        predictions.extend(coco_instances)

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
            image = convert_image_to_rgb(
                sample["image"].permute(1, 2, 0), inferencer.input_format
            )
            vis = Visualizer(image, metadata=metadata)
            vis_image = vis.draw_instance_predictions(instances)

            image_dir = out_dir / name / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            vis_image.save(image_dir / f"{image_id:05d}.jpg")


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
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle dataset before dumping.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg.train.init_checkpoint = args.checkpoint

    out_dir = Path(cfg.train.output_dir)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir) / out_dir.name

    suffix = f"{len(args.degradations)}_degrads_max_num_{args.max_num_samples}"
    if args.shuffle:
        suffix += "_shuffle"

    out_dir = out_dir / f"dump_{suffix}"
    cfg.train.output_dir = args.out_dir = out_dir.resolve().as_posix()

    default_setup(cfg, args)

    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
    args.annotation_file = Path(metadata.json_file).resolve().as_posix()
    # save args
    with open(osp.join(out_dir, "metadata.json"), "w") as f:
        content = vars(args)
        json.dump(content, f, indent=4)

    main(cfg, args)
