import argparse
import json
import logging
import os.path as osp
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional, Union

import detectron2.data.transforms as T
import torch
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_setup
from detectron2.utils.env import seed_all_rng
from omegaconf import DictConfig
from rich.progress import track
from torch.utils.data import DataLoader

from robust_af.detrex.apis import Inferencer
from robust_af.detrex.data.dataset_mappers import DetrDatasetMapper
from robust_af.detrex.data.transforms import Degradation
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

        if self.model.with_robust_image_module:
            self.model.robust_image_module.register_forward_hook(self._save_images)
        if self.model.with_robust_module:
            self.model.robust_module.register_forward_hook(self._save_features)

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
            # make shuffle determinitic
            seed_all_rng(seed)
            mapper = deepcopy(TEST_MAPPER)

            # specify name of degradation
            aug: Degradation = instantiate(mapper.augmentations[0])
            assert isinstance(aug, Degradation)
            aug.get_transform = partial(aug.get_transform, name=name)
            mapper.augmentations[0] = aug
            cfg.mapper = mapper

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
            dataloader_cfg,
            degradations,
            max_num_samples,
            shuffle=shuffle,
            seed=seed + 1,
        )

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

                    if self.images is not None:
                        sample["restored_image"] = self.images[i]
                    if self.features is not None:
                        sample["restored_features"] = {
                            k: v[i] for k, v in self.features.items()
                        }

                    counter += 1
                    last_sample = counter == num_samples
                    yield preds, sample, counter, last_sample


def main(cfg: DictConfig):
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
    predictions = []
    out_dir: Path = Path(cfg.train.output_dir)
    for i, (preds, sample, counter, last_sample) in enumerate(generator):
        image_id = sample["image_id"]
        restored_image = sample.get("restored_image")
        restored_features = sample.get("restored_features")
        instances = preds["instances"]

        if restored_image is not None:
            images[image_id] = restored_image.cpu()
        if restored_features is not None:
            features[image_id] = {k: v.cpu() for k, v in restored_features.items()}

        # convert preds to coco format
        coco_instances = convert_instances_to_coco(
            cfg.dataloader.test.dataset.names, image_id, instances
        )
        predictions.extend(coco_instances)

        # save results per degradation
        if last_sample:
            assert len(images) == counter or len(features) == counter

            name = sample["degradation"]
            kwargs = dict(
                images=None if len(images) == 0 else images,
                features=None if len(features) == 0 else features,
                metadata=dict(degradation=name, num_samples=counter),
            )
            dump_results(out_dir / name, predictions, **kwargs)

            images = {}
            features = {}
            predictions = []


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
        help="The maximum number of samples per category. "
        "Higher number need longer time to calculate.",
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

    out_dir = out_dir / f"dump_{len(args.degradations)}_degradations"
    cfg.train.output_dir = args.out_dir = out_dir.as_posix()

    default_setup(cfg, args)

    # save args
    with open(osp.join(out_dir, "args.json"), "w") as f:
        content = vars(args)
        json.dump(content, f, indent=4)

    main(cfg, vis_period=args.vis_period)
