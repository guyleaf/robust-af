import argparse
import itertools
import json
import logging
import os
import os.path as osp
import random
import sys
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import detectron2.data.transforms as T
import detrex.modeling.ema as ema
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetFromList
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.env import seed_all_rng
from omegaconf import DictConfig
from rich.progress import track
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.hooks import RemovableHandle

from robust_af.detrex.data.dataset_mappers import DetrDatasetMapper
from robust_af.detrex.data.transforms import Degradation
from robust_af.detrex.engine import default_setup
from robust_af.transforms import DEGRADATION_TRANSFORMS

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


class Inferencer:
    def __init__(
        self,
        cfg: Union[str, Path, DictConfig],
        max_num_samples: int = 100,
        vis_period: int = 10,
    ) -> None:
        if not isinstance(cfg, DictConfig):
            cfg = LazyConfig.load(cfg)

        model = self._load_model(cfg)
        self.model = model.eval().to(cfg.train.device)
        self.INs = {
            name: nn.InstanceNorm2d(spec.channels).eval().to(cfg.train.device)
            for name, spec in cfg.model.neck.input_shapes.items()
        }

        self.features: dict[str, torch.Tensor] = {}
        self.handles: list[RemovableHandle] = [
            self.model.backbone.register_forward_hook(self._save_features)
        ]
        self.max_num_samples = max_num_samples
        self.cfg = cfg

        self.in_features = set(cfg.model.neck.in_features)
        self.sample_output_dir = osp.join(self.cfg.train.output_dir, "samples")
        os.makedirs(self.sample_output_dir, exist_ok=True)
        self.vis_period = vis_period

    def _load_model(self, cfg: DictConfig) -> nn.Module:
        model = instantiate(cfg.model)
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(
            cfg.train.init_checkpoint
        )
        # Apply ema state for evaluation
        if (
            cfg.train.model_ema.enabled
            and cfg.train.model_ema.use_ema_weights_for_eval_only
        ):
            ema.apply_model_ema(model)
        return model

    def _save_features(
        self,
        module: nn.Module,
        args: tuple,
        output: dict[str, torch.Tensor],
    ):
        self.features = output
        return {k: v for k, v in output.items() if k in self.in_features}

    def _visualize_image(self, name: str, img: torch.Tensor):
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.model.input_format)
        fig = plt.figure()
        ax: plt.Axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.imshow(img.astype("uint8"))
        fig.savefig(osp.join(self.sample_output_dir, name))
        plt.close(fig)

    def prepare_dataset(self, cfg: DictConfig, unpair: bool):
        dataset: Union[list[dict], Dataset] = instantiate(cfg)

        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset, copy=False)

        # random sampling if unpair
        if unpair:
            indices = range(len(dataset))
            if self.max_num_samples < len(dataset):
                indices = random.sample(indices, k=self.max_num_samples)
        else:
            indices = range(min(self.max_num_samples, len(dataset)))
        dataset = Subset(dataset, indices)
        return dataset

    def prepare_dataloaders(
        self, cfg: DictConfig, degradations: list[str], unpair: bool
    ) -> tuple[DataLoader, dict[str, DataLoader]]:
        dataloader_config = deepcopy(cfg.dataloader.test)
        dataloader_config.batch_size = 1
        dataset_config = dataloader_config.dataset

        dataloaders = dict()
        for name in degradations:
            mapper = deepcopy(TEST_MAPPER)

            # specify name of degradation
            aug = mapper.augmentations[0]
            assert aug["_target_"] == Degradation.__name__
            aug.name = name

            dataloader_config.dataset = self.prepare_dataset(dataset_config, unpair)
            dataloader_config.mapper = mapper
            dataloaders[name] = instantiate(dataloader_config)

        return dataloaders

    def forward(self, inputs: Union[dict, tuple]):
        """Feed the inputs to the model."""
        return self.model(inputs)

    @torch.inference_mode()
    def __call__(
        self,
        degradations: list[str],
        seed: Optional[int] = None,
        unpair: bool = False,
        IN: bool = True,
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        seed_all_rng(seed)

        dataloaders = self.prepare_dataloaders(self.cfg, degradations, unpair)

        # [[C, H, W], ...]
        results = {}
        for name, dataloader in dataloaders.items():
            features = defaultdict(list)
            # IN_features = defaultdict(list)
            for i, data in enumerate(
                track(dataloader, description=f"Inference ({name})")
            ):
                self.forward(data)
                if i % self.vis_period == 0:
                    self._visualize_image(f"{name}_{i}.jpg", data[0]["image"])
                # multi-scale features
                for scale, v in self.features.items():
                    if IN:
                        v = self.INs[scale](v)
                        # IN_features[scale].append(IN_v.squeeze(0).cpu())
                    # [C, H, W]
                    features[scale].append(v.squeeze(0).cpu())
            results[name] = features
            # if IN:
            #     results[f"IN_{name}"] = IN_features
        return results


def standardize_representations(
    results: dict[str, dict[str, list[torch.Tensor]]], output_size: int = 1
) -> dict[str, dict[str, np.ndarray]]:
    for key, result in results.items():
        for scale, samples in result.items():
            # [[C, H, W], ...] -> [[C, new_H, new_W], ...]
            samples = [F.adaptive_avg_pool2d(sample, output_size) for sample in samples]
            # [[C, new_H, new_W], ...] -> [B, C, new_H, new_W] -> [B, C * new_H * new_W]
            samples = torch.stack(samples, axis=0).flatten(1)
            results[key][scale] = samples.numpy()
    return results


def get_file_stream_handler(logger: logging.Logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream != sys.stdout:
            return handler
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE feature visualization",
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
        "Higher number need longer time to calculate. Defaults to 100.",
    )
    parser.add_argument(
        "--feat-size",
        type=int,
        default=1,
        help="Spatial size of features.",
    )

    parser.add_argument(
        "--legend", action="store_true", help="Show the legend of all categories."
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the result in a graphical window."
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
        "--IN", action="store_true", help="Apply instance normalization or not."
    )
    parser.add_argument(
        "--unpair", action="store_true", help="Use unpair images to visualize."
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

    # t-SNE settings
    parser.add_argument(
        "--n-components", type=int, default=2, help="Dimension of the embedded space."
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="""
            The perplexity is related to the number of nearest neighbors that
            is used in other manifold learning algorithms. Larger datasets
            usually require a larger perplexity. Consider selecting a value
            between 5 and 50. Different values can result in significantly
            different results. The perplexity must be less than the number
            of samples.
        """,
    )
    parser.add_argument(
        "--early-exaggeration",
        type=float,
        default=12.0,
        help="""
            Controls how tight natural clusters in the original space are in
            the embedded space and how much space will be between them. For
            larger values, the space between natural clusters will be larger
            in the embedded space. Again, the choice of this parameter is not
            very critical. If the cost function increases during initial
            optimization, the early exaggeration factor or the learning rate
            might be too high.
        """,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="""
            The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
            the learning rate is too high, the data may look like a 'ball' with any
            point approximately equidistant from its nearest neighbours. If the
            learning rate is too low, most points may look compressed in a dense
            cloud with few outliers. If the cost function gets stuck in a bad local
            minimum increasing the learning rate may help.
            Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE,
            etc.) use a definition of learning_rate that is 4 times smaller than
            ours. So our learning_rate=200 corresponds to learning_rate=800 in
            those other implementations. The 'auto' option sets the learning_rate
            to `max(N / early_exaggeration / 4, 50)` where N is the sample size.
        """,
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for the optimization. Should be at"
        "least 250.",
    )
    parser.add_argument(
        "--n-iter-without-progress",
        type=int,
        default=300,
        help="""
            Maximum number of iterations without progress before we abort the
            optimization, used after 250 initial iterations with early
            exaggeration. Note that progress is only checked every 50 iterations so
            this value is rounded to the next multiple of 50.
        """,
    )
    parser.add_argument("--init", type=str, default="pca", help="The init method")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="The number of parallel jobs to run for neighbors search.",
    )
    args = parser.parse_args()
    args.learning_rate = "auto" if args.learning_rate is None else args.learning_rate
    return args


def collect_features(args: argparse.Namespace, cfg: DictConfig):
    inferencer = Inferencer(cfg, max_num_samples=args.max_num_samples)
    results = inferencer(
        args.degradations, seed=cfg.train.seed, unpair=args.unpair, IN=args.IN
    )
    results = standardize_representations(results, output_size=args.feat_size)
    return results


def visualize_with_tsne(
    args: argparse.Namespace, results: dict[str, dict[str, np.ndarray]]
):
    # build t-SNE model
    # disable OpenBLAS multi-threading, due to it is already multi-threaded
    # reference: https://github.com/OpenMathLib/OpenBLAS/blob/develop/USAGE.md#how-can-i-use-openblas-in-multi-threaded-applications
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    tsne_model = TSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        n_iter_without_progress=args.n_iter_without_progress,
        init=args.init,
        random_state=2026,
        n_jobs=args.n_jobs,
        verbose=True,
    )
    tsne_out_dir = osp.join(args.out_dir, "tsne")
    os.makedirs(tsne_out_dir, exist_ok=True)

    LOGGER.info("Running t-SNE.")
    fig = plt.figure(figsize=(9, 6), layout="constrained")
    # degradations -> multi-scales -> samples
    scales = list(list(results.values())[0].keys())

    # visualize feature distribution for each scale
    minmax = MinMaxScaler(feature_range=(-10, 10))
    for scale in scales:
        samples = {k: v[scale] for k, v in results.items()}

        keys = list(samples.keys())
        samples = list(samples.values())
        batch_size, _ = samples[0].shape

        # K: number of degradations
        # [[B, C], ...] -> [K*B, C]
        samples = np.concatenate(samples, axis=0)
        result = tsne_model.fit_transform(samples)

        # normalization + scale to a specific range for better vis
        result = minmax.fit_transform(result)

        fig.suptitle(f"{scale} backbone layer")
        axes = fig.add_subplot(projection="3d" if args.n_components == 3 else None)

        colors = list(
            itertools.chain.from_iterable([i] * batch_size for i in range(len(keys)))
        )
        scatter = axes.scatter(
            *np.transpose(result),
            alpha=0.8,
            s=20,
            c=colors,
            linewidths=0.5,
            edgecolors="black",
            cmap="tab20",
        )
        # axes.set_xlim(-10, 10)
        # axes.set_ylim(-10, 10)
        handles, _ = scatter.legend_elements(prop="colors", num=None)

        if args.legend:
            fig.legend(handles, keys, loc="outside right")

        fig.savefig(osp.join(tsne_out_dir, f"backbone_{scale}.png"))
        if args.show:
            fig.show()
        fig.clear()
    plt.close(fig)
    LOGGER.info(f"Save results to {tsne_out_dir}")


if __name__ == "__main__":
    args = parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg.train.init_checkpoint = args.checkpoint

    if args.out_dir is None:
        args.out_dir = cfg.train.output_dir
    else:
        args.out_dir = osp.join(args.out_dir, osp.basename(cfg.train.output_dir))

    out_dir = args.out_dir
    out_dir = f"{out_dir}_{len(args.degradations)}"
    if args.IN:
        out_dir = f"{out_dir}_IN"
    if args.unpair:
        out_dir = f"{out_dir}_unpairs"
    cfg.train.output_dir = args.out_dir = out_dir = (
        f"{out_dir}_{args.feat_size}_{args.perplexity}_{args.n_components}d"
    )

    default_setup(cfg, args)

    # prepare redirecting stdout/stderr to the file stream
    handler = get_file_stream_handler(LOGGER)
    assert handler is not None
    stdout_r = redirect_stdout(handler.stream)
    stderr_r = redirect_stderr(handler.stream)

    # write some infos to file
    LazyConfig.save(cfg, osp.join(out_dir, "config.yaml"))
    with open(osp.join(out_dir, "args.json"), "w") as f:
        content = vars(args)
        json.dump(content, f, indent=4)

    results = collect_features(args, cfg)
    with stdout_r, stderr_r:
        visualize_with_tsne(args, results)
