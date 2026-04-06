import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rich.console import Console
from rich.live import Live
from rich.progress import track
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from robust_af.transforms import DEGRADATION_TRANSFORMS

CONSOLE = Console()


def find_failures(
    coco_gt: COCO, coco_results_file: Path, iou_thr: float = 0.5, score_thr: float = 0.3
):
    coco_dt = coco_gt.loadRes(coco_results_file.as_posix())
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
    # only evaluate specific images to avoid unnecessary comparison
    coco_eval.params.imgIds = coco_dt.getImgIds()
    coco_eval.evaluate()

    params = coco_eval.params
    iou_index = int(np.where(np.isclose(params.iouThrs, iou_thr))[0][0])
    max_det = params.maxDets[-1]

    num_cat = len(params.catIds) if params.useCats else 1
    num_area = len(coco_eval.params.areaRng)
    stride = num_cat * num_area

    # chunk by image, take only 1st area range (all)
    eval_imgs = [
        sample
        for i in range(0, len(coco_eval.evalImgs), stride)
        for sample in coco_eval.evalImgs[i : i + num_cat]
        if sample is not None
    ]

    failures = set()
    for sample in eval_imgs:
        # FP: confident, non-ignored detections that matched nothing
        # [1 x D]
        dt_m = sample["dtMatches"][iou_index, :max_det]
        dt_scores = np.array(sample["dtScores"])[:max_det]
        dt_ig = sample["dtIgnore"][iou_index, :max_det]

        confident = (dt_scores >= score_thr) & (dt_ig == 0)
        has_fp = np.any(dt_m[confident] == 0)

        # FN: non-ignored GTs not matched or matched by low-score dt
        # [1 x G]
        dt_ids = np.array(sample["dtIds"])
        gt_m = sample["gtMatches"][iou_index]
        gt_ig = sample["gtIgnore"]

        has_fn = False
        for dt_id, ignored in zip(gt_m, gt_ig):
            if ignored:
                continue
            if dt_id == 0:
                has_fn = True
                break
            dt_idx = np.where(dt_ids == dt_id)[0]
            if len(dt_idx) > 0 and sample["dtScores"][dt_idx[0]] < score_thr:
                has_fn = True
                break

        if has_fp or has_fn:
            failures.add(sample["image_id"])
    return np.array(sorted(failures))


def standardize_representations(
    features: dict[str, list[torch.Tensor]], output_size: int = 1
) -> dict[str, np.ndarray]:
    results = {}
    for scale, samples in features.items():
        # [[C, H, W], ...] -> [[C, new_H, new_W], ...]
        samples = [F.adaptive_avg_pool2d(sample, output_size) for sample in samples]
        # [[C, new_H, new_W], ...] -> [B, C, new_H, new_W] -> [B, C * new_H * new_W]
        samples = torch.stack(samples, axis=0).flatten(1)
        results[scale] = samples.numpy()
    return results


def load_degradation_features(
    path: Path, feat_size: int, max_num_samples: Optional[int] = None
):
    features: dict[int, dict[str, torch.Tensor]] = torch.load(path, weights_only=True)
    image_ids = list(features.keys())
    if max_num_samples is not None:
        image_ids = image_ids[:max_num_samples]

    # group by scale
    grouped_features = defaultdict(list)
    for image_id in image_ids:
        for scale, features_ in features[image_id].items():
            grouped_features[scale].append(features_)

    return standardize_representations(
        grouped_features, output_size=feat_size
    ), image_ids


def collect_features(
    dump_dir: Path, args: argparse.Namespace
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, np.ndarray]]:
    coco = COCO(args.annotation_file)

    backbone_features = {}
    features = {}
    failures = {}
    for name in track(
        args.degradations, description="Collecting features...", console=CONSOLE
    ):
        degradation_dir = dump_dir / name
        backbone_features_file = degradation_dir / "backbone_features.pt"
        features_file = degradation_dir / "features.pt"
        coco_results_file = degradation_dir / "coco_results.json"

        backbone_features[name], image_ids = load_degradation_features(
            backbone_features_file, args.feat_size, max_num_samples=args.max_num_samples
        )
        features[name], _ = load_degradation_features(
            features_file, args.feat_size, max_num_samples=args.max_num_samples
        )

        image_ids = np.array(image_ids)
        failed_image_ids = find_failures(
            coco,
            coco_results_file,
            iou_thr=args.iou_threshold,
            score_thr=args.score_threshold,
        )
        failures[name] = np.isin(image_ids, failed_image_ids, assume_unique=True)

    return dict(backbone_features=backbone_features, features=features), failures


def visualize_with_tsne(
    out_dir: Path,
    features: dict[str, dict[str, np.ndarray]],
    failures: dict[str, np.ndarray],
    args: argparse.Namespace,
):
    # build t-SNE model
    # disable OpenBLAS multi-threading, due to it is already multi-threaded
    # reference: https://github.com/OpenMathLib/OpenBLAS/blob/develop/USAGE.md#how-can-i-use-openblas-in-multi-threaded-applications
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    pca = PCA(n_components=50, random_state=7777)
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

    CONSOLE.log("Running t-SNE.")
    fig = plt.figure(figsize=(9, 6), layout="constrained")
    # degradations -> multi-scales -> samples
    scales = list(list(features.values())[0].keys())

    # visualize feature distribution for each scale
    minmax = MinMaxScaler(feature_range=(-10, 10))
    for scale in track(scales, description="Running t-SNE...", console=CONSOLE):
        samples = {k: v[scale] for k, v in features.items()}

        keys = list(samples.keys())
        samples = list(samples.values())
        batch_size, _ = samples[0].shape

        # K: number of degradations
        # [[B, C], ...] -> [K*B, C]
        samples = np.concatenate(samples, axis=0)
        if samples.shape[1] > pca.n_components:
            samples = pca.fit_transform(samples)
        result = tsne_model.fit_transform(samples)

        # normalization + scale to a specific range for better vis
        result = minmax.fit_transform(result)

        fig.suptitle(f"{scale} backbone layer")
        axes = fig.add_subplot(projection="3d" if args.n_components == 3 else None)

        colors = [i for i in range(len(keys)) for _ in range(batch_size)]
        # colors = list(
        #     itertools.chain.from_iterable([i] * batch_size for i in range(len(keys)))
        # )
        linewidths = [
            linewidth.item()
            for k in keys
            for linewidth in np.where(failures[k], 1.0, 0.2)
        ]
        scatter = axes.scatter(
            *np.transpose(result),
            alpha=0.8,
            s=20,
            c=colors,
            linewidths=linewidths,
            edgecolors="black",
            cmap="tab20",
        )
        # axes.set_xlim(-10, 10)
        # axes.set_ylim(-10, 10)
        handles, _ = scatter.legend_elements(prop="colors", num=None)
        fig.legend(handles, keys, loc="outside right")

        fig.savefig(out_dir / f"backbone_{scale}.png")
        if args.show:
            fig.show()
        fig.clear()
    plt.close(fig)
    CONSOLE.log(f"Save results to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE feature visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dump_dir", help="path of dump directory")
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="Use specific annotation file instead of in metadata.json. If None, use the config in metadata.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory of t-SNE results. If None, use dump_dir.",
    )
    parser.add_argument(
        "--feat-size",
        type=int,
        default=1,
        help="Spatial size of features.",
    )
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=None,
        help="The maximum number of samples per degradation. If None, use all images in dump_dir",
    )

    parser.add_argument(
        "--degradations",
        type=str,
        nargs="+",
        default=list(DEGRADATION_TRANSFORMS.keys()),
        help="Visualized degradations. If it is not existed in dump_dir, ignore it.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold to identiy as Matched or Unmatched.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold to filter invalid predictions.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the result in a graphical window."
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


if __name__ == "__main__":
    args = parse_args()
    dump_dir = Path(args.dump_dir)

    # load metadata
    with open(dump_dir / "metadata.json", "r") as f:
        metadata: dict = json.load(f)

    # filter degradations if exists
    target_degradations = set(args.degradations)
    args.degradations = list(
        filter(lambda name: name in target_degradations, metadata["degradations"])
    )
    if args.annotation_file is None:
        args.annotation_file = metadata.get("annotation_file")

    # determine out_dir
    if args.out_dir is None:
        out_dir = dump_dir
    else:
        out_dir = Path(args.out_dir)
    out_dir = (
        out_dir
        / f"tsne_{len(args.degradations)}_{args.feat_size}_{args.perplexity}_{args.n_components}d_iou_{args.iou_threshold}_score_{args.score_threshold}"
    )
    args.out_dir = out_dir.as_posix()

    out_dir.mkdir(parents=True, exist_ok=True)
    # save args
    with open(out_dir / "args.json", "w") as f:
        content = vars(args)
        json.dump(content, f, indent=4)

    results, failures = collect_features(dump_dir, args)
    with Live(console=CONSOLE):
        for name, features in results.items():
            CONSOLE.log(f"t-SNE features: {name}")
            tsne_out_dir = out_dir / name
            tsne_out_dir.mkdir(parents=True, exist_ok=True)
            visualize_with_tsne(tsne_out_dir, features, failures, args)
