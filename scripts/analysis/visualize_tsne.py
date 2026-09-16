import argparse
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import PIL.Image as Image
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rich.console import Console
from rich.live import Live
from rich.progress import track
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from robust_af.transforms import DEGRADATION_TRANSFORMS

_SPATIAL_SIZE_WARN = True

# 15 degradations + 1 identity
PALETTE = px.colors.qualitative.Plotly + [
    px.colors.qualitative.D3[1],
    px.colors.qualitative.D3[3],
    px.colors.qualitative.D3[5],
    px.colors.qualitative.D3[7],
    px.colors.qualitative.D3[8],
    px.colors.qualitative.D3[9],
]
PLOTLY_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "plotly_script.js"
)

CONSOLE = Console()


def setup(args: argparse.Namespace):
    dump_dir = Path(args.dump_dirs[-1])

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
    if args.find_failures:
        dir_name = "failure_analysis"
        if len(args.failure_exclude) > 0:
            dir_name += "_exclude_" + "_".join(args.failure_exclude)
        if not args.fp:
            dir_name += "_no_fp"
        out_dir = out_dir / dir_name
    out_dir = (
        out_dir
        / f"tsne_{len(args.degradations)}_{args.feat_size}_{args.perplexity}_{args.n_components}d_iou_{args.iou_threshold}_score_{args.score_threshold}_dumps_{len(args.dump_dirs)}"
    )
    if not args.show_legend:
        out_dir = out_dir.with_name(f"{out_dir.name}_no_legend")
    args.out_dir = out_dir.as_posix()

    out_dir.mkdir(parents=True, exist_ok=True)
    # save args
    with open(out_dir / "args.json", "w") as f:
        content = vars(args)
        json.dump(content, f, indent=4)


def find_failures(
    coco_gt: COCO,
    coco_results_file: Path,
    iou_thr: float = 0.5,
    score_thr: float = 0.3,
    find_fp: bool = True,
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
        if find_fp:
            # FP: confident, non-ignored detections that matched nothing
            # [1 x D]
            dt_m = sample["dtMatches"][iou_index, :max_det]
            dt_scores = np.array(sample["dtScores"])[:max_det]
            dt_ig = sample["dtIgnore"][iou_index, :max_det]

            confident = (dt_scores > score_thr) & (dt_ig == 0)
            has_fp = np.any(dt_m[confident] == 0)
        else:
            has_fp = False

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
            if len(dt_idx) > 0 and sample["dtScores"][dt_idx[0]] <= score_thr:
                has_fn = True
                break

        if has_fp or has_fn:
            failures.add(sample["image_id"])
    return np.array(sorted(failures))


def standardize_representations(
    features: dict[str, list[torch.Tensor]], output_size: int = 1
) -> dict[str, np.ndarray]:
    global _SPATIAL_SIZE_WARN
    results = {}
    for scale, samples in features.items():
        for i, sample in enumerate(samples):
            h, w = sample.shape[-2:]
            if _SPATIAL_SIZE_WARN and (h < output_size or w < output_size):
                CONSOLE.log(
                    f"[yellow]Warning: spatial size of features is smaller than pool size, ({h}, {w}) < ({output_size}, {output_size})"
                )
            # [C, H, W] -> [C, new_H, new_W]
            samples[i] = F.adaptive_avg_pool2d(sample, output_size)
        # [[C, new_H, new_W], ...] -> [B, C, new_H, new_W] -> [B, C * new_H * new_W]
        samples = torch.stack(samples, axis=0).flatten(1)
        results[scale] = samples.numpy()
    # warning once in one of features
    _SPATIAL_SIZE_WARN = False
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


def collect_dump(
    dump_dir: Path,
    args: argparse.Namespace,
    extract_features: bool = True,
    image_idss: Optional[dict[str, np.ndarray]] = None,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]], dict[str, dict[str, np.ndarray]]
]:
    if image_idss is None:
        image_idss = {}

    coco = COCO(args.annotation_file)

    backbone_features = {}
    features = {}
    failed_image_idss = {}
    degradations: list[str] = args.degradations
    for name in track(
        degradations,
        description="Collecting features and predictions...",
        console=CONSOLE,
    ):
        degradation_dir = dump_dir / name
        if not degradation_dir.exists():
            CONSOLE.log(f"The degradation {name} is not found. Skipped.")
            continue

        backbone_features_file = degradation_dir / "backbone_features.pt"
        features_file = degradation_dir / "features.pt"
        coco_results_file = degradation_dir / "coco_results.json"

        if extract_features:
            backbone_features[name], image_ids = load_degradation_features(
                backbone_features_file,
                args.feat_size,
                max_num_samples=args.max_num_samples,
            )
            if features_file.exists():
                features[name], _ = load_degradation_features(
                    features_file, args.feat_size, max_num_samples=args.max_num_samples
                )
            image_ids = np.array(image_ids)
        else:
            image_ids = image_idss[name]

        assert set(coco.getImgIds(imgIds=image_ids)) == set(image_ids)
        failed_image_ids = find_failures(
            coco,
            coco_results_file,
            iou_thr=args.iou_threshold,
            score_thr=args.score_threshold,
            find_fp=args.fp,
        )
        image_idss[name] = image_ids
        failed_image_idss[name] = failed_image_ids

    return dict(backbone=backbone_features, adapter=features), dict(
        image_idss=image_idss, failed_image_idss=failed_image_idss
    )


def collect_dumps(args: argparse.Namespace):
    def _count_failures(failures: dict[str, np.ndarray]):
        num_failures = sum(np.count_nonzero(f) for f in failures.values())
        return num_failures

    # collect features and predictions
    results, main_metadata = collect_dump(Path(args.dump_dirs[-1]), args)

    if args.find_failures:
        # collect predictions for others
        failuress = {}
        excluded_failed_image_idss = None
        for i, dump_dir in enumerate(args.dump_dirs[:-1]):
            _, metadata = collect_dump(
                Path(dump_dir),
                args,
                extract_features=False,
                image_idss=main_metadata["image_idss"],
            )
            metadata = preprocess_metadata(
                metadata, args, excluded_failed_image_idss=excluded_failed_image_idss
            )
            excluded_failed_image_idss = metadata["excluded_failed_image_idss"]

            failures = metadata["failures"]
            args.titles[i] += f" (failures={_count_failures(failures)})"
            failuress[args.titles[i]] = failures

        main_metadata = preprocess_metadata(
            main_metadata, args, excluded_failed_image_idss=excluded_failed_image_idss
        )
        failures = main_metadata["failures"]
        args.titles[-1] += f" (main, failures={_count_failures(failures)})"
        failuress[args.titles[-1]] = failures
    else:
        failuress = None
    main_metadata["failuress"] = failuress
    return results, main_metadata


def preprocess_metadata(
    metadata: dict,
    args: argparse.Namespace,
    excluded_failed_image_idss: Optional[dict[str, np.ndarray]] = None,
):
    # filter failures
    failed_image_idss = metadata["failed_image_idss"]
    new_excluded_failed_image_idss = {}
    for degradation in args.failure_exclude:
        # (first dump)
        if excluded_failed_image_idss is None:
            exclusive = deepcopy(failed_image_idss[degradation])
        else:
            # (other dumps)
            exclusive = excluded_failed_image_idss[degradation]

        for name, image_ids in failed_image_idss.items():
            failed_image_idss[name] = np.setdiff1d(image_ids, exclusive)
        new_excluded_failed_image_idss[degradation] = exclusive
    metadata["excluded_failed_image_idss"] = new_excluded_failed_image_idss

    # generate failure boolean map
    failures = {}
    image_idss = metadata["image_idss"]
    for degradation, image_ids in image_idss.items():
        failures[degradation] = np.isin(
            image_ids, failed_image_idss[degradation], assume_unique=True
        )
    metadata["failures"] = failures
    return metadata


def run_tsne(features: dict[str, dict[str, np.ndarray]], args: argparse.Namespace):
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

    _degradation_features = next(iter(features.values()))
    # degradations -> multi-scales -> samples
    scales = list(_degradation_features.keys())

    # visualize feature distribution for each scale
    results = {}
    minmax = MinMaxScaler(feature_range=(-10, 10))
    for scale in track(scales, description="Running t-SNE...", console=CONSOLE):
        samples = [v[scale] for v in features.values()]

        # K: number of degradations
        # [[B, C], ...] -> [K*B, C]
        samples = np.concatenate(samples, axis=0)
        if samples.shape[1] > pca.n_components:
            samples = pca.fit_transform(samples)
        result = tsne_model.fit_transform(samples)

        # normalization + scale to a specific range for better vis
        results[scale] = minmax.fit_transform(result)
    return results


def draw_plotly_plot(
    result: np.ndarray,
    image_idss: dict[str, np.ndarray],
    args: argparse.Namespace,
    failures: Optional[dict[str, np.ndarray]] = None,
    marker: dict = dict(size=6, opacity=0.8),
    line: dict = dict(width=0, color="black"),
    failed_line: dict = dict(width=1.5, color="black"),
    show_failures: bool = False,
) -> list[go.Scatter]:
    num_dims = result.shape[1]
    degradations = list(image_idss.keys())
    num_images = len(next(iter(image_idss.values())))

    ScatterCls = go.Scatter3d if num_dims == 3 else go.Scatter
    traces = []
    for i, name in enumerate(degradations):
        marker["color"] = PALETTE[i % len(PALETTE)]
        hovertemplate = "image_id: %{customdata[0]}"

        start = i * num_images
        end = start + num_images
        data = dict(x=result[start:end, 0], y=result[start:end, 1])
        if num_dims == 3:
            data["z"] = result[start:end, 2]
        ids = image_idss[name]
        fail = np.zeros_like(ids, dtype=bool) if failures is None else failures[name]

        # normal points
        normal = ~fail
        if np.any(normal):
            customdata = np.stack([ids[normal], fail[normal].astype(int)], axis=-1)
            kwargs = {k: v[normal].tolist() for k, v in data.items()}
            traces.append(
                ScatterCls(
                    mode="markers",
                    name=name,
                    marker=dict(line=line, **marker),
                    hovertemplate=hovertemplate,
                    legendgroup=name,
                    showlegend=args.show_legend,
                    customdata=customdata.tolist(),
                    **kwargs,
                )
            )

        # failed points
        if np.any(fail):
            customdata = np.stack([ids[fail], fail[fail].astype(int)], axis=-1)
            kwargs = {k: v[fail].tolist() for k, v in data.items()}
            traces.append(
                ScatterCls(
                    mode="markers",
                    name=f"{name} (failed)",
                    marker=dict(line=failed_line, **marker),
                    hovertemplate=hovertemplate,
                    # legendgroup=f"{name}_failed",
                    legendgroup=name,
                    showlegend=False,
                    customdata=customdata.tolist(),
                    **kwargs,
                )
            )
    if failures is not None and show_failures:
        kwargs = dict(x=[None], y=[None])
        if num_dims == 3:
            kwargs["z"] = [None]

        name = "failed ("
        if args.fp:
            name += "FP or "
        name += "FN)"
        traces.append(
            go.Scatter(
                mode="markers",
                marker=dict(line=failed_line, color="#FFFFFF"),
                name=f"{name}<br> @ IoU: {args.iou_threshold}, conf: {args.score_threshold}",
                showlegend=args.show_legend,
                **kwargs,
            )
        )
    return traces


def autocrop_whitespace(
    path: Path,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    padding: int = 25,
    width: bool = False,
    height: bool = True,
):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    # 找出非背景色的 pixel 範圍
    mask = np.any(arr != bg_color, axis=-1)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 加一點 padding 避免貼邊
    # 決定 cropped area
    if width:
        cmin = max(0, cmin - padding)
        cmax = min(arr.shape[1], cmax + padding)
    else:
        cmin = 0
        cmax = arr.shape[1]
    # 只 crop 底部
    rmin = 0
    if height:
        # rmin = max(0, rmin - padding)
        rmax = min(arr.shape[0], rmax + padding)
    else:
        rmax = arr.shape[0]

    cropped = img.crop((cmin, rmin, cmax, rmax))
    cropped.save(path)
    # path = path.with_suffix(".pdf")
    # cropped.save(path)
    return path


def render_plotly_legend(
    out_dir: Path,
    degradations: list[str],
    args: argparse.Namespace,
    marker: dict = dict(size=6, opacity=0.8),
    line: dict = dict(width=0, color="black"),
):
    legend_fig = go.Figure()
    for i, d in enumerate(degradations):
        marker["color"] = PALETTE[i % len(PALETTE)]
        legend_fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(line=line, **marker),
                name=d,
                showlegend=True,
            )
        )
    legend_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=150,
        height=args.plot_size,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    out_file = out_dir / "legend_only.png"
    legend_fig.write_image(out_file, scale=3)
    autocrop_whitespace(out_file)


def render_plotly(
    out_dir: Path,
    # scales -> features
    tsne_results: dict[str, np.ndarray],
    # degradations -> ids
    image_idss: dict[str, np.ndarray],
    args: argparse.Namespace,
    # dumps -> degradations -> failed boolean map (ids)
    failuress: Optional[dict[str, dict[str, np.ndarray]]] = None,
    prefix_title: str = "backbone layer",
    post_title: str = "",
):
    # make chrome run in headless mode
    os.environ.pop("DISPLAY", None)

    import plotly.io as pio

    pio.get_chrome()

    num_plots = 1 if failuress is None else len(failuress)
    cols = min(num_plots, args.num_col_plots)
    rows = math.ceil(num_plots / cols)
    width = cols * args.plot_size
    height = rows * args.plot_size

    figs: dict[str, go.Figure] = {}
    for scale, result in track(tsne_results.items(), "Drawing figures..."):
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes="all",
            shared_yaxes="all",
            subplot_titles=args.titles,
            horizontal_spacing=0.01,
            vertical_spacing=0.01,
        )
        # fig = go.Figure()

        if failuress is None:
            traces = draw_plotly_plot(result, image_idss, args, show_failures=False)
            fig.add_traces(traces)
        else:
            for i, failures in enumerate(failuress.values()):
                show = i == 0  # 只第一個 subplot 顯示 legend
                traces = draw_plotly_plot(
                    result,
                    image_idss,
                    args,
                    failures=failures,
                    show_failures=args.find_failures and show,
                )

                row = i // cols + 1
                col = i % cols + 1
                for trace in traces:
                    trace.showlegend &= show
                    fig.add_trace(trace, row=row, col=col)

        if args.show_main_title:
            fig.update_layout(title=f"{prefix_title} - layer: {scale}{post_title}")
        # make y-axis's scale same with x-axis
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        figs[scale] = fig

    with CONSOLE.status("[yellow]Rendering plots..."):
        with open(PLOTLY_SCRIPT_PATH, "r") as f:
            post_script = f.read()

        for scale, fig in figs.items():
            fig.write_html(file=out_dir / f"{scale}.html", post_script=post_script)
            if not (args.show_main_title or args.show_legend):
                fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
            fig.write_image(
                file=out_dir / f"{scale}.pdf",
                width=width,
                height=height,
                scale=args.scale,
            )
        render_plotly_legend(out_dir, list(image_idss.keys()), args)

    CONSOLE.log(f"Saved plots to {out_dir}")

    for fig in figs.values():
        if args.show:
            fig.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE feature visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dump_dirs",
        nargs="+",
        help="path of dump directories. We use the features and predictions of the last dump to visualize. For others, we only use the predictions.",
    )
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
        "--find-failures", action=argparse.BooleanOptionalAction, default=False
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
        default=0.5,
        help="Score threshold to filter invalid predictions.",
    )
    parser.add_argument(
        "--fp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark false positives.",
    )
    parser.add_argument(
        "--failure-exclude",
        type=str,
        nargs="*",
        default=[],
        help="Don't show any failures (same image_id) if fail in specific degradations based on the first dump.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the result in a graphical window."
    )

    # plotly settings
    parser.add_argument(
        "--show-main-title",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the main title on the plot",
    )
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the legend on the plot",
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        default=[""],
        # default=["Detector", "Detector with SpatialAFR"],
        help="The titles for subplots, following the order of dump_dirs.",
    )
    parser.add_argument(
        "--num-col-plots", type=int, default=3, help="Number of plots per row"
    )
    parser.add_argument("--plot-size", type=int, default=800, help="The size of a plot")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale the plot containing multiple subplots. Each subplot are plot_size x plot_size.",
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
        default=20,
        help="The number of parallel jobs to run for neighbors search.",
    )
    args = parser.parse_args()
    num_dumps = len(args.dump_dirs)
    assert num_dumps == len(args.titles)
    if not args.find_failures:
        assert num_dumps == 1, "No meaning without showing failures."
    args.learning_rate = "auto" if args.learning_rate is None else args.learning_rate
    return args


if __name__ == "__main__":
    args = parse_args()
    setup(args)
    titles = deepcopy(args.titles)
    results, metadata = collect_dumps(args)
    out_dir = Path(args.out_dir)
    with Live(console=CONSOLE):
        for name, features in results.items():
            if len(features) == 0:
                continue

            prefix_title = f"{name} outputs"
            post_title = ""
            if args.find_failures:
                prefix_title = (
                    f"Failure distribution on {prefix_title} (from main model)"
                )
                if len(args.failure_exclude) > 0:
                    post_title = (
                        "<br>Filters: "
                        + ", ".join(args.failure_exclude)
                        + f"-fail in {titles[0]}"
                    )
            else:
                prefix_title = prefix_title.capitalize()

            CONSOLE.log(f"t-SNE features: {name}")
            tsne_out_dir = out_dir / name
            tsne_out_dir.mkdir(parents=True, exist_ok=True)
            tsne_results = run_tsne(features, args)
            render_plotly(
                tsne_out_dir,
                tsne_results,
                metadata["image_idss"],
                args,
                failuress=metadata["failuress"],
                prefix_title=prefix_title,
                post_title=post_title,
            )
