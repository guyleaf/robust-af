# Modified from https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark/blob/de68cba29f2b7af99027d57991ee32c372197ba4/scripts/dataset_statistics.py

import argparse
import itertools
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets
from matplotlib.axes import Axes
from pycocotools.cocoeval import Params
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

DEFAULT_COCO_PARAMS = Params(iouType="bbox")
DEFAULT_COCO_AREAS = list(
    map(lambda area_range: int(area_range[1]), DEFAULT_COCO_PARAMS.areaRng[1:])
)


class StrideSampler(Sampler):
    def __init__(self, data_set, stride):
        self.indices = range(0, len(data_set), stride)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root-dirs",
        nargs="+",
        help="List of images directories for each of the coco files. "
        "If 1 element is given, it will be used for all of the"
        "elements of coco_files",
    )
    parser.add_argument("--annotations", nargs="+", help="List of COCO files path")
    parser.add_argument(
        "--out-file",
        type=str,
        default="statistics.png",
        help="Output path of the figure",
    )

    parser.add_argument(
        "--coco-areas",
        default=DEFAULT_COCO_AREAS,
        type=int,
        nargs="+",
        help="Redefine area rules, but muse contain three numbers, e.g. 30 70 125",
    )
    parser.add_argument("--title", default="", help="The title of the figure")
    args = parser.parse_args()

    assert len(args.annotations) > 0 and len(args.root_dirs) > 0
    assert len(args.annotations) == len(args.root_dirs) or len(args.root_dirs) == 1
    assert len(args.coco_areas) == 3

    if len(args.annotations) != len(args.root_dirs):
        args.root_dirs *= len(args.annotations)

    return args


def collect_statistics(
    annotations_file_paths: list[str], area_rules: list[int] = DEFAULT_COCO_AREAS
):
    x, y, w, h, areas, normalized_areas, area_nums, image_sizes, captions = (
        [],
        [],
        [],
        [],
        [],
        [],
        [0] * len(area_rules),
        [],
        {},
    )
    for ann_file in annotations_file_paths:
        with open(ann_file, "r") as f:
            coco = json.load(f)

        images = coco["images"]
        annotations = coco["annotations"]

        captions[os.path.basename(ann_file)] = [
            im.get("caption", "unknown") for im in images
        ]
        sizes = {im["id"]: (im["width"], im["height"]) for im in images}
        image_sizes.extend(sizes.values())

        for ann in annotations:
            bbox = ann["bbox"]
            cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            area = bbox[2] * bbox[3]
            W, H = sizes[ann["image_id"]]

            x.append(cx / W)
            y.append(cy / H)
            w.append(bbox[2])
            h.append(bbox[3])
            areas.append(area)
            normalized_areas.append(area / (W * H))

            # The area rule, there is an section between two numbers
            for i, area_rule in enumerate(area_rules):
                if area_rule >= area:
                    area_nums[i] += 1
                    break

    # calculate the number of all area
    area_nums.insert(0, sum(area_nums))

    x = np.array(x)
    # matplotlib's figure is origin at bottom-left
    y = 1 - np.array(y)
    w = np.array(w)
    h = np.array(h)
    areas = np.array(areas)
    normalized_areas = np.array(normalized_areas)
    area_label2num = dict(zip(DEFAULT_COCO_PARAMS.areaRngLbl, area_nums))
    image_sizes = np.array(image_sizes)
    return x, y, w, h, areas, normalized_areas, area_label2num, image_sizes, captions


def make_location_plot(axes: Axes, x: np.ndarray, y: np.ndarray):
    axes.plot(x, y, "o", markersize=1)
    axes.set_xlabel("x", fontsize="x-large")
    axes.set_ylabel("y", fontsize="x-large")
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes.tick_params(axis="both", labelsize="large")
    axes.set_title("UAV Location")


def make_area_plot(axes: Axes, areas: np.ndarray):
    # Object size
    bins = 100
    axes.hist(areas, bins, color="orange")
    axes.set_xlabel("Object area ratio", fontsize="x-large")
    axes.set_ylabel("Number of annotations", fontsize="x-large")
    # axes_2.set_xlim([0, 1.0])
    # axes_2.set_ylim([0, 12.5])
    axes.tick_params(axis="both", labelsize="large")
    axes.set_title("UAV Area")


def make_coco_area_plot(axes: Axes, area_label2num: dict[str, int]):
    xi = list(range(len(area_label2num)))
    yi = area_label2num.values()
    rects = axes.bar(xi, yi, width=0.6)
    axes.bar_label(rects)
    axes.set_ylabel("Number of annotations", fontsize="x-large")
    axes.set_xticks(xi)
    axes.set_xticklabels(area_label2num.keys())
    axes.tick_params(axis="both", labelsize="large")
    axes.set_title("UAV Area (COCO)")


def make_image_intensity_plot(
    axes: Axes, image_root_folders: list[str], annotations_file_paths: list[str]
):
    r, g, b = torch.zeros(256), torch.zeros(256), torch.zeros(256)

    for root_dir, ann_file in zip(image_root_folders, annotations_file_paths):
        dataset = torchvision.datasets.CocoDetection(
            root_dir, ann_file, transform=torchvision.transforms.ToTensor()
        )
        dataloader = DataLoader(
            dataset, num_workers=8, sampler=StrideSampler(dataset, 100)
        )
        for data in track(dataloader):
            data = data[0]
            r += torch.histc(data[0, 0, ...] * 255, bins=256, min=0, max=255)
            g += torch.histc(data[0, 1, ...] * 255, bins=256, min=0, max=255)
            b += torch.histc(data[0, 2, ...] * 255, bins=256, min=0, max=255)

    r = r / r.sum()
    g = g / g.sum()
    b = b / b.sum()

    axes.plot(r.numpy(), color="red", label="Red")
    axes.plot(g.numpy(), color="green", label="Green")
    axes.plot(b.numpy(), color="blue", label="Blue")
    axes.set_xlabel("Intensity", fontsize="x-large")
    axes.set_ylabel("PDF", fontsize="x-large")
    axes.set_ylim([0, 0.080])
    axes.set_xlim([0, 255])
    axes.tick_params(axis="both", labelsize="large")
    axes.set_title("Image Intensity")
    axes.legend()


def make_image_size_plot(axes: Axes, sizes: np.ndarray):
    x, y = sizes[:, 0], sizes[:, 1]
    max_w, max_h = x.max(), y.max()
    x = x / max_w
    y = y / max_h

    axes.plot(x, y, "o", markersize=1)
    axes.set_xlabel(f"x (max={max_w})", fontsize="x-large")
    axes.set_ylabel(f"y (max={max_h})", fontsize="x-large")
    # axes.set_xlim([0, 1])
    # axes.set_ylim([0, 1])
    axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes.tick_params(axis="both", labelsize="large")
    axes.set_title("Image Size")


def make_image_area_plot(axes: Axes, sizes: np.ndarray):
    areas = np.prod(sizes, axis=1)
    max_area = areas.max()
    areas = areas / max_area

    # remove outliers
    # reference: https://en.wikipedia.org/wiki/Interquartile_range
    upper_quartile = np.percentile(areas, 75)
    lower_quartile = np.percentile(areas, 25)
    IQR = (upper_quartile - lower_quartile) * 1.5
    lower_bound, upper_bound = lower_quartile - IQR, upper_quartile + IQR
    areas = areas[np.logical_and(upper_bound >= areas, areas >= lower_bound)]

    # Image sizes
    bins = 100
    axes.hist(areas, bins, color="orange")
    axes.set_xlabel(f"Image area ratio (max={max_area})", fontsize="x-large")
    axes.set_ylabel("Number of images", fontsize="x-large")
    axes.tick_params(axis="both", labelsize="large")
    axes.set_title("Image Area (Outlier removed)")


def make_caption_plot(axes: Axes, captions: dict[str, list[str]]):
    labels = sorted(set(itertools.chain.from_iterable(captions.values())))
    x = np.arange(len(labels))
    width = 0.25  # the width of the bars

    max_count = 0
    for multiplier, (subset, texts) in enumerate(captions.items()):
        counter = Counter(texts)
        offset = width * multiplier
        rects = axes.bar(
            x + offset, [counter[label] for label in labels], width, label=subset
        )
        axes.bar_label(rects, padding=3)
        max_count = max(max_count, counter.most_common(1)[0][1])

    axes.set_xticks(x + width, labels=labels)
    axes.set_ylabel("Number of images", fontsize="x-large")
    axes.set_ylim(0, max_count + 5000)
    axes.tick_params(axis="both", labelsize="large")
    axes.set_title("Image Caption")
    axes.legend(loc="upper left", ncols=3)


def main(args):
    annotations_file_paths = args.annotations
    image_root_folders = args.root_dirs
    x, y, w, h, areas, normalized_areas, area_label2num, image_sizes, captions = (
        collect_statistics(annotations_file_paths, args.coco_areas)
    )

    print(f"Average W, H: {w.mean()}, {h.mean()}")
    print("Average Area:", areas.mean())
    print("Small, Large Area:", areas.min(), areas.max())

    W, H = image_sizes.mean(axis=0)
    print(f"Average Image W, H: {W}, {H}")
    print("Average Image Area:", np.prod(image_sizes, axis=1).mean())

    fig, ((axes_1, axes_2, axes_3), (axes_4, axes_5, axes_6), (axes_7, *_)) = (
        plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    )
    fig.suptitle(args.title)

    make_location_plot(axes_1, x, y)
    make_area_plot(axes_2, normalized_areas)
    make_coco_area_plot(axes_3, area_label2num)
    make_image_intensity_plot(axes_4, image_root_folders, annotations_file_paths)
    make_image_size_plot(axes_5, image_sizes)
    make_image_area_plot(axes_6, image_sizes)
    make_caption_plot(axes_7, captions)

    fig.tight_layout()
    fig.savefig(args.out_file)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    main(args)
