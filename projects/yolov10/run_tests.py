import argparse
import concurrent.futures.thread as thread
import os
import queue
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

CONSOLE = Console()


@dataclass
class Subset:
    name: str
    split: str
    robust: bool = False
    split_subset: bool = False
    degradation: Optional[str] = None


@dataclass
class Dataset:
    path: Path
    subset: Subset


def save_cfg(path: Path, cfg: dict):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def run(config: Path, name: str, dataset: Dataset, **kwargs):
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{config.stem}_"))

    splits = [
        item
        for item in dataset.subset.name.replace("val", "").split("_")
        if len(item) != 0
    ]
    splits = list(reversed(splits))
    if len(splits) != 0:
        suffix = "_".join(splits)
        name = f"{name}_{suffix}"

    with open(config, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    if dataset.subset.robust and dataset.subset.split_subset:
        assert (
            dataset.subset.degradation is not None
            and dataset.subset.degradation != "degraded"
        )
        with open(dataset.path, "r", encoding="utf-8") as f:
            dataset_cfg: dict = yaml.safe_load(f)

        dataset_cfg["degradations"] = [dataset.subset.degradation]

        dataset.path = tmp_dir / dataset.path.name
        save_cfg(dataset.path, dataset_cfg)

    work_dir = Path(cfg["name"])
    # concat with folder of training dataset
    # test structure: <training dataset>/<testing dataset>/<exp_name>
    work_dir = work_dir.parent.parent / dataset.path.stem / name
    cfg["name"] = work_dir.as_posix()
    cfg["data"] = dataset.path.as_posix()
    cfg["split"] = dataset.subset.split
    cfg.update(**kwargs)

    cfg_file = tmp_dir / config.name
    save_cfg(cfg_file, cfg)

    # Command
    cmd = [sys.executable, "run.py", cfg_file.as_posix()]
    CONSOLE.print("Run command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp_dir)


def run_with_device(device_pool: queue.Queue, **kwargs):
    # in theory, it shouldn't be blocked here because the num_workers == num_devices.
    device: int = device_pool.get_nowait()
    try:
        return run(device=device, **kwargs)
    finally:
        # always return, even on exception
        device_pool.put(device)


def parse_configs(cfg: dict):
    root = Path(cfg.pop("root"))
    dataset_root = Path(cfg.pop("dataset_root"))
    prefix: str = cfg.pop("prefix")
    degradations: Optional[list[str]] = cfg.pop("degradations", None)
    split_subset: bool = cfg.pop("split_subset", False)
    configs: list[dict] = cfg.pop("configs")

    if degradations is None:
        degradations = ["degraded"]

    for config in configs:
        dataset_name = config["name"]
        path = root / f"{prefix}_{dataset_name}.yaml"

        for split in config["splits"]:
            if "degraded" in split:
                for degradation in degradations:
                    if not split_subset and degradation != "degraded":
                        data = f"{dataset_name}_{degradation}.yaml"
                    else:
                        data = f"{dataset_name}.yaml"
                    data = dataset_root / data

                    subset = Subset(
                        split,
                        split,
                        robust="degraded" in split,
                        split_subset=split_subset,
                        degradation=degradation,
                    )

                    if (
                        subset.robust
                        and subset.degradation != "degraded"
                        and subset.split_subset
                    ):
                        subset.name = subset.name.replace(
                            "degraded",
                            "_".join(reversed(degradation.split("_"))) + "_degraded",
                        )
                    yield path, dict(dataset=Dataset(data, subset))
            else:
                data = dataset_root / f"{dataset_name}.yaml"
                subset = Subset(split, split)
                yield path, dict(dataset=Dataset(data, subset))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cfg", type=str, default="configs/tests.yaml", help="Path to the config."
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    os.environ.pop("DISPLAY", None)
    args = parse_args()
    with open(args.cfg, "r") as f:
        cfg: dict = yaml.safe_load(f)

    model = Path(cfg.pop("model"))
    assert model.is_file()
    name = cfg.pop("name")
    devices: list[int] = cfg.pop("devices")
    assert len(devices) > 0

    device_pool = queue.Queue()
    for device in devices:
        device_pool.put(device)
    with thread.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for i, (path, kwargs) in enumerate(parse_configs(cfg)):
            kwargs = dict(
                config=path, name=name, model=model.as_posix(), **kwargs, **cfg
            )
            futures.append(executor.submit(run_with_device, device_pool, **kwargs))
        for f in as_completed(futures):
            f.result()
