import argparse
import concurrent.futures.thread as thread
import os
import queue
import subprocess
import tempfile
from concurrent.futures import as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from detectron2.config import LazyConfig
from detectron2.data import MetadataCatalog
from omegaconf import DictConfig
from rich.console import Console

from robust_af.detrex.configs import get_config

CONSOLE = Console()


@dataclass
class Subset:
    name: str  # registered subset name
    output_name: str  # custom name appended to the output_dir
    robust: bool = False  # use the robust version of dataloader
    degradation: Optional[str] = None
    split_subset: bool = False


@dataclass
class Dataset:
    name: str  # registered dataset name
    metadata_name: str  # registered metadata name
    subset: Optional[Subset] = None


def run(
    config: Path,
    checkpoint: Path,
    name: str,
    dataset: Dataset,
    envs: Optional[dict[str, str]] = None,
    updates: Optional[DictConfig] = None,
):
    splits = [
        item
        for item in dataset.subset.output_name.replace("val", "").split("_")
        if len(item) != 0
    ]
    if len(splits) != 0:
        suffix = "_".join(splits)
        name = f"{name}_{suffix}"

    cfg = LazyConfig.load(config.as_posix())
    assert isinstance(cfg, DictConfig)

    metadata = MetadataCatalog.get(dataset.metadata_name)
    cfg.model.num_classes = metadata.num_classes

    # switch dataloader and change dataset name
    dataset_cfg = get_config(f"datasets/{dataset.name}_detr.py")
    if dataset.subset.robust:
        dataloader_cfg = dataset_cfg.robust_dataloader
        if dataset.subset.split_subset:
            assert (
                dataset.subset.degradation is not None
                and dataset.subset.degradation != "degraded"
            )
            dataloader_cfg.test.dataset.degradations = [dataset.subset.degradation]
    else:
        dataloader_cfg = dataset_cfg.dataloader
    dataloader_cfg.test.dataset.names = dataset.subset.name
    cfg.dataloader = dataloader_cfg

    # change work dir
    work_dir = Path(cfg.train.output_dir)
    # concat with folder of training dataset
    # test structure: <training dataset>/<testing dataset>/<exp_name>
    work_dir = work_dir.parent.parent / dataset.name / name
    cfg.train.output_dir = work_dir.as_posix()

    # clear adapter config to avoid any side effects from the test config
    # for example, Change GDIP to DIP => need to delete multi_level argument
    cfg.model.pop("robust_module", None)
    cfg.model.pop("robust_image_module", None)
    cfg.merge_with(updates)

    with tempfile.TemporaryDirectory(prefix=config.stem) as tmpdir:
        tmpdir = Path(tmpdir)
        out_file = tmpdir / "config.yaml"
        LazyConfig.save(cfg, out_file.as_posix())

        # prepare envs
        if envs is not None:
            new_envs = os.environ.copy()
            new_envs.update(envs)
            envs = new_envs

        # command
        cmd = ["bash", "test.sh", out_file.as_posix(), checkpoint.as_posix()]
        CONSOLE.print("Run command:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=envs)


def run_with_device(device_pool: queue.Queue, **kwargs):
    # in theory, it shouldn't be blocked here because the num_workers == num_devices.
    device: int = device_pool.get_nowait()
    try:
        envs = dict(CUDA_VISIBLE_DEVICES=str(device))
        return run(envs=envs, **kwargs)
    finally:
        # always return, even on exception
        device_pool.put(device)


def parse_configs(cfg: DictConfig):
    config = Path(cfg.config)
    datasets: list[dict] = cfg.datasets
    degradations: Optional[list[str]] = cfg.pop("degradations", None)
    split_subset: bool = cfg.pop("split_subset", False)

    if degradations is None:
        degradations = ["degraded"]

    for dataset in datasets:
        dataset_name = dataset["name"]
        for degradation in degradations:
            if not split_subset and degradation != "degraded":
                new_dataset_name = f"{dataset_name}_{degradation}"
            else:
                new_dataset_name = dataset_name
            origin_metadata = Dataset(new_dataset_name, dataset["name"])

            for name, subset in dataset["subsets"].items():
                metadata = deepcopy(origin_metadata)
                metadata.subset = Subset(
                    output_name=name,
                    **subset,
                    split_subset=split_subset,
                    degradation=degradation,
                )

                if metadata.subset.robust:
                    if degradation != "degraded":
                        if split_subset:
                            metadata.subset.output_name = (
                                metadata.subset.output_name.replace(
                                    "degraded", f"degraded_{degradation}"
                                )
                            )
                        else:
                            metadata.subset.name += f"_{degradation}"

                yield (config, dict(dataset=metadata))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cfg", type=str, default="configs/tests.py", help="Path to the config."
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    os.environ.pop("DISPLAY", None)
    args = parse_args()

    cfg = LazyConfig.load(args.cfg)
    assert isinstance(cfg, DictConfig)

    config = cfg.pop("config")
    checkpoint = Path(config.checkpoint)
    assert checkpoint.is_file()
    name = config.name
    devices: list[int] = config.devices
    assert len(devices) > 0

    device_pool = queue.Queue()
    for device in devices:
        device_pool.put(device)
    with thread.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for i, (path, kwargs) in enumerate(parse_configs(config)):
            kwargs = dict(
                config=path, checkpoint=checkpoint, name=name, updates=cfg, **kwargs
            )
            futures.append(executor.submit(run_with_device, device_pool, **kwargs))
        for f in as_completed(futures):
            f.result()
