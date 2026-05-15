import argparse
import concurrent.futures.thread as thread
import os
import subprocess
import tempfile
from concurrent.futures import as_completed
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
    split: str  # custom key appended to the output_dir
    name: str  # registered subset name
    robust: bool = False  # use the robust version of dataloader


@dataclass
class Dataset:
    name: str  # registered dataset name
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
        for item in dataset.subset.split.replace("val", "").split("_")
        if len(item) != 0
    ]
    if len(splits) != 0:
        suffix = "_".join(splits)
        name = f"{name}_{suffix}"

    cfg = LazyConfig.load(config.as_posix())
    assert isinstance(cfg, DictConfig)

    metadata = MetadataCatalog.get(dataset.name)
    cfg.model.num_classes = metadata.num_classes

    # switch dataloader and change dataset name
    dataset_cfg = get_config(f"datasets/{dataset.name}_detr.py")
    if dataset.subset.robust:
        dataloader_cfg = dataset_cfg.robust_dataloader
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


def parse_configs(cfg: DictConfig):
    config = Path(cfg.config)
    datasets: list[dict] = cfg.datasets

    for dataset in datasets:
        metadata = Dataset(dataset["name"])
        for name, subset in dataset["subsets"].items():
            metadata = Dataset(dataset["name"], Subset(name, **subset))
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

    # TODO: manage GPU queue?
    num_devices = len(devices)
    with thread.ThreadPoolExecutor(max_workers=num_devices) as executor:
        futures = []
        for i, (path, kwargs) in enumerate(parse_configs(config)):
            envs = dict(CUDA_VISIBLE_DEVICES=str(devices[i % num_devices]))
            kwargs = dict(
                config=path,
                checkpoint=checkpoint,
                name=name,
                envs=envs,
                updates=cfg,
                **kwargs,
            )
            futures.append(executor.submit(run, **kwargs))
        for f in as_completed(futures):
            f.result()
