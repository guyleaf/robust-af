import argparse
import concurrent.futures.thread as thread
import os
import subprocess
import tempfile
from concurrent.futures import as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

CONSOLE = Console()


@dataclass
class Subset:
    name: str
    path: Path


@dataclass
class Dataset:
    name: str
    subset: Optional[Subset] = None


def run(
    config: Path,
    checkpoint: Path,
    name: str,
    dataset: Dataset,
    envs: Optional[dict[str, str]] = None,
    updates: dict = {},
):
    splits = [
        item
        for item in dataset.subset.name.replace("val", "").split("_")
        if len(item) != 0
    ]
    if len(splits) != 0:
        suffix = "_".join(splits)
        name = f"{name}_{suffix}"

    with open(config, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    # find & replace the dataset path in __include__
    includes: list[str] = cfg["__include__"]
    assert "dataset" in includes[0]
    includes[0] = dataset.subset.path.as_posix()

    # resolve all paths because we save the config file to /tmp
    for i, include in enumerate(includes):
        include = Path(include)
        if not include.is_absolute():
            include = config.parent / include
            include = include.resolve()
        assert include.is_file()
        includes[i] = include.as_posix()

    work_dir = Path(cfg["output_dir"])
    # concat with folder of training dataset
    # test structure: <training dataset>/<testing dataset>/<exp_name>
    work_dir = work_dir.parent.parent / dataset.name / name
    cfg["output_dir"] = work_dir.as_posix()

    # clear adapter config to avoid any side effects from the test config
    tmp = cfg.get("RobustRTDETR", None)
    if tmp is not None:
        tmp.pop("robust_module", None)
        tmp.pop("robust_image_module", None)
    cfg.pop("MultiScaleProcessor", None)
    # TODO: clear these based on adapters.py
    cfg.pop("DIP", None)
    cfg.pop("GDIP", None)
    cfg.pop("DENet", None)
    cfg.update(updates)

    with tempfile.NamedTemporaryFile(
        prefix=f"{config.stem}_",
        suffix=".yml",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as file:
        yaml.safe_dump(cfg, file, sort_keys=False, allow_unicode=True)

    # prepare envs
    if envs is not None:
        new_envs = os.environ.copy()
        new_envs.update(envs)
        envs = new_envs

    # command
    cmd = ["bash", "test.sh", file.name, checkpoint.as_posix()]
    CONSOLE.print("Run command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=envs)
    finally:
        os.unlink(file.name)


def parse_configs(cfg: dict):
    root = Path(cfg.pop("root"))
    dataset_root = Path(cfg.pop("dataset_root")).resolve()
    prefix: str = cfg.pop("prefix")
    degradations: Optional[list[str]] = cfg.pop("degradations", None)
    configs: list[dict] = cfg.pop("configs")

    if degradations is None:
        degradations = ["degraded"]

    for config in configs:
        dataset_name = config["name"]
        path = root / f"{prefix}_{dataset_name}.yml"

        for degradation in degradations:
            if degradation != "degraded":
                new_dataset_name = f"{dataset_name}_{degradation}"
            else:
                new_dataset_name = dataset_name
            origin_metadata = Dataset(new_dataset_name)

            for subset, subset_path in config["subsets"].items():
                subset_path = Path(subset_path)
                # switch to the specific degradation type if the subset is degraded version.
                if "degraded" in subset_path.name and degradation != "degraded":
                    subset_name = subset_path.name.replace(
                        "degraded", f"degraded_{degradation}"
                    )
                    subset_path = subset_path.with_name(subset_name)

                if not subset_path.is_absolute():
                    subset_path = dataset_root / subset_path
                assert subset_path.is_file(), subset_path

                metadata = deepcopy(origin_metadata)
                metadata.subset = Subset(subset, subset_path)
                yield path, dict(dataset=metadata)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cfg", type=str, default="configs/tests.yml", help="Path to the config."
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    os.environ.pop("DISPLAY", None)
    args = parse_args()
    with open(args.cfg, "r") as f:
        cfg: dict = yaml.safe_load(f)

    checkpoint = Path(cfg.pop("checkpoint"))
    assert checkpoint.is_file()
    name = cfg.pop("name")
    max_runs: int = cfg.pop("max_runs")
    assert max_runs > 0
    envs = cfg.pop("envs")

    envs = {k: str(v) for k, v in envs.items()}
    with thread.ThreadPoolExecutor(max_workers=max_runs) as executor:
        futures = []
        for i, (path, kwargs) in enumerate(parse_configs(cfg)):
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
