import argparse
import concurrent.futures.thread as thread
import os
import shutil
import subprocess
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
    path: Path
    robust: bool = False
    split_subset: bool = False
    degradation: Optional[str] = None


@dataclass
class Dataset:
    name: str
    subset: Subset


def save_cfg(path: Path, cfg: dict):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def resolve_paths(root: Path, paths: list[str]):
    for i, path in enumerate(paths):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
            path = path.resolve()
        assert path.is_file()
        paths[i] = path.as_posix()
    return paths


def run(
    config: Path,
    checkpoint: Path,
    name: str,
    dataset: Dataset,
    envs: Optional[dict[str, str]] = None,
    updates: dict = {},
):
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{config.stem}_"))

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

    if dataset.subset.robust and dataset.subset.split_subset:
        assert (
            dataset.subset.degradation is not None
            and dataset.subset.degradation != "degraded"
        )
        with open(dataset.subset.path, "r", encoding="utf-8") as f:
            dataset_cfg: dict = yaml.safe_load(f)

        dataset_cfg.setdefault("val_dataloader", {})
        dataset_cfg["val_dataloader"].setdefault("dataset", {})
        dataset_cfg["val_dataloader"]["dataset"]["degradations"] = [
            dataset.subset.degradation
        ]

        # resolve all paths because we save the config file to /tmp
        dataset_cfg["__include__"] = resolve_paths(
            dataset.subset.path.parent, dataset_cfg["__include__"]
        )

        dataset.subset.path = tmp_dir / dataset.subset.path.name
        save_cfg(dataset.subset.path, dataset_cfg)

    # find & replace the dataset path in __include__
    includes: list[str] = cfg["__include__"]
    assert "dataset" in includes[0]
    includes[0] = dataset.subset.path.as_posix()

    # resolve all paths because we save the config file to /tmp
    cfg["__include__"] = resolve_paths(config.parent, includes)

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

    cfg_path = tmp_dir / config.name
    save_cfg(cfg_path, cfg)

    # prepare envs
    if envs is not None:
        new_envs = os.environ.copy()
        new_envs.update(envs)
        envs = new_envs

    # command
    cmd = ["bash", "test.sh", cfg_path.as_posix(), checkpoint.as_posix()]
    CONSOLE.print("Run command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=envs)
    finally:
        shutil.rmtree(tmp_dir)


def parse_configs(cfg: dict):
    root = Path(cfg.pop("root"))
    dataset_root = Path(cfg.pop("dataset_root")).resolve()
    prefix: str = cfg.pop("prefix")
    degradations: Optional[list[str]] = cfg.pop("degradations", None)
    split_subset: bool = cfg.pop("split_subset", False)
    configs: list[dict] = cfg.pop("configs")

    if degradations is None:
        degradations = ["degraded"]

    for config in configs:
        dataset_name = config["name"]
        path = root / f"{prefix}_{dataset_name}.yml"

        for degradation in degradations:
            if not split_subset and degradation != "degraded":
                new_dataset_name = f"{dataset_name}_{degradation}"
            else:
                new_dataset_name = dataset_name

            for subset_name, subset_path in config["subsets"].items():
                subset_path = Path(subset_path)
                if not subset_path.is_absolute():
                    subset_path = dataset_root / subset_path

                subset = Subset(
                    subset_name,
                    subset_path,
                    robust="degraded" in subset_path.name,
                    split_subset=split_subset,
                    degradation=degradation,
                )

                if subset.robust:
                    if degradation != "degraded":
                        if subset.split_subset:
                            subset.name = subset.name.replace(
                                "degraded", f"degraded_{degradation}"
                            )
                        else:
                            # switch to the specific degradation type if the subset is degraded version.
                            subset_name = subset.path.name.replace(
                                "degraded", f"degraded_{degradation}"
                            )
                            subset.path = subset.path.with_name(subset_name)

                assert subset.path.is_file(), subset.path
                yield path, dict(dataset=Dataset(new_dataset_name, subset))


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
