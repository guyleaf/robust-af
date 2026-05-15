import argparse
import concurrent.futures.thread as thread
import os
import subprocess
import sys
import tempfile
from concurrent.futures import as_completed
from pathlib import Path

import yaml
from rich.console import Console

CONSOLE = Console()


def run(config: Path, name: str, split: str, **kwargs):
    splits = [item for item in split.replace("val", "").split("_") if len(item) != 0]
    splits = list(reversed(splits))
    if len(splits) != 0:
        suffix = "_".join(splits)
        name = f"{name}_{suffix}"

    with open(config, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    work_dir = Path(cfg["name"])
    work_dir = work_dir.with_name(name)
    cfg["name"] = work_dir.as_posix()
    cfg["split"] = split
    cfg.update(**kwargs)

    with tempfile.NamedTemporaryFile(
        prefix=f"{config.stem}_",
        suffix=".yaml",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as file:
        yaml.safe_dump(cfg, file, sort_keys=False, allow_unicode=True)

    # with open(config, "w", encoding="utf-8") as f:
    #     yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # Command
    cmd = [sys.executable, "run.py", file.name]
    CONSOLE.print("Run command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(file.name)


def parse_configs(cfg: dict):
    root = Path(cfg.pop("root"))
    prefix: str = cfg.pop("prefix")
    configs: list[dict] = cfg.pop("configs")

    for config in configs:
        dataset_name = config["name"]
        path = root / f"{prefix}_{dataset_name}.yaml"
        for split in config["splits"]:
            yield path, dict(split=split)


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

    # TODO: manage GPU queue?
    num_devices = len(devices)
    with thread.ThreadPoolExecutor(max_workers=num_devices) as executor:
        futures = []
        for i, (path, kwargs) in enumerate(parse_configs(cfg)):
            device = devices[i % num_devices]
            kwargs = dict(model=model.as_posix(), device=device, **kwargs, **cfg)
            futures.append(executor.submit(run, config=path, name=name, **kwargs))
        for f in as_completed(futures):
            f.result()
