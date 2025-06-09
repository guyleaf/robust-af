import importlib.resources
import os

from detectron2.config import LazyConfig


def get_config(config_path):
    """
    Returns a config object from a config_path.

    Args:
        config_path (str): config file name relative to robust_u2u_od.detrex's "configs/"
            directory, e.g., "common/train.py"

    Returns:
        omegaconf.DictConfig: a config object
    """
    cfg_file = importlib.resources.files("robust_u2u_od.detrex.configs") / config_path
    if os.path.exists(cfg_file):
        raise RuntimeError(
            "{} not available in robust_u2u_od.detrex configs!".format(config_path)
        )
    cfg = LazyConfig.load(cfg_file)
    return cfg
