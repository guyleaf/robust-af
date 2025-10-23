import importlib.resources

from detectron2.config import LazyConfig


def get_config(config_path):
    """
    Returns a config object from a config_path.

    Args:
        config_path (str): config file name relative to robust_au_od.detrex's "configs/"
            directory, e.g., "common/train.py"

    Returns:
        omegaconf.DictConfig: a config object
    """
    path = importlib.resources.files("robust_au_od.detrex.configs").joinpath(
        config_path
    )
    with importlib.resources.as_file(path) as cfg_file:
        if not cfg_file.is_file():
            raise RuntimeError(
                "{} not available in robust_au_od.detrex configs!".format(config_path)
            )
        cfg = LazyConfig.load(str(cfg_file))
    return cfg
