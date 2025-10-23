# ruff: noqa: F401
from . import configs, data, engine, modeling, utils

__all__ = list(globals().keys())


def setup_environment():
    # Dummy setup function for preloading before detectron2 via DETECTRON2_ENV_MODULE env variable
    pass
