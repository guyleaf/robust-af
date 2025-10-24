# ruff: noqa: F401
from . import data, nn, zoo

__all__ = list(globals().keys())


def setup_environment():
    # Dummy setup function for preloading before rtdetrv2 via RTDETRV2_ENV_MODULE env variable
    pass
