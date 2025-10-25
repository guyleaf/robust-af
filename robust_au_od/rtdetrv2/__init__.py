# ruff: noqa: F401
from . import data, nn, solver, zoo

__all__ = list(globals().keys())


def setup_environment():
    """setup function for preloading before rtdetrv2 via RTDETRV2_ENV_MODULE env variable"""
    solver.setup_environment()
