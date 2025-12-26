# ruff: noqa: F401, E402
def setup_environment():
    """setup function for preloading before rtdetrv2 via RTDETRV2_ENV_MODULE env variable"""
    from . import solver

    solver.setup_environment()


from . import data, nn, zoo

__all__ = list(globals().keys())
