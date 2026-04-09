# ruff: noqa: F401, E402
def setup_environment():
    # Dummy setup function for preloading before detectron2 via DETECTRON2_ENV_MODULE env variable
    pass


from . import apis, configs, data, engine, evaluation, modeling, utils

__all__ = list(globals().keys())
