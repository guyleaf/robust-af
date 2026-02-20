# ruff: noqa: F401, E402
def setup_environment():
    """setup function for preloading before ultralytics via ULTRALYTICS_ENV_MODULE env variable"""
    from functools import partial

    import ultralytics.data.utils as data_utils

    from ..utils import setup_environment

    setup_environment()

    # NOTE: hacky way, configure keys which will be resolved by path
    data_utils.check_det_dataset = partial(
        data_utils.check_det_dataset,
        resolve_paths=("train", "val", "test", "degraded_val", "degraded_test"),
    )
