from ultralytics.cfg import MODES, TASKS, check_dict_alignment, entrypoint
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    IterableSimpleNamespace,
    checks,
    yaml_load,
)

# Default configuration
DEFAULT_CFG_DICT = dict(
    **DEFAULT_CFG_DICT,
    robust=False,
    degradation=dict(
        enabled=False,
        # whether always apply on test (YOLODataset only)
        always=False,
        seed=2025,
        identity=True,
        ignored_degradations=[],
    ),
    sync_bn=False,
)
DEFAULT_ROBUST_CFG_DICT = dict(
    **DEFAULT_CFG_DICT,
    cst_loss=dict(module="nn.MSELoss", weight=20),
    image_cst_loss=None,
    freeze_bn=True,
)
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
DEFAULT_ROBUST_CFG = IterableSimpleNamespace(**DEFAULT_ROBUST_CFG_DICT)


def validate_global_cfg(cfg: dict):
    f"""Referenced from {entrypoint}"""
    full_args_dict = {
        **DEFAULT_CFG_DICT,
        **DEFAULT_ROBUST_CFG_DICT,
        **{k: None for k in TASKS},
        **{k: None for k in MODES},
    }

    # Check keys
    check_dict_alignment(full_args_dict, cfg)

    # Mode
    mode = cfg.get("mode")
    if mode is None:
        raise ValueError(f"'mode' argument is missing. Valid modes are {MODES}.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.")

    # Task
    task = cfg.get("task")
    if task:
        if task not in TASKS:
            raise ValueError(f"Invalid 'task={task}'. Valid tasks are {TASKS}.")

    # Model
    model = cfg.get("model")
    if model is None:
        raise ValueError("'model' argument is missing.")

    # Mode
    if mode in ("predict", "track") and "source" not in cfg:
        raise ValueError("'source' argument is missing.")
    elif mode in ("train", "val"):
        if "data" not in cfg and "resume" not in cfg:
            raise ValueError("'data' argument is missing.")
    elif mode == "export":
        if "format" not in cfg:
            raise ValueError("'format' argument is missing.")


def load_global_cfg(path: str):
    f"""Referenced from {entrypoint}"""
    cfg = {
        k: val for k, val in yaml_load(checks.check_yaml(path)).items() if k != "cfg"
    }
    validate_global_cfg(cfg)
    return cfg
