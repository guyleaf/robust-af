# ruff: noqa: F401
from . import (
    register_dds,
    register_dut_anti_uav,
    register_robust_anti_uav,
    register_robust_anti_uav_low,
    register_robust_dut_anti_uav,
    register_robust_dut_anti_uav_low,
    register_uav_eagle,
)
from .coco import load_coco_json, register_coco_instances

__all__ = list(globals().keys())
