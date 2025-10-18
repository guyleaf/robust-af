import argparse
from typing import Union

import torch
from detectron2.config.config import CfgNode
from detectron2.engine import default_setup as original_default_setup
from omegaconf import DictConfig, OmegaConf


def _try_get_key(cfg: Union[CfgNode, DictConfig], *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def default_setup(cfg: Union[CfgNode, DictConfig], args: argparse.Namespace):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    original_default_setup(cfg, args)

    # By default, disable TF32 to get consistent results between before and Ampere (and later) GPU devices
    allow_tf32 = _try_get_key(cfg, "ALLOW_TF32", "train.allow_tf32", default=False)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
