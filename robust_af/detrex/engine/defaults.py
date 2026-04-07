from detectron2.engine import default_setup as d2_default_setup
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 and robust_af logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    d2_default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(
        cfg.train.output_dir,
        distributed_rank=rank,
        name="robust_af.detrex",
        abbrev_name="robust_af",
    )
