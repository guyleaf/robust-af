#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""

import copy
import logging
import os
import sys
import time
import warnings

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detrex.utils import WandbWriter
from torch.nn.parallel import DataParallel, DistributedDataParallel

from robust_u2u_od.detrex.engine import BestCheckpointer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

logger = logging.getLogger(__name__)


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = (
            "AMPTrainer does not support single-process multi-device training!"
        )
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


def do_test(cfg, model, training=False):
    if "evaluator" not in cfg.dataloader:
        return

    ret = inference_on_dataset(
        model,
        instantiate(cfg.dataloader.test),
        instantiate(cfg.dataloader.evaluator),
    )
    logger.info("Subset: test")
    print_csv_format(ret)

    if training:
        test_dataloader = copy.deepcopy(cfg.dataloader.test)
        cfg.dataloader.test.dataset.names = "${...train.dataset.names}"
        # cfg.dataloader.evaluator.dataset_name = "${..test.dataset.names}"
        train_ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        logger.info("Subset: train")
        print_csv_format(train_ret)
        cfg.dataloader.test = test_dataloader

        # keep compatibility with BestCheckpointer
        for k, v in ret.items():
            v["train"] = train_ret[k]
    return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    if cfg.train.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # suppress UserWarning in SyncBatchNorm due to using the private function, _all_gather_base.
        warnings.filterwarnings(
            "ignore",
            message=r"torch\.distributed\._all_gather_base",
            category=UserWarning,
        )

    if args.hacked:
        # this is an hack of train_net
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not match_name_keywords(n, ["backbone"])
                    and not match_name_keywords(
                        n, ["reference_points", "sampling_offsets"]
                    )
                    and p.requires_grad
                ],
                "lr": 2e-4,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if match_name_keywords(n, ["backbone"]) and p.requires_grad
                ],
                "lr": 2e-5,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if match_name_keywords(n, ["reference_points", "sampling_offsets"])
                    and p.requires_grad
                ],
                "lr": 2e-5,
            },
        ]
        optim = torch.optim.AdamW(param_dicts, 2e-4, weight_decay=1e-4)
    else:
        # instantiate optimizer
        cfg.optimizer.params.model = model
        optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params
        if cfg.train.clip_grad.enabled
        else None,
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    if comm.is_main_process():
        writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
        if cfg.train.wandb.enabled:
            PathManager.mkdirs(cfg.train.wandb.params.dir)
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(
                cfg.train.eval_period, lambda: do_test(cfg, model, training=True)
            ),
            hooks.PeriodicWriter(writers, period=cfg.train.log_period)
            if comm.is_main_process()
            else None,
            BestCheckpointer(
                cfg.train.eval_period, checkpointer, **cfg.train.best_checkpointer
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    # Enable fast debugging by running several iterations to check for any bugs.
    if cfg.train.fast_dev_run.enabled:
        cfg.train.max_iter = 20
        cfg.train.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--hacked", action="store_true", help="Use hacked version of DINO."
    )
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
