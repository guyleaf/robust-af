import math
import time
import warnings
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import ultralytics.utils.dist as dist
from ultralytics.models.yolov10.train import (
    YOLOv10DetectionTrainer as ORIGINAL_YOLOv10DetectionTrainer,
)
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import de_parallel

from ...utils import ReproducibleRandomContext, convert_to_frozen_batchnorm_2d
from ..data import build_yolo_dataset
from ..nn.tasks import RobustYOLOv10DetectionModel
from ..utils import (
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_ROBUST_CFG,
    DEFAULT_ROBUST_CFG_DICT,
)
from .val import RobustYOLOv10DetectionValidator, YOLOv10DetectionValidator


class YOLOv10DetectionTrainer(ORIGINAL_YOLOv10DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        # A workaround to configure the file generation function for DDP
        dist.generate_ddp_file = partial(
            dist.generate_ddp_file, default_cfg=DEFAULT_CFG_DICT
        )

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = (
            max(round(self.args.warmup_epochs * nb), 100)
            if self.args.warmup_epochs > 0
            else -1
        )  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for "
            + (
                f"{self.args.time} hours..."
                if self.args.time
                else f"{self.epochs} epochs..."
            )
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(
                        1,
                        int(
                            np.interp(
                                ni, xi, [1, self.args.nbs / self.batch_size]
                            ).round()
                        ),
                    )
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if j == 0 else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni, xi, [self.args.warmup_momentum, self.args.momentum]
                            )

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1)
                        if self.tloss is not None
                        else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (
                            self.args.time * 3600
                        )
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(
                                broadcast_list, 0
                            )  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            mem,
                            *losses,
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {
                f"lr/pg{ir}": x["lr"]
                for ir, x in enumerate(self.optimizer.param_groups)
            }  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in (-1, 0):
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(
                    self.model,
                    include=["yaml", "nc", "args", "names", "stride", "class_weights"],
                )

                # Validation
                if (
                    (
                        self.args.val
                        and (
                            ((epoch + 1) % self.args.val_period == 0)
                            or (self.epochs - epoch) <= 10
                        )
                    )
                    or final_epoch
                    or self.stopper.possible_stop
                    or self.stop
                ):
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(
                    metrics={
                        **self.label_loss_items(self.tloss),
                        **self.metrics,
                        **self.lr,
                    }
                )
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (
                        self.args.time * 3600
                    )

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (
                        epoch - self.start_epoch + 1
                    )
                    self.epochs = self.args.epochs = math.ceil(
                        self.args.time * 3600 / mean_epoch_time
                    )
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.scheduler.step()
            self.run_callbacks("on_fit_epoch_end")
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(
                    broadcast_list, 0
                )  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = (
            "box_om",
            "cls_om",
            "dfl_om",
            "box_oo",
            "cls_oo",
            "dfl_oo",
        )
        return YOLOv10DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
        )


class RobustYOLOv10DetectionTrainer(YOLOv10DetectionTrainer):
    def __init__(self, cfg=DEFAULT_ROBUST_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # A workaround to configure the file generation function for DDP
        dist.generate_ddp_file = partial(
            dist.generate_ddp_file, default_cfg=DEFAULT_ROBUST_CFG_DICT
        )

    def preprocess_batch(self, batch: dict):
        """Preprocesses a batch of images by scaling and converting to float."""
        with ReproducibleRandomContext():
            batch = super().preprocess_batch(batch)
        batch["clear"] = super().preprocess_batch(batch["clear"])
        return batch

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            robust=True,
        )

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = (
            "box_om",
            "cls_om",
            "dfl_om",
            "box_oo",
            "cls_oo",
            "dfl_oo",
        ) + tuple(
            f"cst_{i}" for i in range(len(de_parallel(self.model).yaml["robust"]))
        )
        return RobustYOLOv10DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = RobustYOLOv10DetectionModel(
            cfg,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        if self.args.freeze_bn:
            if isinstance(self.args.freeze_bn, bool):
                freeze_list = self.args.freeze
            elif isinstance(self.args.freeze_bn, (int, list)):
                freeze_list = self.args.freeze_bn
            else:
                raise ValueError(
                    "The freeze_bn argument only supports boolean, int or list."
                )

            freeze_list = (
                freeze_list
                if isinstance(freeze_list, list)
                else list(range(freeze_list))
                if isinstance(freeze_list, int)
                else []
            )
            for i in freeze_list:
                # NOTE: currently, only supports freezing BatchNorm2d
                model.model[i] = convert_to_frozen_batchnorm_2d(model.model[i])

            if verbose:
                LOGGER.info("Freeze BatchNorms!")
        return model

    def plot_training_samples(self, batch: dict, ni: int):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

        clear_batch = batch["clear"]
        plot_images(
            images=clear_batch["img"],
            batch_idx=clear_batch["batch_idx"],
            cls=clear_batch["cls"].squeeze(-1),
            bboxes=clear_batch["bboxes"],
            paths=clear_batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}_clear.jpg",
            on_plot=self.on_plot,
        )
