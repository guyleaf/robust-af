from copy import deepcopy
from functools import partial

import ultralytics.utils.dist as dist
from ultralytics.models.yolov10.train import (
    YOLOv10DetectionTrainer as ORIGINAL_YOLOv10DetectionTrainer,
)
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import de_parallel

from robust_u2u_od.utils import ReproducibleRandomContext
from robust_u2u_od.yolov10.utils import update_batch_norm_mode

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
        if self.args.eval_bn_on_freeze:
            # wrap the train method to set BatchNorm2d to eval mode if frozon
            freeze_list = (
                self.args.freeze
                if isinstance(self.args.freeze, list)
                else list(range(self.args.freeze))
                if isinstance(self.args.freeze, int)
                else []
            )

            def _train(self: RobustYOLOv10DetectionModel, mode: bool = True):
                super(RobustYOLOv10DetectionModel, self).train(mode)
                if mode:
                    for i in freeze_list:
                        update_batch_norm_mode(self.model[i], False)
                    if verbose:
                        LOGGER.info("Set BatchNorms in freeze list to eval mode!")
                return self

            RobustYOLOv10DetectionModel.train = _train

        model = RobustYOLOv10DetectionModel(
            cfg,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
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
        plot_images(
            images=batch["clear_img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}_clear.jpg",
            on_plot=self.on_plot,
        )
