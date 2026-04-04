import random
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import detrex.modeling.ema as ema
import torch
import torch.nn as nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetFromList
from detectron2.utils.env import seed_all_rng
from omegaconf import DictConfig
from rich.progress import track
from torch.utils.data import DataLoader, Dataset, Subset


class Inferencer:
    def __init__(self, cfg: Union[str, Path, DictConfig]) -> None:
        if not isinstance(cfg, DictConfig):
            cfg = LazyConfig.load(cfg)
            assert isinstance(cfg, DictConfig)

        self.device = cfg.train.device
        model = self._load_model(cfg)
        self.model = model.eval().to(self.device)
        self.cfg = cfg

    def _load_model(self, cfg: DictConfig) -> nn.Module:
        model = instantiate(cfg.model)
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(
            cfg.train.init_checkpoint
        )
        # Apply ema state for evaluation
        if (
            cfg.train.model_ema.enabled
            and cfg.train.model_ema.use_ema_weights_for_eval_only
        ):
            ema.apply_model_ema(model)
        return model

    def prepare_dataset(
        self, cfg: DictConfig, max_num_samples: int, shuffle: bool = False
    ):
        dataset: Union[list[dict], Dataset] = instantiate(cfg)

        if isinstance(dataset, list):
            dataset = DatasetFromList(dataset, copy=False)

        if shuffle:
            indices = range(len(dataset))
            if max_num_samples < len(dataset):
                indices = random.sample(indices, k=max_num_samples)
        else:
            indices = range(min(max_num_samples, len(dataset)))
        dataset = Subset(dataset, indices)
        return dataset

    def prepare_dataloader(
        self, cfg: DictConfig, max_num_samples: int, shuffle: bool = False
    ) -> DataLoader:
        cfg = deepcopy(cfg)
        cfg.dataset = self.prepare_dataset(
            cfg.dataset, max_num_samples, shuffle=shuffle
        )
        return instantiate(cfg)

    @torch.inference_mode()
    def forward(self, batch_inputs: list[dict]) -> list[dict]:
        """Feed the inputs to the model."""
        return self.model(batch_inputs)

    def __call__(
        self,
        dataloader_cfg: Union[str, Path, DictConfig],
        shuffle: bool = False,
        seed: Optional[int] = 2026,
        max_num_samples: int = 100,
    ):
        """Infer images from dataloader

        Args:
            dataloader_cfg (Union[str, Path, DictConfig]): Dataloader config. If it is a file path, load it and use the test dataloader.
            seed (Optional[int], optional): The seed for randomness. Defaults to 2026.
            shuffle (bool, optional): Shuffle dataset before inference. Defaults to False.
            max_num_samples (int, optional): Total number of images to be inferred. Defaults to 100.

        Yields:
            predictions (dict): The predictions of the sample.
            sample (dict): The metadata of the sample.
        """
        seed_all_rng(seed)

        # build dataloader
        if not isinstance(dataloader_cfg, DictConfig):
            dataloader_cfg = LazyConfig.load(dataloader_cfg).dataloader.test
        dataloader = self.prepare_dataloader(
            dataloader_cfg, max_num_samples, shuffle=shuffle
        )

        for batch_inputs in track(dataloader, description="Inference"):
            batch_inputs: list[dict]
            batch_preds = self.forward(batch_inputs)

            for i, preds in enumerate(batch_preds):
                sample = batch_inputs[i]
                yield preds, sample
