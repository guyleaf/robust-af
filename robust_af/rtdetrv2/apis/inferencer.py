import random
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from rich.progress import track
from rtdetrv2.core import BaseConfig, YAMLConfig, create
from rtdetrv2.data.dataset._dataset import DetDataset
from rtdetrv2.misc import setup_seed
from torch.utils.data import DataLoader, Dataset, Subset


class Inferencer:
    def __init__(self, cfg: Union[str, Path, BaseConfig], deploy: bool = True):
        if not isinstance(cfg, BaseConfig):
            cfg = YAMLConfig(str(cfg))

        if cfg.device:
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module = self._load_model(cfg)
        self.postprocessor: nn.Module = cfg.postprocessor

        if deploy:
            self.model = self.model.deploy()
            # always convert from label to category
            # because we know the category and name from the dataset
            # self.postprocessor = self.postprocessor.deploy()
            self.postprocessor = self.postprocessor.eval()
        else:
            self.model = self.model.eval()
            self.postprocessor = self.postprocessor.eval()

        self.model = self.model.to(self.device)
        self.postprocessor = self.postprocessor.to(self.device)

    def _load_model(self, cfg: BaseConfig):
        if cfg.resume:
            path = cfg.resume
            if path.startswith("http"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    path, map_location="cpu", weights_only=True
                )
            else:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)

            if "last_epoch" in checkpoint:
                last_epoch = checkpoint["last_epoch"]
                print(f"Load last_epoch: {last_epoch}")

            if cfg.use_ema and "ema" in checkpoint:
                print("Load checkpoint.ema")
                state = checkpoint["ema"]["module"]
            else:
                print("Load checkpoint.model")
                state = checkpoint["model"]
        else:
            raise AttributeError("Only support resume to load model.state_dict by now.")

        cfg.model.load_state_dict(state)
        print("Load model.state_dict")
        return cfg.model

    def prepare_dataset(
        self, dataset: Dataset, max_num_samples: int, shuffle: bool = False
    ):
        if shuffle:
            indices = range(len(dataset))
            if max_num_samples < len(dataset):
                indices = random.sample(indices, k=max_num_samples)
        else:
            indices = range(min(max_num_samples, len(dataset)))
        dataset = Subset(dataset, indices)
        return dataset

    def prepare_dataloader(
        self,
        cfg: YAMLConfig,
        max_num_samples: int,
        name: str = "val_dataloader",
        shuffle: bool = False,
        image_ids: Optional[list[int]] = None,
    ):
        cfg = deepcopy(cfg)

        # create dataset instance (hacky way)
        global_cfg = cfg.global_cfg
        global_cfg["tmp_dataset"] = dataset_cfg = cfg.yaml_cfg[name]["dataset"]
        dataset_cfg["image_ids"] = image_ids
        dataset: Dataset = create("tmp_dataset", global_cfg)
        dataset = self.prepare_dataset(dataset, max_num_samples, shuffle=shuffle)

        cfg.yaml_cfg[name]["dataset"] = dataset
        dataloader: DataLoader = getattr(cfg, name)
        assert isinstance(dataloader.dataset, Subset)
        return dataloader

    @torch.inference_mode()
    def forward(
        self,
        samples: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        label2category: Optional[dict[int, int]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Feed the inputs to the model.

        Returns:
            list[dict[str, torch.Tensor]]: The predictions of samples.
                If deploy=True, labels are raw.
                Otherwise, they are mapped to dataset categories via postprocessor by using label2category.
        """
        # TODO: move into preprocess and postprecess method?
        samples = samples.to(self.device)
        orig_target_sizes = torch.stack(
            [t["orig_size"].to(self.device) for t in targets], dim=0
        )

        outputs = self.model(samples)
        outputs = self.postprocessor(
            outputs, orig_target_sizes, label2category=label2category
        )

        # # if deploy=True
        # if isinstance(outputs, tuple):
        #     outputs = [
        #         dict(labels=lab, boxes=box, scores=sco)
        #         for lab, box, sco in zip(*outputs)
        #     ]
        return outputs

    def __call__(
        self,
        cfg: Union[str, Path, YAMLConfig],
        shuffle: bool = False,
        seed: Optional[int] = 2026,
        max_num_samples: int = 100,
        image_ids: Optional[list[int]] = None,
    ):
        """Infer images from dataloader

        Args:
            cfg (Union[str, Path, YAMLConfig]): Config. If it is a file path, load it and use the val dataloader.
            shuffle (bool, optional): Shuffle dataset before inference. Defaults to False.
            seed (Optional[int], optional): The seed for randomness. Defaults to 2026.
            max_num_samples (int, optional): Total number of images to be inferred. Defaults to 100.
            image_ids (Optional[list[int]], optional): Infer specific images.

        Yields:
            predictions (dict[str, torch.Tensor]): The predictions of the sample.
            target (dict): The metadata of the sample.
        """
        if seed is not None:
            setup_seed(seed)

        # build dataloader
        if not isinstance(cfg, YAMLConfig):
            cfg = YAMLConfig(str(cfg))
        dataloader = self.prepare_dataloader(
            cfg, max_num_samples, shuffle=shuffle, image_ids=image_ids
        )

        dataset = dataloader.dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        assert isinstance(dataset, DetDataset)
        label2category = dataset.label2category
        category2name = dataset.category2name

        for samples, targets in track(dataloader, description="Inference"):
            preds = self.forward(samples, targets, label2category=label2category)

            for i, (pred, target) in enumerate(zip(preds, targets)):
                target["image"] = samples[i]
                target["category2name"] = category2name
                yield pred, target
