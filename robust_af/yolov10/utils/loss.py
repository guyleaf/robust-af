from typing import Any, Callable, Optional

import torch
import torch.nn as nn


class RobustDetectLoss:
    def __init__(
        self,
        model_loss: Callable[[Any, dict], tuple[torch.Tensor, torch.Tensor]],
        image_cst_loss: Optional[nn.Module] = None,
        cst_loss: Optional[nn.Module] = None,
        weight_dict: dict = dict(image_cst_loss=20, cst_loss=20),
    ):
        self.weight_dict = weight_dict

        self.model_loss = model_loss
        self.image_cst_loss = image_cst_loss
        self.cst_loss = cst_loss

    def compute_image_cst_loss(
        self,
        rhss: list[torch.Tensor],
        clear_rhss: list[torch.Tensor],
        prefix: str = "image_cst_loss",
    ) -> torch.Tensor:
        loss: torch.Tensor = self.image_cst_loss(rhss, clear_rhss)
        # mean over the feature dims & mean over batch_size
        indices = list(range(1, loss.ndim))
        loss = loss.mean(indices).mean()
        return loss * self.weight_dict[prefix]

    def compute_cst_loss(
        self,
        rhss: list[torch.Tensor],
        clear_rhss: list[torch.Tensor],
        prefix: str = "cst_loss",
    ) -> torch.Tensor:
        global_weight = self.weight_dict.get(prefix)
        if global_weight is not None:
            weights = {f"{prefix}_{i}": global_weight for i in range(len(rhss))}
        else:
            weights = self.weight_dict

        loss_items = rhss[0].new_empty((len(rhss)))
        for i, (rhs, clear_rhs) in enumerate(zip(rhss, clear_rhss)):
            loss: torch.Tensor = self.cst_loss(rhs, clear_rhs)
            # mean over the feature dims & mean over batch_size
            indices = list(range(1, loss.ndim))
            loss = loss.mean(indices).mean()
            loss_items[i] = loss * weights[f"{prefix}_{i}"]
        return loss_items

    def __call__(self, preds: dict, batch: dict):
        loss, loss_items = self.model_loss(preds["preds"], batch)
        batch_size = batch["img"].shape[0]

        # Follow the ultralytics design, they usually multiply loss with batch_size. (I still don't understand why they do that)
        # related issue: https://github.com/ultralytics/ultralytics/issues/3282#issuecomment-1813494534

        if self.image_cst_loss is not None:
            rhss: list[torch.Tensor] = preds["image_rhss"]
            clear_rhss: list[torch.Tensor] = preds["clear_image_rhss"]
            cst_loss = self.compute_image_cst_loss(rhss, clear_rhss)
            loss += cst_loss * batch_size
            loss_items = torch.cat((loss_items, cst_loss.detach()))

        if self.cst_loss is not None:
            rhss: list[torch.Tensor] = preds["rhss"]
            clear_rhss: list[torch.Tensor] = preds["clear_rhss"]
            cst_loss_items = self.compute_cst_loss(rhss, clear_rhss)
            loss += cst_loss_items.sum() * batch_size
            loss_items = torch.cat((loss_items, cst_loss_items.detach()))
        return loss, loss_items
