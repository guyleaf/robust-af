from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from rtdetrv2.core import register


@register()
class RobustCriterion(nn.Module):
    __inject__ = ["model_loss", "image_cst_loss", "cst_loss"]

    def __init__(
        self,
        model_loss: nn.Module,
        image_cst_loss: Optional[nn.Module] = None,
        cst_loss: Optional[nn.Module] = None,
        weight_dict: dict = dict(loss_image_cst=20, loss_cst=20),
    ):
        super().__init__()
        self.weight_dict = weight_dict

        self.model_loss = model_loss
        self.image_cst_loss = image_cst_loss
        self.cst_loss = cst_loss

    def loss_image_cst(
        self,
        rhss: torch.Tensor,
        clear_rhss: torch.Tensor,
        targets: list[Union[dict, Sequence[dict]]],
        prefix: str = "loss_image_cst",
    ):
        weight: float = self.weight_dict.get(prefix)
        cst_loss: torch.Tensor = self.image_cst_loss(rhss, clear_rhss)
        # mean over the feature dims & mean over batch_size
        indices = list(range(1, cst_loss.ndim))
        cst_loss = cst_loss.mean(indices).mean()
        return {prefix: cst_loss * weight}

    def loss_cst(
        self,
        rhss: list[torch.Tensor],
        clear_rhss: list[torch.Tensor],
        targets: list[Union[dict, Sequence[dict]]],
        prefix: str = "loss_cst",
    ):
        global_weight = self.weight_dict.get(prefix)
        if global_weight is not None:
            weights = {f"{prefix}_{i}": global_weight for i in range(len(rhss))}
        else:
            weights = self.weight_dict

        cst_losses = {}
        for i, (rhs, clear_rhs) in enumerate(zip(rhss, clear_rhss)):
            cst_loss: torch.Tensor = self.cst_loss(rhs, clear_rhs)
            # mean over the feature dims & mean over batch_size
            indices = list(range(1, cst_loss.ndim))
            cst_loss = cst_loss.mean(indices).mean()
            cst_losses[f"{prefix}_{i}"] = cst_loss * weights[f"{prefix}_{i}"]
        return cst_losses

    def forward(self, outputs: dict, targets: list[dict], **kwargs):
        """
        This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = self.model_loss(outputs, targets, **kwargs)

        if self.image_cst_loss is not None:
            rhss = outputs["image_rhss"]
            clear_rhss = outputs["clear_image_rhss"]
            cst_losses = self.loss_image_cst(rhss, clear_rhss, targets)
            losses.update(cst_losses)

        if self.cst_loss is not None:
            rhss = outputs["rhss"]
            clear_rhss = outputs["clear_rhss"]
            cst_losses = self.loss_cst(rhss, clear_rhss, targets)
            losses.update(cst_losses)
        return losses


@register()
class RobustCriterionv2(nn.Module):
    __inject__ = ["model_loss", "image_content_loss", "content_loss", "style_loss"]

    def __init__(
        self,
        model_loss: nn.Module,
        style_loss: nn.Module,
        image_content_loss: Optional[nn.Module] = None,
        content_loss: Optional[nn.Module] = None,
        weight_dict: dict = dict(loss_image_content=20, loss_content=20, loss_style=20),
    ):
        super().__init__()
        self.weight_dict = weight_dict

        self.model_loss = model_loss
        self.image_content_loss = image_content_loss
        self.content_loss = content_loss
        self.style_loss = style_loss

    def loss_image_content(
        self,
        rhss: list[torch.Tensor],
        clear_rhss: list[torch.Tensor],
        prefix: str = "loss_image_content",
    ):
        weight = self.weight_dict.get(prefix)
        loss: torch.Tensor = self.image_content_loss(rhss, clear_rhss)
        # mean over the feature dims & mean over batch_size
        indices = list(range(1, loss.ndim))
        loss = loss.mean(indices).mean()
        return {prefix: loss * weight}

    def loss_content(
        self,
        rhss: list[torch.Tensor],
        clear_rhss: list[torch.Tensor],
        prefix: str = "loss_content",
    ):
        global_weight = self.weight_dict.get(prefix)
        if global_weight is not None:
            weights = {f"{prefix}_{i}": global_weight for i in range(len(rhss))}
        else:
            weights = self.weight_dict

        losses = {}
        for i, (rhs, clear_rhs) in enumerate(zip(rhss, clear_rhss)):
            loss: torch.Tensor = self.content_loss(rhs, clear_rhs)
            # mean over the feature dims & mean over batch_size
            indices = list(range(1, loss.ndim))
            loss = loss.mean(indices).mean()
            losses[f"{prefix}_{i}"] = loss * weights[f"{prefix}_{i}"]
        return losses

    def loss_style(
        self,
        rhs: list[torch.Tensor],
        clear_rhs: list[torch.Tensor],
        prefix: str = "loss_style",
    ):
        global_weight = self.weight_dict.get(prefix)
        if global_weight is not None:
            weights = {f"{prefix}_{i}": global_weight for i in range(len(rhs))}
        else:
            weights = self.weight_dict

        losses = {}
        for i, (feats, clear_feats) in enumerate(zip(rhs, clear_rhs)):
            # calculate statistics of feature maps for each channel
            indices = list(range(2, feats.ndim))
            # [B, C]
            stds, means = torch.std_mean(feats, dim=indices, correction=0)
            clear_stds, clear_means = torch.std_mean(
                clear_feats, dim=indices, correction=0
            )

            # calculate style loss
            # [B, C]
            loss_means: torch.Tensor = self.style_loss(means, clear_means)
            loss_stds: torch.Tensor = self.style_loss(stds, clear_stds)
            loss = loss_means + loss_stds

            # mean over the channels & mean over batch_size
            loss = loss.mean(dim=1).mean()
            losses[f"{prefix}_{i}"] = loss * weights[f"{prefix}_{i}"]
        return losses

    def forward(self, outputs: dict, targets: list[dict], **kwargs):
        """
        This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = self.model_loss(outputs, targets, **kwargs)

        if self.image_content_loss is not None:
            rhss = outputs["image_rhss"]
            clear_rhss = outputs["clear_image_rhss"]
            content_losses = self.loss_image_content(rhss, clear_rhss)
            losses.update(content_losses)

        if self.content_loss is not None:
            rhss = outputs["rhss"]
            clear_rhss = outputs["clear_rhss"]
            content_losses = self.loss_content(rhss, clear_rhss)
            losses.update(content_losses)

            style_losses = self.loss_style(rhss, clear_rhss)
            losses.update(style_losses)
        return losses
