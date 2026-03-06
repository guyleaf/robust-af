from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from rtdetrv2.core import register


@register()
class RobustCriterion(nn.Module):
    __inject__ = ["model_loss", "cst_loss"]

    def __init__(
        self,
        model_loss: nn.Module,
        cst_loss: Optional[nn.Module] = None,
        weight_dict: dict = dict(loss_cst=20),
    ):
        super().__init__()
        self.weight_dict = weight_dict

        self.model_loss = model_loss
        self.cst_loss = cst_loss

    def loss_cst(
        self,
        outputs: dict,
        targets: list[Union[dict, Sequence[dict]]],
        prefix: str = "loss_cst",
    ):
        if self.cst_loss is None:
            return {}

        robust_hidden_states: list[torch.Tensor] = outputs["robust_hidden_states"]
        clear_robust_hidden_states: list[torch.Tensor] = outputs[
            "clear_robust_hidden_states"
        ]

        global_weight = self.weight_dict.get(prefix)
        if global_weight is not None:
            weights = {
                f"{prefix}_{i}": global_weight for i in range(len(robust_hidden_states))
            }
        else:
            weights = self.weight_dict

        cst_losses = {}
        for i, (robust_hidden_state, clear_robust_hidden_state) in enumerate(
            zip(robust_hidden_states, clear_robust_hidden_states)
        ):
            cst_loss: torch.Tensor = self.cst_loss(
                robust_hidden_state, clear_robust_hidden_state
            )
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
        # if isinstance(targets[0], Sequence):
        #     # currently, we don't need clear targets
        #     targets = [item[0] for item in targets]

        losses = self.model_loss(outputs, targets, **kwargs)

        cst_losses = self.loss_cst(outputs, targets)
        losses.update(cst_losses)

        return losses


@register()
class RobustCriterionv2(nn.Module):
    __inject__ = ["model_loss", "content_loss", "style_loss"]

    def __init__(
        self,
        model_loss: nn.Module,
        content_loss: nn.Module,
        style_loss: nn.Module,
        weight_dict: dict = dict(loss_content=20, loss_style=20),
    ):
        super().__init__()
        self.weight_dict = weight_dict

        self.model_loss = model_loss
        self.content_loss = content_loss
        self.style_loss = style_loss

    def loss_content(
        self,
        rhs: list[torch.Tensor],
        clear_rhs: list[torch.Tensor],
        prefix: str = "loss_content",
    ):
        global_weight = self.weight_dict.get(prefix)
        if global_weight is not None:
            weights = {f"{prefix}_{i}": global_weight for i in range(len(rhs))}
        else:
            weights = self.weight_dict

        losses = {}
        for i, (feats, clear_feats) in enumerate(zip(rhs, clear_rhs)):
            loss: torch.Tensor = self.content_loss(feats, clear_feats)
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

        rhs = outputs["robust_hidden_states"]
        clear_rhs = outputs["clear_robust_hidden_states"]

        content_losses = self.loss_content(rhs, clear_rhs)
        losses.update(content_losses)

        style_losses = self.loss_style(rhs, clear_rhs)
        losses.update(style_losses)
        return losses
