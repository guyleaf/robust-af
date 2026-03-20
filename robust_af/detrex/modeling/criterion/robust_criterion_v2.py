from typing import Optional

import torch
import torch.nn as nn


class RobustCriterionv2(nn.Module):
    """This class computes the robust loss with any model criterion."""

    def __init__(
        self,
        criterion: nn.Module,
        weight_dict: dict,
        loss_image_cst: Optional[nn.Module] = None,
        loss_cst: Optional[nn.Module] = None,
    ):
        """Create the criterion.

        Parameters:
            criterion: model criterion.
            loss_cst: loss for calculating consistency error. Must set reduction=none.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            start_suffix_index: Start index of losses for robust hidden states. Defaults to 0.
        """
        super().__init__()
        if hasattr(criterion, "weight_dict"):
            weight_dict.update(criterion.weight_dict)

        self.criterion = criterion
        self.loss_image_cst = loss_image_cst
        self.loss_cst = loss_cst
        self.weight_dict = weight_dict

    def compute_image_cst_loss(
        self,
        rhss: torch.Tensor,
        clear_rhss: torch.Tensor,
        prefix: str = "loss_image_cst",
    ):
        loss: torch.Tensor = self.loss_image_cst(rhss, clear_rhss)
        # mean over the feature dims & mean over batch_size
        indices = list(range(1, loss.ndim))
        loss = loss.mean(indices).mean()
        return {prefix: loss}

    def compute_cst_loss(
        self,
        rhss: dict[str, torch.Tensor],
        clear_rhss: dict[str, torch.Tensor],
        prefix: str = "loss_cst",
    ):
        losses = {}
        for k in rhss:
            loss: torch.Tensor = self.loss_cst(rhss[k], clear_rhss[k])
            # mean over the feature dims & mean over batch_size
            indices = list(range(1, loss.ndim))
            loss = loss.mean(indices).mean()
            losses[f"{prefix}_{k}"] = loss
        return losses

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = self.criterion(outputs, targets, dn_metas=dn_metas)

        if self.loss_image_cst is not None:
            rhss: torch.Tensor = outputs["image_rhss"]
            clear_rhss: torch.Tensor = outputs["clear_image_rhss"]
            cst_losses = self.compute_image_cst_loss(rhss, clear_rhss)
            losses.update(cst_losses)

        if self.loss_cst is not None:
            rhss: dict[str, torch.Tensor] = outputs["rhss"]
            clear_rhss: dict[str, torch.Tensor] = outputs["clear_rhss"]
            cst_losses = self.compute_cst_loss(rhss, clear_rhss)
            losses.update(cst_losses)
        return losses
