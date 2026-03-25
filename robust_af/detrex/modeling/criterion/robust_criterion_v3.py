from typing import Optional

import torch
import torch.nn as nn


class RobustCriterionv3(nn.Module):
    def __init__(
        self,
        criterion: nn.Module,
        loss_style: nn.Module,
        weight_dict: dict,
        loss_image_content: Optional[nn.Module] = None,
        loss_content: Optional[nn.Module] = None,
    ):
        """Create the criterion.

        Parameters:
            criterion: model criterion.
            loss_content: loss for calculating consistency error for content. Must set reduction=none.
            loss_style: loss for calculating consistency error for style. Must set reduction=none.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            start_suffix_index: Start index of losses for robust hidden states. Defaults to 0.
        """
        super().__init__()
        if hasattr(criterion, "weight_dict"):
            weight_dict.update(criterion.weight_dict)

        self.criterion = criterion
        self.loss_image_content = loss_image_content
        self.loss_content = loss_content
        self.loss_style = loss_style
        self.weight_dict = weight_dict

    def compute_image_content_loss(
        self,
        rhss: torch.Tensor,
        clear_rhss: torch.Tensor,
        prefix: str = "loss_image_content",
    ):
        loss: torch.Tensor = self.loss_image_content(rhss, clear_rhss)
        # mean over the feature dims & mean over batch_size
        indices = list(range(1, loss.ndim))
        loss = loss.mean(indices).mean()
        return {prefix: loss}

    def compute_content_loss(
        self,
        rhs: dict[str, torch.Tensor],
        clear_rhs: dict[str, torch.Tensor],
        prefix: str = "loss_content",
    ):
        losses = {}
        # layer by layer
        for k in rhs:
            loss: torch.Tensor = self.loss_content(rhs[k], clear_rhs[k])
            # mean over the feature dims & mean over batch_size
            indices = list(range(1, loss.ndim))
            loss = loss.mean(indices).mean()
            losses[f"{prefix}_{k}"] = loss
        return losses

    def compute_style_loss(
        self,
        rhs: dict[str, torch.Tensor],
        clear_rhs: dict[str, torch.Tensor],
        prefix: str = "loss_style",
    ):
        losses = {}
        # layer by layer
        for k in rhs:
            feats = rhs[k]
            clear_feats = clear_rhs[k]

            # calculate statistics of feature maps for each channel
            indices = list(range(2, feats.ndim))
            # [B, C]
            stds, means = torch.std_mean(feats, dim=indices, correction=0)
            clear_stds, clear_means = torch.std_mean(
                clear_feats, dim=indices, correction=0
            )

            # calculate style loss
            # [B, C]
            loss_means: torch.Tensor = self.loss_style(means, clear_means)
            loss_stds: torch.Tensor = self.loss_style(stds, clear_stds)
            loss = loss_means + loss_stds

            # mean over the channels & mean over batch_size
            loss = loss.mean(dim=1).mean()
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

        if self.loss_image_content is not None:
            rhss = outputs["image_rhss"]
            clear_rhss = outputs["clear_image_rhss"]
            content_losses = self.compute_image_content_loss(rhss, clear_rhss)
            losses.update(content_losses)

        if self.loss_content is not None:
            rhss = outputs["rhss"]
            clear_rhss = outputs["clear_rhss"]
            content_losses = self.compute_content_loss(rhss, clear_rhss)
            losses.update(content_losses)

            style_losses = self.compute_style_loss(rhss, clear_rhss)
            losses.update(style_losses)
        return losses
