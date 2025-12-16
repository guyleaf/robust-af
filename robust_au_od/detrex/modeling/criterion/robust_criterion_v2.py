import torch
import torch.nn as nn


class RobustCriterionv2(nn.Module):
    """This class computes the robust loss with any model criterion."""

    def __init__(
        self,
        criterion: nn.Module,
        loss_cst: nn.Module,
        weight_dict: dict,
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
        self.loss_cst = loss_cst
        self.weight_dict = weight_dict

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = self.criterion(outputs, targets, dn_metas=dn_metas)

        # Compute all the requested losses

        cst_losses = self.compute_cst_loss(outputs)
        losses.update(cst_losses)

        return losses

    def compute_cst_loss(self, outputs: dict):
        robust_hidden_states: dict[str, torch.Tensor] = outputs["robust_hidden_states"]
        clear_robust_hidden_states: dict[str, torch.Tensor] = outputs[
            "clear_robust_hidden_states"
        ]

        losses = {}
        for k in robust_hidden_states:
            loss: torch.Tensor = self.loss_cst(
                robust_hidden_states[k], clear_robust_hidden_states[k]
            )
            # mean over the feature dims & mean over batch_size
            indices = list(range(1, loss.ndim))
            loss = loss.mean(indices).mean()
            losses[f"loss_cst_{k}"] = loss
        return losses


class RobustCriterionv2Debug(nn.Module):
    def __init__(
        self,
        criterion: nn.Module,
        loss_content: nn.Module,
        loss_style: nn.Module,
        weight_dict: dict,
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
        self.loss_content = loss_content
        self.loss_style = loss_style
        self.weight_dict = weight_dict

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = self.criterion(outputs, targets, dn_metas=dn_metas)

        # Compute all the requested losses
        rhs = outputs["robust_hidden_states"]
        clear_rhs = outputs["clear_robust_hidden_states"]

        content_losses = self.compute_content_loss(rhs, clear_rhs)
        losses.update(content_losses)

        style_losses = self.compute_style_loss(rhs, clear_rhs)
        losses.update(style_losses)
        return losses

    def compute_content_loss(
        self, rhs: dict[str, torch.Tensor], clear_rhs: dict[str, torch.Tensor]
    ):
        losses = {}
        # layer by layer
        for k in rhs:
            loss: torch.Tensor = self.loss_content(rhs[k], clear_rhs[k])
            # mean over the feature dims & mean over batch_size
            indices = list(range(1, loss.ndim))
            loss = loss.mean(indices).mean()
            losses[f"loss_content_{k}"] = loss
        return losses

    def compute_style_loss(
        self, rhs: dict[str, torch.Tensor], clear_rhs: dict[str, torch.Tensor]
    ):
        losses = {}
        # layer by layer
        for k in rhs:
            feats = rhs[k]
            clear_feats = clear_rhs[k]

            # calculate statistics of feature maps for each channel
            indices = list(range(2, feats.ndim))
            # [B, C]
            stds, means = torch.std_mean(feats, dim=indices)
            clear_stds, clear_means = torch.std_mean(clear_feats, dim=indices)

            # calculate style loss
            # [B, C]
            loss_means: torch.Tensor = self.loss_style(means, clear_means)
            loss_stds: torch.Tensor = self.loss_style(stds, clear_stds)
            loss = loss_means + loss_stds

            # mean over the channels & mean over batch_size
            loss = loss.mean(dim=1).mean()
            losses[f"loss_style_{k}"] = loss
        return losses
