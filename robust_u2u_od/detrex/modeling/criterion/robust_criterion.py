import torch
import torch.nn as nn


class RobustCriterion(nn.Module):
    """This class computes the robust loss with any model criterion."""

    def __init__(
        self,
        criterion: nn.Module,
        loss_cst: nn.Module,
        weight_dict: dict,
        start_suffix_index: int = 0,
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
        self.start_suffix_index = start_suffix_index

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
        robust_hidden_states: list[torch.Tensor] = outputs["robust_hidden_states"]
        clear_robust_hidden_states: list[torch.Tensor] = outputs[
            "clear_robust_hidden_states"
        ]

        losses = {}
        for i, (mlvl_robust_hidden_state, clear_mlvl_robust_hidden_state) in enumerate(
            zip(robust_hidden_states, clear_robust_hidden_states),
            start=self.start_suffix_index,
        ):
            for j, (robust_hidden_state, clear_robust_hidden_state) in enumerate(
                zip(mlvl_robust_hidden_state, clear_mlvl_robust_hidden_state)
            ):
                loss: torch.Tensor = self.loss_cst(
                    robust_hidden_state, clear_robust_hidden_state
                )
                # mean over the feature dims & mean over batch_size
                indices = list(range(1, loss.ndim))
                loss = loss.mean(indices).mean()
                losses[f"loss_cst_{i}_{j}"] = loss
        return losses
