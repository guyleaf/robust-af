from typing import Sequence, Union

import torch
import torch.nn as nn
from rtdetrv2.core import register


@register()
class RobustCriterion(nn.Module):
    __inject__ = ["model_loss", "cst_loss"]

    def __init__(
        self,
        model_loss: nn.Module,
        cst_loss: nn.Module,
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
