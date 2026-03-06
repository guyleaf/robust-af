from typing import Any, Callable, Optional

import torch
import torch.nn as nn


class RobustDetectLoss:
    def __init__(
        self,
        model_loss: Callable[[Any, dict], tuple[torch.Tensor, torch.Tensor]],
        cst_loss: Optional[nn.Module] = None,
        weight: float = 20,
    ):
        self.weight = weight

        self.model_loss = model_loss
        self.cst_loss = cst_loss

    def __call__(self, preds: dict, batch: dict):
        loss, loss_items = self.model_loss(preds["preds"], batch)
        if self.cst_loss is None:
            return loss, loss_items

        batch_size = batch["img"].shape[0]
        robust_hidden_states: list[torch.Tensor] = preds["robust_hidden_states"]
        clear_robust_hidden_states: list[torch.Tensor] = preds[
            "clear_robust_hidden_states"
        ]

        cst_loss_items = loss_items.new_empty((len(robust_hidden_states)))
        for i, (robust_hidden_state, clear_robust_hidden_state) in enumerate(
            zip(robust_hidden_states, clear_robust_hidden_states)
        ):
            cst_loss: torch.Tensor = self.cst_loss(
                robust_hidden_state, clear_robust_hidden_state
            )
            # sum over the feature dims & mean over batch_size
            indices = list(range(1, cst_loss.ndim))
            cst_loss = cst_loss.mean(indices).mean()
            cst_loss_items[i] = cst_loss
        cst_loss_items *= self.weight

        # Follow the ultralytics design, they usually multiply loss with batch_size. (I still don't understand why they do that)
        # related issue: https://github.com/ultralytics/ultralytics/issues/3282#issuecomment-1813494534
        return loss + cst_loss_items.sum() * batch_size, torch.cat(
            (loss_items, cst_loss_items.detach())
        )
