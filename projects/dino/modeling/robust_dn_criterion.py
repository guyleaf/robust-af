# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

from .dn_criterion import DINOCriterion


class RobustDINOCriterion(DINOCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self, *args, loss_consistency: nn.Module, start_robust_gap_index: int, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_consistency = loss_consistency
        self.start_robust_gap_index = start_robust_gap_index

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = super().forward(outputs, targets, dn_metas=dn_metas)

        # Compute all the requested losses

        consistency_losses = self.compute_consistency_loss(outputs)
        losses.update(consistency_losses)

        return losses

    def compute_consistency_loss(self, outputs: dict):
        robust_hidden_states: list[torch.Tensor] = outputs["robust_hidden_states"]
        clear_robust_hidden_states: list[torch.Tensor] = outputs[
            "clear_robust_hidden_states"
        ]

        losses = {}
        for i, (robust_hidden_state, clear_robust_hidden_state) in enumerate(
            zip(robust_hidden_states, clear_robust_hidden_states),
            start=self.start_robust_gap_index,
        ):
            loss = self.loss_consistency(robust_hidden_state, clear_robust_hidden_state)
            losses[f"loss_consistency_{i}"] = loss
        return losses
