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


from typing import List, Union

import torch
import torch.nn as nn
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage

from robust_u2u_od.detrex.utils import freeze_all, unfreeze_modules_and_parameters

from .dino import DINO


class RobustDINO(DINO):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        *args,
        train_decoder: bool = False,
        train_query_selection: bool = False,
        train_object_queries: bool = False,
        train_level_embed: bool = False,
        train_cdn: bool = False,
        train_heads: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.training_parts: List[Union[nn.Module, nn.Parameter]] = [
            self.transformer.encoder.robust_layers,
        ]
        if train_decoder:
            self.training_parts += [self.transformer.decoder]
        if train_query_selection:
            self.training_parts += [
                self.transformer.enc_output,
                self.transformer.enc_output_norm,
            ]
        if train_object_queries:
            self.training_parts += [self.transformer.tgt_embed]
        if train_level_embed:
            self.training_parts += [self.transformer.level_embeds]
        if train_cdn:
            self.training_parts += [self.label_enc]
        if train_heads:
            self.training_parts += [self.class_embed, self.bbox_embed]
        self.freeze()

    def freeze(self):
        self = freeze_all(self)
        unfreeze_modules_and_parameters(self.training_parts)
        return self

    def preprocess_image(self, batched_inputs: list[dict], key: str = "image"):
        images = [self.normalizer(x[key].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def _forward_transformer(
        self,
        multi_level_feats: tuple[torch.Tensor, ...],
        batched_image_shape: tuple[int, int],
        batched_inputs: list[dict],
        robust: bool = True,
    ):
        transformer_inputs_dict, transformer_outputs_dict = self._pre_transformer(
            multi_level_feats, batched_image_shape, batched_inputs
        )

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
            robust_hidden_states,
        ) = self.transformer(**transformer_inputs_dict, robust=robust)
        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        output = self._post_transformer(
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,
            dn_meta=transformer_inputs_dict["dn_meta"],
        )
        transformer_outputs_dict.update(output)
        transformer_outputs_dict["robust_hidden_states"] = robust_hidden_states
        return transformer_outputs_dict

    def _loss(self, batched_inputs: list[dict]):
        images = self.preprocess_image(batched_inputs)
        multi_level_feats = self._extract_feats(images.tensor)
        output = self._forward_transformer(
            multi_level_feats, images.tensor.shape[2:], batched_inputs
        )

        # visualize training samples
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                results = self.inference(box_cls, box_pred, images.image_sizes)
                self.visualize_training(batched_inputs, results)

        # clear features are only for loss calculation.
        images = self.preprocess_image(batched_inputs, key="clear_image")
        with torch.no_grad():
            multi_level_feats = self._extract_feats(images.tensor)
            clear_output = self._forward_transformer(
                multi_level_feats, images.tensor.shape[2:], batched_inputs, robust=False
            )
            output["clear_robust_hidden_states"] = clear_output["robust_hidden_states"]
            del clear_output

        # compute loss
        targets = output.pop("targets")
        dn_meta = output.pop("dn_meta")
        loss_dict = self.criterion(output, targets, dn_meta)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict
