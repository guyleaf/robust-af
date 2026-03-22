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


from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures.instances import Instances
from detectron2.utils.events import get_event_storage

from robust_af.utils import freeze_all, unfreeze_modules_and_parameters

from .dino import DINO


class RobustDINOv2(DINO):
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
        robust_image_module: Optional[nn.Module] = None,
        robust_module: Optional[nn.Module] = None,
        train_encoder: bool = False,
        train_decoder: bool = False,
        train_query_selection: bool = False,
        train_object_queries: bool = False,
        train_level_embed: bool = False,
        train_cdn: bool = False,
        train_heads: bool = True,
        train_all: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.robust_image_module = robust_image_module
        self.robust_module = robust_module

        self.training_parts: List[Union[nn.Module, nn.Parameter]] = []
        if self.with_robust_image_module:
            self.training_parts += [self.robust_image_module]
        if self.with_robust_module:
            self.training_parts += [self.robust_module]

        if train_encoder:
            self.training_parts += [self.transformer.encoder]
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

        if not train_all:
            self.freeze()

    def freeze(self):
        self = freeze_all(self)
        unfreeze_modules_and_parameters(self.training_parts)
        return self

    @property
    def with_robust_image_module(self):
        return self.robust_image_module is not None

    @property
    def with_robust_module(self):
        return self.robust_module is not None

    def visualize_training(
        self,
        batched_inputs: list[dict],
        results: list[Instances],
        vis_name: str = "Left: GT bounding boxes;  Right: Predicted boxes",
    ):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_box = 20

        for input, results_per_image in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=results_per_image.pred_boxes[:max_vis_box]
                .tensor.detach()
                .cpu()
                .numpy()
            )
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def _extract_feats(
        self, images: torch.Tensor, robust: bool = True, returns_rhs: bool = False
    ):
        # image-level restoration
        if robust and self.with_robust_image_module:
            images = self.robust_image_module(images)
        image_rhss = images

        # original features
        features = self.backbone(images)  # output feature dict

        # feature-level restoration
        if robust and self.with_robust_module:
            features = self.robust_module(features)
        rhss = features

        # project backbone features to the required dimension of transformer
        # we use multi-scale features in DINO
        features = self.neck(features)
        if returns_rhs:
            return features, {"image_rhss": image_rhss, "rhss": rhss}
        else:
            return features

    def _loss(self, batched_inputs: list[dict]):
        images = self.preprocess_image(batched_inputs)
        multi_level_feats, rhs_dict = self._extract_feats(
            images.tensor, returns_rhs=True
        )
        output = self._forward_transformer(
            multi_level_feats, images.tensor.shape[2:], batched_inputs
        )
        output.update(rhs_dict)

        # visualize training samples
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                results = self.inference(box_cls, box_pred, images.image_sizes)
                self.visualize_training(batched_inputs, results, vis_name="Degraded")

        # clear features are only for loss calculation.
        clear_batched_inputs = [inputs["clear"] for inputs in batched_inputs]
        images = self.preprocess_image(clear_batched_inputs)
        with torch.no_grad():
            multi_level_feats, rhs_dict = self._extract_feats(
                images.tensor, robust=False, returns_rhs=True
            )
            for k, v in rhs_dict.items():
                output[f"clear_{k}"] = v

        # visualize training samples
        if self.vis_period > 0:
            with torch.no_grad():
                clear_output = self._forward_transformer(
                    multi_level_feats, images.tensor.shape[2:], clear_batched_inputs
                )
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                box_cls = clear_output["pred_logits"]
                box_pred = clear_output["pred_boxes"]
                results = self.inference(box_cls, box_pred, images.image_sizes)
                self.visualize_training(clear_batched_inputs, results, vis_name="Clear")
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
