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

import copy
from typing import Optional

import torch
import torch.nn as nn
from fairscale.nn.checkpoint import checkpoint_wrapper

from .dino_transformer import DINOTransformer, DINOTransformerEncoder


class RobustDINOTransformerEncoder(DINOTransformerEncoder):
    """Robust transformer encoder for DINO.

    Common robust settings:
        1. post setting (default): start_robust_gap_index = 1, num_robust_layers = num_layers
        2. pre setting: start_robust_gap_index = 0, num_robust_layers = num_layers
        3. full setting: start_robust_gap_index = 0, num_robust_layers = num_layers + 1

    Simple example for what gap index is:
        (gap 0)
        ======= encoder layer 0 =======
        (gap 1)
        ======= encoder layer 1 =======
        (gap 2)
        ======= encoder layer 2 =======
        (gap 3)
        ======= encoder layer 3 =======
        (gap 4)

    Args:
        robust_layer (nn.Module): the robust processing layer. All the layers will use the copies.
        num_robust_layers (int): the number of robust layers. Must be less or equal to the number of encoder layers.
        start_robust_gap_index (int): Start index of encoder layers to apply robust layer.
        share_robust_layer (bool): Share the robust layer among feature levels.
    """

    def __init__(
        self,
        robust_layer: nn.Module,
        *args,
        num_layers: int = 6,
        num_feature_levels: int = 4,
        num_robust_layers: int = 6,
        start_robust_gap_index: int = 0,
        share_robust_layer: bool = True,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        assert start_robust_gap_index <= num_layers + 1
        assert 0 <= num_robust_layers <= num_layers + 1 - start_robust_gap_index
        super().__init__(
            *args,
            num_layers=num_layers,
            num_feature_levels=num_feature_levels,
            **kwargs,
        )

        self.num_robust_layers = num_robust_layers
        self.start_robust_gap_index = start_robust_gap_index

        # initialize robust layers
        self.robust_layers = nn.ModuleList()
        for _ in range(self.num_robust_layers):
            layer = copy.deepcopy(robust_layer)
            if use_checkpoint:
                layer = checkpoint_wrapper(layer)

            if share_robust_layer:
                robust_feat_layers = nn.ModuleList([layer] * num_feature_levels)
            else:
                robust_feat_layers = nn.ModuleList(
                    copy.deepcopy(layer) for _ in range(num_feature_levels)
                )

            self.robust_layers.append(robust_feat_layers)

    def _generate_robust_features(
        self,
        layers: nn.ModuleList,
        feats: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        robust: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # [B, L, C] -> [levels, B, L, C]
        # level_start_index[0] is always 0. tensor_split doesn't need it.
        mlvl_feats = feats.tensor_split(level_start_index[1:].cpu(), dim=1)
        if not robust:
            return feats, mlvl_feats

        new_mlvl_feats = []
        for i, (feats, (h, w)) in enumerate(zip(mlvl_feats, spatial_shapes)):
            batch_size, _, feat_dims = feats.shape
            # [B, L, C] -> [B, C, H, W]
            feats = feats.transpose(1, 2).view(batch_size, feat_dims, h, w)
            # TODO: remove padding and process individually?
            feats = layers[i](feats)
            # [B, C, H, W] -> [B, L, C]
            feats = feats.view(batch_size, feat_dims, -1).transpose(1, 2)
            new_mlvl_feats.append(feats)

        # return new_mlvl_feats to calculate robust loss
        return torch.cat(new_mlvl_feats, dim=1), new_mlvl_feats

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[torch.Tensor] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        robust: bool = True,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        robust_hidden_states = []

        # before 1st encoder layer
        if self.start_robust_gap_index == 0:
            query, mlvl_feats = self._generate_robust_features(
                self.robust_layers[0],
                query,
                spatial_shapes,
                level_start_index,
                robust=robust,
            )
            robust_hidden_states.append(mlvl_feats)

        for i, layer in enumerate(self.layers, start=1):
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs,
            )

            robust_gap_index = i - self.start_robust_gap_index
            if 0 <= robust_gap_index < self.num_robust_layers:
                query, mlvl_feats = self._generate_robust_features(
                    self.robust_layers[robust_gap_index],
                    query,
                    spatial_shapes,
                    level_start_index,
                    robust=robust,
                )
                robust_hidden_states.append(mlvl_feats)

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query, robust_hidden_states


class RobustDINOTransformer(DINOTransformer):
    """Robust transformer module for DINO

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embeds,
        attn_masks,
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in multi_level_masks], 1
        )

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory, robust_hidden_states = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
            output_memory
        )
        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
            + output_proposals
        )  # unsigmoided.

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embeds[1] is not None:
            reference_points = torch.cat(
                [query_embeds[1].sigmoid(), reference_points], 1
            )
        init_reference_out = reference_points

        # extract region features
        target_unact = torch.gather(
            output_memory,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]),
        )
        if self.learnt_init_query:
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if query_embeds[0] is not None:
            target = torch.cat([query_embeds[0], target], 1)

        # decoder
        inter_states, inter_references = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            attn_masks=attn_masks,
            **kwargs,
        )

        inter_references_out = inter_references
        return (
            inter_states,
            init_reference_out,
            inter_references_out,
            target_unact,
            topk_coords_unact.sigmoid(),
            robust_hidden_states,
        )
