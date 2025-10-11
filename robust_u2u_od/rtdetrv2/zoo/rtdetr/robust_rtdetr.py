from typing import Optional, Union

import torch
import torch.nn as nn
from rtdetrv2.core import register
from rtdetrv2.zoo.rtdetr import RTDETR, RTDETRTransformerv2

from ....utils import freeze_all, unfreeze_modules_and_parameters


@register()
class RobustRTDETR(RTDETR):
    __inject__ = [
        "backbone",
        "robust_module",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        robust_module: nn.Module,
        encoder: nn.Module,
        decoder: RTDETRTransformerv2,
        train_query_selection: bool = False,
        train_cdn: bool = False,
        train_heads: bool = True,
    ):
        super().__init__(backbone, encoder, decoder)
        self.robust_module = robust_module

        self.training_parts: list[Union[nn.Module, nn.Parameter]] = [self.robust_module]
        if train_query_selection:
            self.training_parts += [
                decoder.enc_output,
                decoder.enc_score_head,
                decoder.enc_bbox_head,
            ]
        if train_cdn:
            self.training_parts += [decoder.denoising_class_embed]
        if train_heads:
            self.training_parts += [decoder.dec_score_head, decoder.dec_bbox_head]
        self._freeze()

    def _freeze(self):
        self = freeze_all(self)
        unfreeze_modules_and_parameters(self.training_parts)
        return self

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict]] = None,
    ):
        if self.training:
            assert x.ndim == 5 and x.shape[1] == 2
            # assert (
            #     targets is not None
            #     and isinstance(targets[0], Sequence)
            #     and len(targets[0]) == 2
            # )
            x, clear_x = x.split(1, dim=1)
            x.squeeze_()
            clear_x.squeeze_()
            # # currently, we don't need clear targets
            # targets = [item[0] for item in targets]
        else:
            clear_x = None

        x = self.backbone(x)
        if clear_x is not None:
            with torch.no_grad():
                clear_x = self.backbone(clear_x)

        # restore features
        restored_x = self.robust_module(x)

        x = self.encoder(restored_x)
        x = self.decoder(x, targets)

        if self.training:
            assert isinstance(x, dict)
            x["robust_hidden_states"] = restored_x
            x["clear_robust_hidden_states"] = clear_x
        return x
