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
        "encoder",
        "decoder",
        "robust_image_module",
        "robust_module",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: RTDETRTransformerv2,
        robust_image_module: Optional[nn.Module] = None,
        robust_module: Optional[nn.Module] = None,
        train_query_selection: bool = False,
        train_cdn: bool = False,
        train_heads: bool = True,
    ):
        super().__init__(backbone, encoder, decoder)
        self.robust_image_module = robust_image_module
        self.robust_module = robust_module

        self.training_parts: list[Union[nn.Module, nn.Parameter]] = []
        if self.with_robust_image_module:
            self.training_parts += [self.robust_image_module]
        if self.with_robust_module:
            self.training_parts += [self.robust_module]

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

    @property
    def with_robust_image_module(self):
        return self.robust_image_module is not None

    @property
    def with_robust_module(self):
        return self.robust_module is not None

    def _forward_inference(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict]] = None,
    ):
        # image-level restoration
        if self.with_robust_image_module:
            x = self.robust_image_module(x)

        x = self.backbone(x)

        # feature-level restoration
        if self.with_robust_module:
            x = self.robust_module(x)

        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def _forward_train(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict]] = None,
    ):
        assert x.ndim == 5 and x.shape[1] == 2
        x, clear_x = x.split(1, dim=1)
        x.squeeze_()
        clear_x.squeeze_()

        # rhs => robust_hidden_states
        with torch.no_grad():
            rhs_dict = {
                "clear_image_rhss": clear_x,
                "clear_rhss": self.backbone(clear_x),
            }

        # image-level restoration
        if self.with_robust_image_module:
            x = rhs_dict["image_rhss"] = self.robust_image_module(x)

        x = self.backbone(x)

        # feature-level restoration
        if self.with_robust_module:
            x = rhs_dict["rhss"] = self.robust_module(x)

        x = self.encoder(x)
        x = self.decoder(x, targets)

        assert isinstance(x, dict)
        x.update(rhs_dict)
        return x

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict]] = None,
    ):
        if self.training:
            return self._forward_train(x, targets=targets)
        else:
            return self._forward_inference(x, targets=targets)
