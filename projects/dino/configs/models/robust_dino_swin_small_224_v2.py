import copy

import torch.nn as nn
from detectron2.config import LazyCall as L
from omegaconf import OmegaConf

from projects.dino.modeling import (
    RobustDINOv2,
)
from robust_af.detrex.modeling import MultiScaleProcessor
from robust_af.detrex.modeling.criterion import RobustCriterionv2
from robust_af.models.feature_adapters import AMFG

from .dino_swin_small_224 import model as dino_model

model = L(RobustDINOv2)(
    train_encoder=False,
    train_decoder=False,
    train_query_selection=False,
    train_object_queries=False,
    train_level_embed=False,
    train_cdn=False,
    train_heads=True,
    robust_module=L(MultiScaleProcessor)(
        p1=L(AMFG)(
            embed_dims=192,
            spatial_attention=1,
        ),
        p2=L(AMFG)(
            embed_dims=384,
            spatial_attention=1,
        ),
        p3=L(AMFG)(
            embed_dims=768,
            spatial_attention=1,
        ),
    ),
)

model = OmegaConf.merge(dino_model, model)

model.criterion.num_classes = "${...num_classes}"
# wrap the criterion
model.criterion = L(RobustCriterionv2)(
    criterion=model.criterion,
    loss_cst=L(nn.MSELoss)(reduction="none"),
    weight_dict={
        "loss_cst": 20.0,
    },
)

# set loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
weight_dict = {}
for i in model.neck.in_features:
    assert i in model.robust_module, (
        f"The robust module should have a processing module for {i}."
    )
    weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
model.criterion.weight_dict = weight_dict
