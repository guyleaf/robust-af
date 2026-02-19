import copy

import torch.nn as nn
from detectron2.config import LazyCall as L
from omegaconf import OmegaConf

from projects.dino.modeling import (
    RobustDINOv2,
)
from robust_af.detrex.modeling import MultiScaleProcessor
from robust_af.detrex.modeling.criterion.robust_criterion_v2 import (
    RobustCriterionv2Debug,
)
from robust_af.models.robust_layers import AMFG

from .dino_r50 import model as dino_model

model = L(RobustDINOv2)(
    train_encoder=False,
    train_decoder=False,
    train_query_selection=False,
    train_object_queries=False,
    train_level_embed=False,
    train_cdn=False,
    train_heads=True,
    robust_module=L(MultiScaleProcessor)(
        res3=L(AMFG)(
            embed_dims=512,
            spatial_attention=1,
        ),
        res4=L(AMFG)(
            embed_dims=1024,
            spatial_attention=1,
        ),
        res5=L(AMFG)(
            embed_dims=2048,
            spatial_attention=1,
        ),
    ),
)

model = OmegaConf.merge(dino_model, model)

model.criterion.num_classes = "${...num_classes}"
# wrap the criterion
model.criterion = L(RobustCriterionv2Debug)(
    criterion=model.criterion,
    loss_content=L(nn.MSELoss)(reduction="none"),
    loss_style=L(nn.MSELoss)(reduction="none"),
    weight_dict={
        "loss_content": 20.0,
        "loss_style": 20.0,
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
