import copy

import torch.nn as nn
from detectron2.config import LazyCall as L
from omegaconf import OmegaConf

from projects.dino.modeling import RobustDINOv3
from robust_au_od.detrex.modeling import build_module_dict
from robust_au_od.detrex.modeling.backbone import RobustResNet
from robust_au_od.detrex.modeling.criterion.robust_criterion_v2 import RobustCriterionv2
from robust_au_od.models.robust_layers import AMFG

from .dino_r50 import model as dino_model

model = L(RobustDINOv3)(
    train_encoder=False,
    train_decoder=False,
    train_query_selection=False,
    train_object_queries=False,
    train_level_embed=False,
    train_cdn=False,
    train_heads=True,
    backbone=L(RobustResNet)(
        robust_module=L(build_module_dict)(
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
        )
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
    assert i in model.backbone.robust_module, (
        f"The robust module should have a processing module for {i}."
    )
    weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
model.criterion.weight_dict = weight_dict
