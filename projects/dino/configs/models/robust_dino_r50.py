import copy

import torch.nn as nn
from detectron2.config import LazyCall as L
from omegaconf import OmegaConf

from projects.dino.modeling import (
    RobustDINO,
    RobustDINOTransformer,
    RobustDINOTransformerEncoder,
)
from robust_af.detrex.modeling.criterion import RobustCriterion
from robust_af.models.feature_adapters import AMFG

from .dino_r50 import model as dino_model

model = L(RobustDINO)(
    train_decoder=False,
    train_query_selection=False,
    train_object_queries=False,
    train_level_embed=False,
    train_cdn=False,
    train_heads=True,
    transformer=L(RobustDINOTransformer)(
        encoder=L(RobustDINOTransformerEncoder)(
            robust_layer=L(AMFG)(
                embed_dims="${..embed_dim}",
                spatial_attention=1,
            ),
            num_robust_layers=1,
            start_robust_gap_index=0,
            share_robust_layer=False,
        ),
    ),
)

model = OmegaConf.merge(dino_model, model)

model.criterion.num_classes = "${...num_classes}"
# wrap the criterion
model.criterion = L(RobustCriterion)(
    criterion=model.criterion,
    loss_cst=L(nn.MSELoss)(reduction="none"),
    start_suffix_index="${..transformer.encoder.start_robust_gap_index}",
    weight_dict={
        "loss_cst": 20.0,
    },
)

# set loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
weight_dict = {}
for i in range(
    model.transformer.encoder.start_robust_gap_index,
    model.transformer.encoder.start_robust_gap_index
    + model.transformer.encoder.num_robust_layers,
):
    for j in range(model.transformer.num_feature_levels):
        weight_dict.update({k + f"_{i}_{j}": v for k, v in base_weight_dict.items()})
model.criterion.weight_dict = weight_dict
