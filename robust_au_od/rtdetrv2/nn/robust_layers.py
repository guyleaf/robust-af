from rtdetrv2.core import register

from ...models import robust_layers
from ...models.robust_layers.afr import SpatialAFRDebug

AMFG = register()(robust_layers.AMFG)
FrequencyAMFG = register()(robust_layers.FrequencyAMFG)
SpatialAMFG = register()(robust_layers.SpatialAMFG)

AMFGv2 = register()(robust_layers.AMFGv2)
FrequencyAMFGv2 = register()(robust_layers.FrequencyAMFGv2)
SpatialAMFGv2 = register()(robust_layers.SpatialAMFGv2)

AFR = register()(robust_layers.AFR)
SpatialAFR = register()(robust_layers.SpatialAFR)
FrequencyAFR = register()(robust_layers.FrequencyAFR)
SpatialAFRDebug = register()(SpatialAFRDebug)

SimpleNN = register()(robust_layers.SimpleNN)
