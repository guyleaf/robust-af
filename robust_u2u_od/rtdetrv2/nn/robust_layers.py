from rtdetrv2.core import register

from ...models import robust_layers

AMFG = register()(robust_layers.AMFG)
FrequencyAMFG = register()(robust_layers.FrequencyAMFG)
SpatialAMFG = register()(robust_layers.SpatialAMFG)
