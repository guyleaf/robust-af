from rtdetrv2.core import register

from ...models import feature_adapters

# image-level adapters

# feature-level adapters
AMFG = register()(feature_adapters.AMFG)
FrequencyAMFG = register()(feature_adapters.FrequencyAMFG)
SpatialAMFG = register()(feature_adapters.SpatialAMFG)

AMFGv2 = register()(feature_adapters.AMFGv2)
FrequencyAMFGv2 = register()(feature_adapters.FrequencyAMFGv2)
SpatialAMFGv2 = register()(feature_adapters.SpatialAMFGv2)

AFR = register()(feature_adapters.AFR)
SpatialAFR = register()(feature_adapters.SpatialAFR)
FrequencyAFR = register()(feature_adapters.FrequencyAFR)
SpatialAFRDebug = register()(feature_adapters.afr.SpatialAFRDebug)

SimpleNN = register()(feature_adapters.SimpleNN)


# baselines
try:
    from ...models.image_adapters import baselines as image_adapter_baselines
    # from ...models.feature_adapters import baselines as feature_adapter_baselines

    # image-level adapters
    DENet = register()(image_adapter_baselines.DENet)

    # feature-level adapters
except ImportError:
    pass
