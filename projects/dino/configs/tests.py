# ruff: noqa: F401
from detectron2.config import LazyCall as L
from omegaconf import OmegaConf

from robust_af.detrex.data.datasets import register_dds as dds
from robust_af.detrex.data.datasets import register_dut_anti_uav as dut_anti_uav
from robust_af.detrex.data.datasets import (
    register_robust_anti_uav_low as robust_anti_uav_low,
)
from robust_af.detrex.data.datasets import register_uav_eagle as uav_eagle
from robust_af.detrex.modeling.processors import MultiScaleProcessor  # noqa: F401
from robust_af.models.feature_adapters import SimpleNN, SpatialAFR  # noqa: F401
from robust_af.models.image_adapters.baselines import DIP, GDIP, DENet

config = OmegaConf.create()
config.checkpoint = "/home/leafying/data/checkpoints/detrex/dino_swin_small_224_4scale/dut_anti_uav/robust_dino_swin_small_224_v2_4scale_36ep_1e-5_lr_simple_nn_no_train_heads_from_24ep/model_best_0018849.pth"
config.name = "robust_dino_swin_small_224_v2_4scale_36ep_1e-5_lr_simple_nn_no_train_heads_from_24ep"

# for each run, it runs on one of gpus
config.devices = [0, 1, 2, 3]
# config.devices = [0, 1]
# config.devices = [2, 3]

# config
train_dataset_name = "dut_anti_uav"
# train_dataset_name = "dut_anti_uav_fog"

# config.degradations = None
# config.degradations = ["fog"]
config.degradations = [
    "snow",
    "fog",
    "rain",
    "gaussian_noise",
    "iso_noise",
    "multiplicative_noise",
    "resampling_blur",
    "motion_blur",
    "zoom_blur",
    "color_jitter",
    "compression",
    "elastic",
    "glass_blur",
    "brightness",
    "contrast",
]
# split current subset by degradations instead of using new subset
config.split_subset = True

# config.config = "dino_r50_4scale.py"
# config.config = "dino_swin_small_224_4scale.py"
# config.config = "robust_dino_r50_v2_4scale.py"
config.config = "robust_dino_swin_small_224_v2_4scale.py"

config.datasets = [
    dict(
        name=dut_anti_uav.DATASET_NAME,
        # subset keys are customizable
        subsets=dict(
            # name: check ANN_FILES in register_xxx.py
            # robust: use robust_dataloader. defaults to False.
            # val=dict(name=f"{dut_anti_uav.DATASET_NAME}_val"),
            # val_degraded=dict(
            #     name=f"{dut_anti_uav.DATASET_NAME}_val_degraded", robust=True
            # ),
            # test=dict(name=f"{dut_anti_uav.DATASET_NAME}_test"),
            test_degraded=dict(
                name=f"{dut_anti_uav.DATASET_NAME}_test_degraded", robust=True
            ),
        ),
    ),
    # dict(
    #     name=uav_eagle.DATASET_NAME,
    #     subsets=dict(
    #         test=dict(name=f"{uav_eagle.DATASET_NAME}_test"),
    #         test_degraded=dict(
    #             name=f"{uav_eagle.DATASET_NAME}_test_degraded", robust=True
    #         ),
    #     ),
    # ),
    # dict(
    #     name=robust_anti_uav_low.DATASET_NAME,
    #     subsets=dict(
    #         val=dict(name=f"{robust_anti_uav_low.DATASET_NAME}_val"),
    #         val_degraded=dict(
    #             name=f"{robust_anti_uav_low.DATASET_NAME}_val_degraded", robust=True
    #         ),
    #     ),
    # ),
    # dict(
    #     name=dds.DATASET_NAME,
    #     subsets=dict(
    #         test=dict(name=f"{dds.DATASET_NAME}_test"),
    #         test_degraded=dict(name=f"{dds.DATASET_NAME}_test_degraded", robust=True),
    #     ),
    # ),
]
config.config = f"configs/{train_dataset_name}/test/{config.config}"

#############################################################################

# custom configs below will be merged into the config
# NOTE: Be careful! Make sure setting full config. Otherwise, you may encounter any side effects from test configs.
model = OmegaConf.create()

if "robust" in config.config:
    model.robust_module = None
    model.robust_image_module = None

########### DINO R50 ###########
# (DINO R50 only) use the original implementation of dab-detr position embedding if training epochs > 12.
# model.position_embedding = dict(temperature=20, offset=0.0)

# model.robust_image_module = L(DENet)(compat_mode=False)
# model.robust_image_module = L(GDIP)(multi_level=True)
# model.robust_image_module = L(DIP)()

# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SimpleNN)(embed_dims=512, activation="LeakyReLU"),
#     res4=L(SimpleNN)(embed_dims=1024, activation="LeakyReLU"),
#     res5=L(SimpleNN)(embed_dims=2048, activation="LeakyReLU"),
# )
# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(SpatialAFR)(embed_dims=512, activation="ReLU"),
#     res4=L(SpatialAFR)(embed_dims=1024, activation="ReLU"),
#     res5=L(SpatialAFR)(embed_dims=2048, activation="ReLU"),
# )

########### DINO Swin ###########
model.robust_module = L(MultiScaleProcessor)(
    p1=L(SimpleNN)(embed_dims=192, activation="LeakyReLU"),
    p2=L(SimpleNN)(embed_dims=384, activation="LeakyReLU"),
    p3=L(SimpleNN)(embed_dims=768, activation="LeakyReLU"),
)
# model.robust_module = L(MultiScaleProcessor)(
#     p1=L(SpatialAFR)(embed_dims=192, activation="ReLU"),
#     p2=L(SpatialAFR)(embed_dims=384, activation="ReLU"),
#     p3=L(SpatialAFR)(embed_dims=768, activation="ReLU"),
# )
