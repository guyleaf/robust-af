from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog

from robust_au_od.detrex.configs import get_config
from robust_au_od.detrex.data.datasets.register_dut_anti_uav import DATASET_NAME
from robust_au_od.detrex.modeling.processors import MultiScaleProcessor
from robust_au_od.models.robust_layers.afr import SpatialAFRDebug
from robust_au_od.models.robust_layers import SimpleNN

# from robust_au_od.models.robust_layers.afr import SpatialAFRNonParameteric
from ...models.robust_dino_r50_v2 import model

degraded = True
test_dataset_name = DATASET_NAME

dataset = get_config(f"datasets/{test_dataset_name}_detr.py")
if degraded:
    dataloader = dataset.robust_dataloader
else:
    dataloader = dataset.dataloader

train = get_config("train.py").train

metadata = MetadataCatalog.get(test_dataset_name)

suffix = "_degraded" if degraded else ""
output_dir = f"./outputs/dino_r50_4scale/{DATASET_NAME}/{test_dataset_name}/robust_dino_r50_v2_4scale_12ep_1e-5_lr_simple_nn_no_train_heads_from_36ep_debug{suffix}_2"

# ==============================================================

# model.robust_module = L(MultiScaleProcessor)(
#     res3=L(AMFGv2)(embed_dims=512, spatial_attention=1),
#     res4=L(AMFGv2)(embed_dims=1024, spatial_attention=1),
#     res5=L(AMFGv2)(embed_dims=2048, spatial_attention=1),
# )
model.robust_module = L(MultiScaleProcessor)(
    res3=L(SimpleNN)(embed_dims=512),
    res4=L(SimpleNN)(embed_dims=1024),
    res5=L(SimpleNN)(embed_dims=2048),
)

# modify model config
# use the original implementation of dab-detr position embedding.
model.position_embedding.temperature = 20
model.position_embedding.offset = 0.0

# change the number of classes
model.num_classes = metadata.num_classes

# set output dir
train.output_dir = output_dir

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.test.num_workers = 4
if degraded:
    # (robust_dataloader only) disable skip connection in degradations
    dataloader.test.mapper.identity = False

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

# set the random seed
train.seed = 2025
