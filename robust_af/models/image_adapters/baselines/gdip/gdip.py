import torch
import torch.nn as nn

from .channel_projector import ChannelProjector
from .gdip_ops import GatedDIP
from .vision_encoder import VisionEncoder


class GDIP(nn.Module):
    def __init__(
        self,
        multi_level: bool = False,
        reversed_order: bool = False,
        embed_dims: int = 256,
        encoder_cfg: dict = {},
        **kwargs,
    ):
        super().__init__()
        self._reversed_order = reversed_order
        num_layers = encoder_cfg.get("num_layers", 5)
        if multi_level:
            out_features = [f"conv_{i}" for i in range(num_layers)]
        else:
            out_features = [f"conv_{num_layers - 1}"]
        self._out_features = out_features

        # in official implementation, encoder + projector == VisionEncoder
        self.encoder = VisionEncoder(out_features=self._out_features, **encoder_cfg)
        self.projector = ChannelProjector(
            self.encoder.layer_channels, embed_dims=embed_dims
        )

        # build single/multi-level GDIP
        gdips = nn.ModuleDict()
        for layer_name in self.encoder.layer_channels:
            gdips[layer_name] = GatedDIP(embed_dims=embed_dims, **kwargs)
        self.gdips = gdips

    def forward(self, images: torch.Tensor):
        # encoder + projector => output_dict['linear_proj_xx'] in official implementation
        latents = self.encoder(images)
        latents = self.projector(latents)

        # top-down
        out_features = self._out_features
        if self._reversed_order:
            # bottom-up
            out_features = reversed(out_features)

        for name in out_features:
            images = self.gdips[name](images, latents[name])
        return images
