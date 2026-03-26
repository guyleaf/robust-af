import torch
import torch.nn as nn


class ChannelProjector(nn.Module):
    def __init__(self, input_shapes: dict[str, int], embed_dims: int = 256):
        super().__init__()

        self.projectors = nn.ModuleDict()
        for name, in_channels in input_shapes.items():
            projector = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, embed_dims),
                nn.ReLU(inplace=True),
            )
            self.projectors[name] = projector

    def forward(self, inputs: dict[str, torch.Tensor]):
        assert len(inputs) == len(self.projectors)
        outputs = {name: self.projectors[name](x) for name, x in inputs.items()}
        return outputs
