from typing import Optional

import torch
import torch.nn as nn


def _build_activation(name: str, **kwargs) -> nn.Module:
    kwargs.setdefault("inplace", True)
    act = getattr(nn, name)
    try:
        return act(**kwargs)
    except Exception:
        return act()


class SimpleNN(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 32,
        activation: Optional[str] = "LeakyReLU",
    ):
        super().__init__()
        self.conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups, embed_dims)
        if activation is not None:
            self.act = _build_activation(activation)
        else:
            self.act = None

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.conv(x)
        x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return skip + x
