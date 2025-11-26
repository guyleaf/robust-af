import torch
import torch.nn as nn


class MultiScaleProcessor(nn.Module):
    def __init__(self, **kwargs: nn.Module):
        super().__init__()
        self.module_mapper = nn.ModuleDict(kwargs)

    def forward(self, feats: dict[str, torch.Tensor]):
        """Forward function for MultiScaleProcessor

        Args:
            inputs (Dict[str, torch.Tensor]): The backbone feature maps.

        Return:
            Dict[str, torch.Tensor]: A dict of the processed features.
        """
        new_feats = {}
        for k, feat in feats.items():
            new_feats[k] = self.module_mapper[k](feat)
        return new_feats
