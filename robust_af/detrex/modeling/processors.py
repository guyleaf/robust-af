import torch
import torch.nn as nn


class MultiScaleProcessor(nn.Module):
    def __init__(self, **kwargs: nn.Module):
        super().__init__()
        self.module_mapper = nn.ModuleDict(kwargs)

    def forward(self, *inputs: dict[str, torch.Tensor]):
        """Forward function for MultiScaleProcessor

        Args:
            inputs (Dict[str, torch.Tensor]): The backbone feature maps.

        Return:
            Dict[str, torch.Tensor]: A dict of the processed features.
        """
        new_feats = {}
        for k in inputs[0]:
            args = [feats[k] for feats in inputs]
            new_feats[k] = self.module_mapper[k](*args)
        return new_feats
