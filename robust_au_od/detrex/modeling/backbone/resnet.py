import torch
import torch.nn as nn
from detectron2.modeling import ResNet


class RobustResNet(ResNet):
    def __init__(
        self,
        stem: nn.Module,
        robust_module: dict[str, nn.Module],
        stages,
        num_classes=None,
        out_features=None,
        freeze_at=0,
    ):
        super().__init__(stem, stages, num_classes, out_features, freeze_at)
        self.robust_module = nn.ModuleDict(robust_module)
        self._out_robust_features = set(robust_module.keys())
        assert self._out_robust_features.issubset(self.stage_names), (
            "The outputs of robust features should be a subset of stages."
        )

    def _forward_robust(self, name: str, x: torch.Tensor, robust: bool = True):
        if not robust:
            return x

        if name in self.robust_module:
            x = self.robust_module[name](x)
        return x

    def forward(self, x: torch.Tensor, robust: bool = True):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            1. dict[str->Tensor]: names and the corresponding features
            2. dict[str->Tensor]: names and the corresponding robust features
        """
        assert x.dim() == 4, (
            f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        )
        outputs = {}
        robust_outputs = {}
        x = self.stem(x)
        x = self._forward_robust("stem", x, robust=robust)
        if "stem" in self._out_features:
            outputs["stem"] = x
        if "stem" in self._out_robust_features:
            robust_outputs["stem"] = x

        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            x = self._forward_robust(name, x, robust=robust)
            if name in self._out_features:
                outputs[name] = x
            if name in self._out_robust_features:
                robust_outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs, robust_outputs
