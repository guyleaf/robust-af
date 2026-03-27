import contextlib
from types import ModuleType
from typing import Optional, Union

import torch
import torch.nn as nn
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.loss import v10DetectLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    make_divisible,
)

from ...models import feature_adapters, image_adapters
from ..utils.loss import RobustDetectLoss

ADAPTER_MODULES = []

# longest path matching
try:
    from ...models.feature_adapters import baselines as feature_adapter_baselines
    from ...models.image_adapters import baselines as image_adapter_baselines

    ADAPTER_MODULES += [image_adapter_baselines, feature_adapter_baselines]
except ImportError:
    image_adapter_baselines = None
    feature_adapter_baselines = None

ADAPTER_MODULES += [image_adapters, feature_adapters]


class RobustDetectionModel(tasks.DetectionModel):
    def __init__(
        self,
        cfg,
        ch: int = 3,
        nc: Optional[int] = None,
        verbose: bool = True,
    ):  # model, input channels, number of classes
        """Initialize the Robust YOLOv8 detection model with the given config and parameters."""
        yaml = cfg if isinstance(cfg, dict) else tasks.yaml_model_load(cfg)  # cfg dict

        # determine the range of robust layers
        num_robust_image = len(yaml.get("robust_image", []))
        num_backbone = len(yaml["backbone"])
        num_image = len(yaml.get("robust", []))
        # [start, end)
        self.robust_image_layer_range = (0, num_robust_image)
        self.robust_layer_range = (
            num_robust_image + num_backbone,
            num_robust_image + num_backbone + num_image,
        )

        # A workaround to replace the parsing function
        # Ultralytics codebase is messy. (QQ)
        tasks.parse_model = parse_robust_model
        super().__init__(cfg=yaml, ch=ch, nc=nc, verbose=verbose)

    # handle missing attributes during unpicking
    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("robust_image_layer_range", (0, 0))

    def _map_state_dict(self, csd: dict, verbose: bool = False):
        sd = self.state_dict()

        # calculate number of robust layers
        num_robust_image = (
            self.robust_image_layer_range[1] - self.robust_image_layer_range[0]
        )
        num_robust = self.robust_layer_range[1] - self.robust_layer_range[0]
        num_inserted = num_robust_image + num_robust

        # calculate number of parameters in robust layers
        num_inserted_keys = 0
        for i in range(*self.robust_image_layer_range):
            num_inserted_keys += len(self.model[i].state_dict())
        for i in range(*self.robust_layer_range):
            num_inserted_keys += len(self.model[i].state_dict())

        # for compatibility, handle base detector checkpoint
        # map the weights to correct index
        if len(csd) + num_inserted_keys == len(sd):
            # Example
            # checkpoint:                        backbone[0..10]                        head[11..23]
            # model:       robust_image[0..R-1]  backbone[R..R+10]  robust[R+11..R+13]  head[R+14..R+26]
            new_csd = {}
            backbone_end_in_ckpt = self.robust_layer_range[0] - num_robust_image
            for k, v in csd.items():
                k_splits = k.split(".")
                layer_i = int(k_splits[1])

                if layer_i < backbone_end_in_ckpt:
                    # backbone layer -> offset by num_robust_image
                    layer_i += num_robust_image
                else:
                    # head layer -> offset by num_robust_image + num_robust
                    layer_i += num_inserted

                # map to new key
                k_splits[1] = str(layer_i)
                new_k = ".".join(k_splits)
                if verbose:
                    LOGGER.info(f"{colorstr('load_state_dict:')} map {k} to {new_k}.")
                new_csd[new_k] = v
            csd = new_csd
        return csd

    def _predict_once(
        self,
        x: torch.Tensor,
        profile: bool = False,
        visualize: bool = False,
        embed: Optional[list] = None,
        robust: bool = True,
        returns_rhss: bool = False,
    ):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        If returns_robust_hidden_states is True:
            (list[torch.Tensor]): The hidden states of robust layers.
        """
        y, dt, embeddings, rhss = [], [], [], []  # outputs
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers

            is_robust_layer = (
                self.robust_image_layer_range[0] <= i < self.robust_image_layer_range[1]
                or self.robust_layer_range[0] <= i < self.robust_layer_range[1]
            )
            should_run = not is_robust_layer or robust
            if should_run:
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)  # run
            # else act as identity

            if returns_rhss and is_robust_layer:
                rhss.append(x)
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(
                    nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                )  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        if returns_rhss:
            index = self.robust_image_layer_range[1]
            return x, (rhss[:index], rhss[index:])
        else:
            return x

    def load(self, weights: Union[dict, nn.Module], verbose: bool = True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = (
            weights["model"] if isinstance(weights, dict) else weights
        )  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = self._map_state_dict(csd, verbose=verbose)
        incompatible_keys = self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Missing keys: {len(incompatible_keys.missing_keys)} items")
            LOGGER.info(
                f"Unexpected keys: {len(incompatible_keys.unexpected_keys)} items"
            )
            LOGGER.info(
                f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights"
            )

    def loss(self, batch: dict, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        clear_img = batch["clear"]["img"]
        assert img.shape == clear_img.shape, (
            "The degraded and normal images should be a pair."
        )

        # NOTE: we need hidden states. so, we cannot reuse preds without returning hidden states.
        preds, (image_rhss, rhss) = self._predict_once(img, returns_rhss=True)
        with torch.no_grad():
            _, (clear_image_rhss, clear_rhss) = self._predict_once(
                clear_img, robust=False, returns_rhss=True
            )
        preds = dict(
            preds=preds,
            image_rhss=image_rhss,
            clear_image_rhss=clear_image_rhss,
            rhss=rhss,
            clear_rhss=clear_rhss,
        )
        return self.criterion(preds, batch)

    def _build_basic_loss(self, cfg) -> tuple[Optional[nn.Module], float]:
        if cfg is None or cfg["module"] is None:
            return None, 20
        else:
            loss = get_module(cfg["module"])(reduction="none")
            weight = cfg["weight"]
            return loss, weight

    def _build_robust_loss(self, criterion):
        image_cst_loss, image_cst_loss_weight = self._build_basic_loss(
            self.args.image_cst_loss
        )
        cst_loss, cst_loss_weight = self._build_basic_loss(self.args.cst_loss)
        LOGGER.info(f"image_cst_loss: {image_cst_loss}")
        LOGGER.info(f"cst_loss: {cst_loss}")
        return RobustDetectLoss(
            criterion,
            image_cst_loss=image_cst_loss,
            cst_loss=cst_loss,
            weight_dict=dict(
                image_cst_loss=image_cst_loss_weight, cst_loss=cst_loss_weight
            ),
        )

    def init_criterion(self):
        return self._build_robust_loss(super().init_criterion())


class RobustYOLOv10DetectionModel(RobustDetectionModel):
    def init_criterion(self):
        return self._build_robust_loss(v10DetectLoss(self))


def get_module_path(module: ModuleType, prefix: str = "robust_af.models."):
    return module.__name__.removeprefix(prefix)


def get_module(module: str) -> type[nn.Module]:
    if module.startswith("nn."):
        return getattr(nn, module[3:])

    for submodule in ADAPTER_MODULES:
        prefix = f"{get_module_path(submodule)}."
        if module.startswith(prefix):
            return getattr(submodule, module[len(prefix) :])

    return getattr(modules, module)


def parse_robust_model(
    d: dict, ch: int, verbose: bool = True
):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (
        d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(
                f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'."
            )
        depth, width, max_channels = scales[scale]

    if act:
        modules.Conv.default_act = eval(
            act
        )  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
        )
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m_name, args) in enumerate(
        d.get("robust_image", []) + d["backbone"] + d.get("robust", []) + d["head"]
    ):  # from, number, module, args
        # backward compatibility with old checkpoints
        if m_name.startswith("robust_modules."):
            m_name = m_name.replace(
                "robust_modules.", f"{get_module_path(feature_adapters)}."
            )
        m = get_module(m_name)

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            modules.Classify,
            modules.Conv,
            modules.ConvTranspose,
            modules.GhostConv,
            modules.Bottleneck,
            modules.GhostBottleneck,
            modules.SPP,
            modules.SPPF,
            modules.DWConv,
            modules.Focus,
            modules.BottleneckCSP,
            modules.C1,
            modules.C2,
            modules.C2f,
            modules.RepNCSPELAN4,
            modules.ADown,
            modules.SPPELAN,
            modules.C2fAttn,
            modules.C3,
            modules.C3TR,
            modules.C3Ghost,
            nn.ConvTranspose2d,
            modules.DWConvTranspose2d,
            modules.C3x,
            modules.RepC3,
            modules.PSA,
            modules.SCDown,
            modules.C2fCIB,
        }:
            c1, c2 = ch[f], args[0]
            if (
                c2 != nc
            ):  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is modules.C2fAttn:
                args[1] = make_divisible(
                    min(args[1], max_channels // 2) * width, 8
                )  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                    if args[2] > 1
                    else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {
                modules.BottleneckCSP,
                modules.C1,
                modules.C2,
                modules.C2f,
                modules.C2fAttn,
                modules.C3,
                modules.C3TR,
                modules.C3Ghost,
                modules.C3x,
                modules.RepC3,
                modules.C2fCIB,
            }:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is modules.AIFI:
            args = [ch[f], *args]
        elif m in {modules.HGStem, modules.HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is modules.HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is modules.ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is modules.Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {
            modules.Detect,
            modules.WorldDetect,
            modules.Segment,
            modules.Pose,
            modules.OBB,
            modules.ImagePoolingAttn,
            modules.v10Detect,
        }:
            args.append([ch[x] for x in f])
            if m is modules.Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif (
            m is modules.RTDETRDecoder
        ):  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is modules.CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is modules.CBFuse:
            c2 = ch[f[-1]]
        elif any(
            m_name.startswith(f"{get_module_path(module)}.")
            for module in ADAPTER_MODULES
        ):
            c2 = ch[f]
            if m_name.startswith(get_module_path(feature_adapters)):
                args = [c2, *args]
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(
                f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}"
            )  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
