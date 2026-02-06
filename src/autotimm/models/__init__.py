"""YOLOX model components."""

from autotimm.models.csp_darknet import (
    BaseConv,
    Bottleneck,
    CSPDarknet,
    CSPLayer,
    DWConv,
    Focus,
    SPPBottleneck,
    SiLU,
    build_csp_darknet,
    get_activation,
)
from autotimm.models.yolox_pafpn import YOLOXPAFPN, build_yolox_pafpn
from autotimm.models.yolox_scheduler import YOLOXLRScheduler, YOLOXWarmupLR
from autotimm.models.yolox_utils import (
    get_yolox_model_info,
    list_yolox_backbones,
    list_yolox_heads,
    list_yolox_models,
    list_yolox_necks,
)

__all__ = [
    # CSPDarknet components
    "BaseConv",
    "Bottleneck",
    "CSPDarknet",
    "CSPLayer",
    "DWConv",
    "Focus",
    "SPPBottleneck",
    "SiLU",
    "build_csp_darknet",
    "get_activation",
    # YOLOXPAFPN components
    "YOLOXPAFPN",
    "build_yolox_pafpn",
    # Schedulers
    "YOLOXLRScheduler",
    "YOLOXWarmupLR",
    # Utilities
    "list_yolox_models",
    "list_yolox_backbones",
    "list_yolox_necks",
    "list_yolox_heads",
    "get_yolox_model_info",
]
