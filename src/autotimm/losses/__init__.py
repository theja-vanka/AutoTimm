"""Loss functions for object detection and other tasks."""

from autotimm.losses.detection import (
    CenternessLoss,
    FCOSLoss,
    FocalLoss,
    GIoULoss,
)
from autotimm.losses.registry import (
    LossRegistry,
    get_loss_registry,
    list_available_losses,
    register_custom_loss,
)
from autotimm.losses.segmentation import (
    CombinedSegmentationLoss,
    DiceLoss,
    FocalLossPixelwise,
    MaskLoss,
    TverskyLoss,
)

__all__ = [
    "CenternessLoss",
    "CombinedSegmentationLoss",
    "DiceLoss",
    "FCOSLoss",
    "FocalLoss",
    "FocalLossPixelwise",
    "GIoULoss",
    "LossRegistry",
    "MaskLoss",
    "TverskyLoss",
    "get_loss_registry",
    "list_available_losses",
    "register_custom_loss",
]
