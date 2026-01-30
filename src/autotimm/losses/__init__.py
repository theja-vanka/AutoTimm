"""Loss functions for object detection and other tasks."""

from autotimm.losses.detection import (
    CenternessLoss,
    FCOSLoss,
    FocalLoss,
    GIoULoss,
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
    "MaskLoss",
    "TverskyLoss",
]
