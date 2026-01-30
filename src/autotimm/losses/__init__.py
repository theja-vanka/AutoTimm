"""Loss functions for object detection and other tasks."""

from autotimm.losses.detection import (
    CenternessLoss,
    FCOSLoss,
    FocalLoss,
    GIoULoss,
)

__all__ = [
    "CenternessLoss",
    "FCOSLoss",
    "FocalLoss",
    "GIoULoss",
]
