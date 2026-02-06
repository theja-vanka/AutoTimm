"""Task-specific adapters for interpretation methods."""

from autotimm.interpretation.adapters.detection import explain_detection
from autotimm.interpretation.adapters.segmentation import explain_segmentation

__all__ = [
    "explain_detection",
    "explain_segmentation",
]
