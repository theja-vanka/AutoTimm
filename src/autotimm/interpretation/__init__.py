"""Model interpretation and visualization tools for AutoTimm."""

from autotimm.interpretation.base import BaseInterpreter
from autotimm.interpretation.gradcam import GradCAM, GradCAMPlusPlus
from autotimm.interpretation.integrated_gradients import (
    IntegratedGradients,
    SmoothGrad,
)
from autotimm.interpretation.attention import AttentionRollout, AttentionFlow
from autotimm.interpretation.api import (
    explain_prediction,
    visualize_batch,
    compare_methods,
    quick_explain,
)
from autotimm.interpretation.adapters import (
    explain_detection,
    explain_segmentation,
)
from autotimm.interpretation.feature_viz import FeatureVisualizer
from autotimm.interpretation.callbacks import (
    InterpretationCallback,
    FeatureMonitorCallback,
)
from autotimm.interpretation.metrics import ExplanationMetrics

# Optional: Interactive visualization (requires plotly)
try:
    from autotimm.interpretation.interactive import InteractiveVisualizer

    _INTERACTIVE_AVAILABLE = True
except ImportError:
    _INTERACTIVE_AVAILABLE = False
    InteractiveVisualizer = None

# Performance optimization utilities
from autotimm.interpretation.optimization import (
    ExplanationCache,
    BatchProcessor,
    PerformanceProfiler,
    optimize_for_inference,
)

__all__ = [
    # Base
    "BaseInterpreter",
    # Methods
    "GradCAM",
    "GradCAMPlusPlus",
    "IntegratedGradients",
    "SmoothGrad",
    "AttentionRollout",
    "AttentionFlow",
    # High-level API
    "explain_prediction",
    "visualize_batch",
    "compare_methods",
    "quick_explain",
    # Task-specific
    "explain_detection",
    "explain_segmentation",
    # Feature visualization
    "FeatureVisualizer",
    # Callbacks
    "InterpretationCallback",
    "FeatureMonitorCallback",
    # Metrics
    "ExplanationMetrics",
    # Interactive (optional)
    "InteractiveVisualizer",
    # Performance optimization
    "ExplanationCache",
    "BatchProcessor",
    "PerformanceProfiler",
    "optimize_for_inference",
]
