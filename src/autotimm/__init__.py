"""AutoTimm: automated deep learning image tasks powered by timm and PyTorch Lightning."""

import sys

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("autotimm")
except PackageNotFoundError:
    __version__ = "unknown"

from autotimm.backbone import (
    BackboneConfig,
    FeatureBackboneConfig,
    ModelSource,
    create_backbone,
    create_feature_backbone,
    get_feature_channels,
    get_feature_info,
    get_feature_strides,
    get_model_source,
    list_backbones,
    list_hf_hub_backbones,
)
from autotimm.models import (
    get_yolox_model_info,
    list_yolox_backbones,
    list_yolox_heads,
    list_yolox_models,
    list_yolox_necks,
)
from autotimm.data.datamodule import ImageDataModule
from autotimm.data.detection_datamodule import DetectionDataModule
from autotimm.data.multilabel_datamodule import MultiLabelImageDataModule
from autotimm.data.instance_datamodule import InstanceSegmentationDataModule
from autotimm.data.segmentation_datamodule import SegmentationDataModule
from autotimm.data.timm_transforms import (
    create_inference_transform,
    get_transforms_from_backbone,
    resolve_backbone_data_config,
)
from autotimm.data.transform_config import TransformConfig, list_transform_presets
from autotimm.data.preset_manager import (
    BackendRecommendation,
    compare_backends,
    recommend_backend,
)
from autotimm.heads import (
    ASPP,
    ClassificationHead,
    DeepLabV3PlusHead,
    DetectionHead,
    FCNHead,
    FPN,
    MaskHead,
    YOLOXHead,
)
from autotimm.loggers import LoggerConfig, LoggerManager
from autotimm.losses import CenternessLoss, FCOSLoss, FocalLoss, GIoULoss
from autotimm.losses.segmentation import (
    CombinedSegmentationLoss,
    DiceLoss,
    FocalLossPixelwise,
    MaskLoss,
    TverskyLoss,
)
from autotimm.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.tasks.classification import ImageClassifier
from autotimm.tasks.instance_segmentation import InstanceSegmentor
from autotimm.tasks.object_detection import ObjectDetector
from autotimm.tasks.semantic_segmentation import SemanticSegmentor
from autotimm.tasks.yolox_detector import YOLOXDetector
from autotimm.trainer import AutoTrainer, TunerConfig
from autotimm.utils import (
    count_parameters,
    list_optimizers,
    list_schedulers,
    seed_everything,
)
from autotimm.export import (
    export_to_torchscript,
    load_torchscript,
    export_checkpoint_to_torchscript,
    validate_torchscript_export,
)

# Interpretation
from autotimm.interpretation import (
    GradCAM,
    GradCAMPlusPlus,
    IntegratedGradients,
    SmoothGrad,
    AttentionRollout,
    AttentionFlow,
    explain_prediction,
    compare_methods,
    visualize_batch,
    explain_detection,
    explain_segmentation,
    FeatureVisualizer,
    InterpretationCallback,
    FeatureMonitorCallback,
    ExplanationMetrics,
    InteractiveVisualizer,
)

# Import submodules for convenient access
from autotimm import data
from autotimm import heads
from autotimm import losses
from autotimm import metrics as metrics_module
from autotimm import tasks
from autotimm import interpretation

# Create module aliases by registering them in sys.modules
# This allows: import autotimm.loss, from autotimm.loss import DiceLoss, etc.
sys.modules["autotimm.loss"] = losses
sys.modules["autotimm.metric"] = metrics_module
sys.modules["autotimm.head"] = heads
sys.modules["autotimm.task"] = tasks

# Also make them available as attributes for: autotimm.loss.DiceLoss
loss = losses
metric = metrics_module
head = heads
task = tasks

__all__ = [
    "__version__",
    # Trainer
    "AutoTrainer",
    "TunerConfig",
    # Backbone
    "BackboneConfig",
    "FeatureBackboneConfig",
    "ModelSource",
    "create_backbone",
    "create_feature_backbone",
    "get_feature_channels",
    "get_feature_info",
    "get_feature_strides",
    "get_model_source",
    "list_backbones",
    "list_hf_hub_backbones",
    # Heads
    "ASPP",
    "ClassificationHead",
    "DeepLabV3PlusHead",
    "DetectionHead",
    "FCNHead",
    "FPN",
    "MaskHead",
    "YOLOXHead",
    # Tasks
    "ImageClassifier",
    "InstanceSegmentor",
    "ObjectDetector",
    "SemanticSegmentor",
    "YOLOXDetector",
    # Data
    "DetectionDataModule",
    "ImageDataModule",
    "InstanceSegmentationDataModule",
    "MultiLabelImageDataModule",
    "SegmentationDataModule",
    # Transform config
    "TransformConfig",
    "create_inference_transform",
    "get_transforms_from_backbone",
    "list_transform_presets",
    "resolve_backbone_data_config",
    # Preset manager
    "BackendRecommendation",
    "compare_backends",
    "recommend_backend",
    # Losses
    "CenternessLoss",
    "CombinedSegmentationLoss",
    "DiceLoss",
    "FCOSLoss",
    "FocalLoss",
    "FocalLossPixelwise",
    "GIoULoss",
    "MaskLoss",
    "TverskyLoss",
    # Logging & Metrics
    "LoggerConfig",
    "LoggerManager",
    "LoggingConfig",
    "MetricConfig",
    "MetricManager",
    # Utils
    "count_parameters",
    "list_optimizers",
    "list_schedulers",
    "seed_everything",
    # Export
    "export_to_torchscript",
    "load_torchscript",
    "export_checkpoint_to_torchscript",
    "validate_torchscript_export",
    # YOLOX Utils
    "list_yolox_models",
    "list_yolox_backbones",
    "list_yolox_necks",
    "list_yolox_heads",
    "get_yolox_model_info",
    # Interpretation
    "GradCAM",
    "GradCAMPlusPlus",
    "IntegratedGradients",
    "SmoothGrad",
    "AttentionRollout",
    "AttentionFlow",
    "explain_prediction",
    "compare_methods",
    "visualize_batch",
    "explain_detection",
    "explain_segmentation",
    "FeatureVisualizer",
    "InterpretationCallback",
    "FeatureMonitorCallback",
    "ExplanationMetrics",
    "InteractiveVisualizer",
    # Submodules
    "data",
    "heads",
    "losses",
    "metrics_module",
    "tasks",
    "interpretation",
    # Submodule aliases
    "loss",
    "metric",
    "head",
    "task",
]
