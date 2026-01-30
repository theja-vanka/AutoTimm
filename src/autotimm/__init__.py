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
    create_backbone,
    create_feature_backbone,
    get_feature_channels,
    get_feature_info,
    get_feature_strides,
    list_backbones,
)
from autotimm.data.datamodule import ImageDataModule
from autotimm.data.detection_datamodule import DetectionDataModule
from autotimm.data.instance_datamodule import InstanceSegmentationDataModule
from autotimm.data.segmentation_datamodule import SegmentationDataModule
from autotimm.heads import (
    ASPP,
    ClassificationHead,
    DeepLabV3PlusHead,
    DetectionHead,
    FCNHead,
    FPN,
    MaskHead,
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
from autotimm.trainer import AutoTrainer, TunerConfig
from autotimm.utils import count_parameters, list_optimizers, list_schedulers

# Import submodules for convenient access
from autotimm import data
from autotimm import heads
from autotimm import losses
from autotimm import metrics as metrics_module
from autotimm import tasks

# Create module aliases by registering them in sys.modules
# This allows: import autotimm.loss, from autotimm.loss import DiceLoss, etc.
sys.modules['autotimm.loss'] = losses
sys.modules['autotimm.metric'] = metrics_module
sys.modules['autotimm.head'] = heads
sys.modules['autotimm.task'] = tasks

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
    "create_backbone",
    "create_feature_backbone",
    "get_feature_channels",
    "get_feature_info",
    "get_feature_strides",
    "list_backbones",
    # Heads
    "ASPP",
    "ClassificationHead",
    "DeepLabV3PlusHead",
    "DetectionHead",
    "FCNHead",
    "FPN",
    "MaskHead",
    # Tasks
    "ImageClassifier",
    "InstanceSegmentor",
    "ObjectDetector",
    "SemanticSegmentor",
    # Data
    "DetectionDataModule",
    "ImageDataModule",
    "InstanceSegmentationDataModule",
    "SegmentationDataModule",
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
    # Submodules
    "data",
    "heads",
    "losses",
    "metrics_module",
    "tasks",
    # Submodule aliases
    "loss",
    "metric",
    "head",
    "task",
]
