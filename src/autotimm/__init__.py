"""AutoTimm: automated deep learning image tasks powered by timm and PyTorch Lightning."""

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
from autotimm.heads import ClassificationHead, DetectionHead, FPN
from autotimm.loggers import LoggerConfig, LoggerManager
from autotimm.losses import CenternessLoss, FCOSLoss, FocalLoss, GIoULoss
from autotimm.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.tasks.classification import ImageClassifier
from autotimm.tasks.object_detection import ObjectDetector
from autotimm.trainer import AutoTrainer, TunerConfig
from autotimm.utils import count_parameters, list_optimizers, list_schedulers

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
    "ClassificationHead",
    "DetectionHead",
    "FPN",
    # Tasks
    "ImageClassifier",
    "ObjectDetector",
    # Data
    "DetectionDataModule",
    "ImageDataModule",
    # Losses
    "CenternessLoss",
    "FCOSLoss",
    "FocalLoss",
    "GIoULoss",
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
]
