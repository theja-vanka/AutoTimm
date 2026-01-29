"""AutoTimm: automated deep learning image tasks powered by timm and PyTorch Lightning."""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("autotimm")
except PackageNotFoundError:
    __version__ = "unknown"

from autotimm.backbone import BackboneConfig, create_backbone, list_backbones
from autotimm.data.datamodule import ImageDataModule
from autotimm.heads import ClassificationHead
from autotimm.loggers import LoggerConfig, LoggerManager
from autotimm.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.tasks.classification import ImageClassifier
from autotimm.trainer import AutoTrainer, TunerConfig
from autotimm.utils import count_parameters, list_optimizers, list_schedulers

__all__ = [
    "__version__",
    "AutoTrainer",
    "BackboneConfig",
    "ClassificationHead",
    "ImageClassifier",
    "ImageDataModule",
    "LoggerConfig",
    "LoggerManager",
    "LoggingConfig",
    "MetricConfig",
    "MetricManager",
    "TunerConfig",
    "count_parameters",
    "create_backbone",
    "list_backbones",
    "list_optimizers",
    "list_schedulers",
]
