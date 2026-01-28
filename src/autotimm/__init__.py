"""AutoTimm: automated deep learning image tasks powered by timm and PyTorch Lightning."""

from autotimm._version import __version__
from autotimm.backbone import BackboneConfig, create_backbone, list_backbones
from autotimm.data.datamodule import ImageDataModule
from autotimm.heads import ClassificationHead
from autotimm.loggers import create_logger
from autotimm.tasks.classification import ImageClassifier
from autotimm.trainer import create_trainer
from autotimm.utils import count_parameters

__all__ = [
    "__version__",
    "BackboneConfig",
    "ClassificationHead",
    "ImageClassifier",
    "ImageDataModule",
    "count_parameters",
    "create_backbone",
    "create_logger",
    "create_trainer",
    "list_backbones",
]
