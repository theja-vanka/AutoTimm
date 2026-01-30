# API Reference

This section provides detailed API documentation for all AutoTimm classes and functions.

## Core Classes

| Class | Description |
|-------|-------------|
| [ImageClassifier](classifier.md) | End-to-end image classifier with timm backbone |
| [ObjectDetector](detection.md) | FCOS-style anchor-free object detector |
| [ImageDataModule](data.md) | Data module for images (folder or built-in datasets) |
| [DetectionDataModule](detection_data.md) | Data module for object detection (COCO format) |
| [AutoTrainer](trainer.md) | Configured PyTorch Lightning Trainer |

## Configuration Classes

| Class | Description |
|-------|-------------|
| [BackboneConfig](backbone.md#autotimm.BackboneConfig) | Timm backbone configuration |
| [FeatureBackboneConfig](backbone.md#featurebackboneconfig) | Feature extraction backbone configuration |
| [MetricConfig](metrics.md#autotimm.MetricConfig) | Single metric configuration |
| [MetricManager](metrics.md#autotimm.MetricManager) | Multiple metrics manager |
| [LoggingConfig](metrics.md#autotimm.LoggingConfig) | Enhanced logging options |
| [LoggerConfig](loggers.md#autotimm.LoggerConfig) | Logger backend configuration |
| [LoggerManager](loggers.md#autotimm.LoggerManager) | Multiple loggers manager |
| [TunerConfig](trainer.md#autotimm.TunerConfig) | Auto-tuning configuration |

## Manager Classes API

`MetricManager` and `LoggerManager` share a consistent interface:

| Method | MetricManager | LoggerManager |
|--------|---------------|---------------|
| `len(manager)` | Number of configs | Number of loggers |
| `iter(manager)` | Iterate over configs | Iterate over loggers |
| `manager[i]` | Get config by index | Get logger by index |
| `manager.configs` | List of MetricConfig | List of LoggerConfig |
| Get by name | `get_metric_by_name(name)` | `get_logger_by_backend(name)` |
| Get config | `get_config_by_name(name)` | - |

## Heads

| Head | Description |
|------|-------------|
| [ClassificationHead](heads.md#classificationhead) | Simple classification head with dropout |
| [DetectionHead](heads.md#detectionhead) | FCOS-style detection head (cls, bbox, centerness) |
| [FPN](heads.md#fpn) | Feature Pyramid Network for multi-scale features |

## Loss Functions

| Loss | Description |
|------|-------------|
| [FocalLoss](losses.md#focalloss) | Focal loss for addressing class imbalance |
| [GIoULoss](losses.md#giouloss) | Generalized IoU loss for bounding box regression |
| [CenternessLoss](losses.md#centernessloss) | Binary cross-entropy for centerness prediction |
| [FCOSLoss](losses.md#fcosloss) | Combined FCOS detection loss |

## Utility Functions

| Function | Description |
|----------|-------------|
| [create_backbone](backbone.md#autotimm.create_backbone) | Create a headless timm model |
| [create_feature_backbone](backbone.md#create_feature_backbone) | Create a feature extraction backbone |
| [list_backbones](backbone.md#autotimm.list_backbones) | Search available timm models |
| [get_feature_info](backbone.md#get_feature_info) | Get feature information from backbone |
| [get_feature_channels](backbone.md#get_feature_channels) | Extract feature channels for each level |
| [get_feature_strides](backbone.md#get_feature_strides) | Get stride information for FPN construction |
| [count_parameters](utils.md#autotimm.count_parameters) | Count model parameters |
| [list_optimizers](utils.md#autotimm.list_optimizers) | List available optimizers |
| [list_schedulers](utils.md#autotimm.list_schedulers) | List available schedulers |

## Quick Links

### Image Classification

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
    MetricManager,
)
```

### Object Detection

```python
from autotimm import (
    AutoTrainer,
    ObjectDetector,
    DetectionDataModule,
    FeatureBackboneConfig,
    MetricConfig,
)
```

### With Logging

```python
from autotimm import (
    LoggerConfig,
    LoggerManager,
    LoggingConfig,
)
```

### With Auto-Tuning

```python
from autotimm import TunerConfig
```

### Backbone Utilities

```python
from autotimm import (
    BackboneConfig,
    FeatureBackboneConfig,
    create_backbone,
    create_feature_backbone,
    get_feature_channels,
    get_feature_info,
    get_feature_strides,
    list_backbones,
)
```

### Utilities

```python
from autotimm import (
    count_parameters,
    list_optimizers,
    list_schedulers,
)
```

## Module Structure

```
autotimm/
├── __init__.py                # Public API exports
├── backbone.py                # BackboneConfig, FeatureBackboneConfig, create_backbone, 
│                              # create_feature_backbone, get_feature_*, list_backbones
├── data/
│   ├── datamodule.py          # ImageDataModule
│   ├── dataset.py             # ImageFolderCV2
│   ├── transforms.py          # Transform presets
│   ├── detection_datamodule.py # DetectionDataModule
│   ├── detection_dataset.py   # DetectionDataset
│   └── detection_transforms.py # Detection augmentations
├── heads.py                   # ClassificationHead, DetectionHead, FPN
├── loggers.py                 # LoggerConfig, LoggerManager
├── losses/
│   └── detection.py           # FocalLoss, GIoULoss, CenternessLoss, FCOSLoss
├── metrics.py                 # MetricConfig, MetricManager, LoggingConfig
├── tasks/
│   ├── classification.py      # ImageClassifier
│   └── object_detection.py    # ObjectDetector
├── trainer.py                 # AutoTrainer, TunerConfig
└── utils.py                   # Utility functions
```
