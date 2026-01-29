# API Reference

This section provides detailed API documentation for all AutoTimm classes and functions.

## Core Classes

| Class | Description |
|-------|-------------|
| [ImageClassifier](classifier.md) | End-to-end image classifier with timm backbone |
| [ImageDataModule](data.md) | Data module for images (folder or built-in datasets) |
| [AutoTrainer](trainer.md) | Configured PyTorch Lightning Trainer |

## Configuration Classes

| Class | Description |
|-------|-------------|
| [BackboneConfig](backbone.md#autotimm.BackboneConfig) | Timm backbone configuration |
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

## Utility Functions

| Function | Description |
|----------|-------------|
| [create_backbone](backbone.md#autotimm.create_backbone) | Create a headless timm model |
| [list_backbones](backbone.md#autotimm.list_backbones) | Search available timm models |
| [count_parameters](utils.md#autotimm.count_parameters) | Count model parameters |
| [list_optimizers](utils.md#autotimm.list_optimizers) | List available optimizers |
| [list_schedulers](utils.md#autotimm.list_schedulers) | List available schedulers |

## Quick Links

### Getting Started

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
    MetricManager,
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
    create_backbone,
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
├── __init__.py           # Public API exports
├── backbone.py           # BackboneConfig, create_backbone, list_backbones
├── data/
│   ├── datamodule.py     # ImageDataModule
│   ├── dataset.py        # ImageFolderCV2
│   └── transforms.py     # Transform presets
├── heads.py              # ClassificationHead
├── loggers.py            # LoggerConfig, LoggerManager
├── metrics.py            # MetricConfig, MetricManager, LoggingConfig
├── tasks/
│   └── classification.py # ImageClassifier
├── trainer.py            # AutoTrainer, TunerConfig
└── utils.py              # Utility functions
```
