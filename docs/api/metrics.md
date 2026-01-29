# Metrics

Metric configuration and management for training.

## MetricConfig

Configuration for a single metric.

### API Reference

::: autotimm.MetricConfig
    options:
      show_source: true

### Usage Examples

#### Basic Accuracy

```python
from autotimm import MetricConfig

accuracy = MetricConfig(
    name="accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={"task": "multiclass"},
    stages=["train", "val", "test"],
    prog_bar=True,
)
```

#### F1 Score

```python
f1 = MetricConfig(
    name="f1",
    backend="torchmetrics",
    metric_class="F1Score",
    params={"task": "multiclass", "average": "macro"},
    stages=["val", "test"],
)
```

#### Top-K Accuracy

```python
top5 = MetricConfig(
    name="top5_accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={"task": "multiclass", "top_k": 5},
    stages=["val", "test"],
)
```

#### Custom Metric

```python
custom = MetricConfig(
    name="custom",
    backend="custom",
    metric_class="mypackage.metrics.CustomMetric",
    params={"threshold": 0.5},
    stages=["val"],
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique identifier |
| `backend` | `str` | Required | `"torchmetrics"` or `"custom"` |
| `metric_class` | `str` | Required | Class name or full path |
| `params` | `dict` | Required | Constructor parameters |
| `stages` | `list[str]` | Required | `["train", "val", "test"]` |
| `log_on_step` | `bool` | `False` | Log each step |
| `log_on_epoch` | `bool` | `True` | Log each epoch |
| `prog_bar` | `bool` | `False` | Show in progress bar |

---

## MetricManager

Manages multiple metrics across training stages.

### API Reference

::: autotimm.MetricManager
    options:
      show_source: true
      members:
        - __init__
        - get_train_metrics
        - get_val_metrics
        - get_test_metrics
        - get_metric_config
        - get_metric_by_name
        - get_config_by_name
        - configs
        - num_classes

### Usage Examples

#### Basic Usage

```python
from autotimm import MetricConfig, MetricManager


def main():
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
    ]

    manager = MetricManager(configs=metric_configs, num_classes=10)

    print(f"Number of metrics: {len(manager)}")
    print(f"Number of classes: {manager.num_classes}")


if __name__ == "__main__":
    main()
```

#### Access Stage Metrics

```python
def main():
    # ... create manager ...

    train_metrics = manager.get_train_metrics()  # ModuleDict
    val_metrics = manager.get_val_metrics()
    test_metrics = manager.get_test_metrics()

    # Use in training loop
    for name, metric in train_metrics.items():
        metric.update(preds, targets)
        value = metric.compute()


if __name__ == "__main__":
    main()
```

#### Get Metric Config

```python
def main():
    # ... create manager ...

    config = manager.get_metric_config("val", "accuracy")
    print(config.prog_bar)  # True


if __name__ == "__main__":
    main()
```

#### Access Metrics by Name

```python
def main():
    # ... create manager ...

    # Get metric instance by name
    accuracy_metric = manager.get_metric_by_name("accuracy")
    accuracy_metric = manager.get_metric_by_name("accuracy", stage="val")

    # Get config by name
    config = manager.get_config_by_name("accuracy")
    print(config.stages)  # ["train", "val", "test"]


if __name__ == "__main__":
    main()
```

#### Iterate Over Configs

```python
def main():
    # ... create manager ...

    # Iterate over all configs
    for config in manager:
        print(f"{config.name}: {config.stages}")

    # Access by index
    first_config = manager[0]
    print(f"First metric: {first_config.name}")

    # Length
    print(f"Number of metrics: {len(manager)}")


if __name__ == "__main__":
    main()
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `configs` | `list[MetricConfig]` | List of metric configs |
| `num_classes` | `int` | Number of classes |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_train_metrics()` | `ModuleDict` | Train stage metrics |
| `get_val_metrics()` | `ModuleDict` | Validation stage metrics |
| `get_test_metrics()` | `ModuleDict` | Test stage metrics |
| `get_metric_config(stage, name)` | `MetricConfig \| None` | Get config by stage/name |
| `get_metric_by_name(name, stage)` | `Module \| None` | Get metric instance by name |
| `get_config_by_name(name)` | `MetricConfig \| None` | Get config by name |
| `len(manager)` | `int` | Number of metric configs |
| `iter(manager)` | Iterator | Iterate over configs |
| `manager[i]` | `MetricConfig` | Get config by index |

---

## LoggingConfig

Configuration for enhanced logging during training.

### API Reference

::: autotimm.LoggingConfig
    options:
      show_source: true

### Usage Examples

#### Basic Logging

```python
from autotimm import LoggingConfig

config = LoggingConfig(
    log_learning_rate=True,
    log_gradient_norm=True,
)
```

#### Full Logging

```python
config = LoggingConfig(
    log_learning_rate=True,
    log_gradient_norm=True,
    log_weight_norm=True,
    log_confusion_matrix=True,
    log_predictions=False,
    predictions_per_epoch=8,
    verbosity=2,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_learning_rate` | `bool` | Required | Log LR each step |
| `log_gradient_norm` | `bool` | Required | Log gradient norms |
| `log_weight_norm` | `bool` | `False` | Log weight norms |
| `log_confusion_matrix` | `bool` | `False` | Log confusion matrix |
| `log_predictions` | `bool` | `False` | Log sample predictions |
| `predictions_per_epoch` | `int` | `8` | Predictions to log |
| `verbosity` | `int` | `1` | 0=minimal, 1=normal, 2=verbose |

### Logged Values

| Metric | Key | Condition |
|--------|-----|-----------|
| Learning rate | `train/lr` | `log_learning_rate=True` |
| Gradient norm | `train/grad_norm` | `log_gradient_norm=True` |
| Weight norm | `train/weight_norm` | `log_weight_norm=True` |
| Confusion matrix | `val/confusion_matrix` | `log_confusion_matrix=True` |

---

## Common Torchmetrics

### Classification

| Metric Class | Common Params |
|--------------|---------------|
| `Accuracy` | `task="multiclass"`, `top_k=1` |
| `F1Score` | `task="multiclass"`, `average="macro"` |
| `Precision` | `task="multiclass"`, `average="macro"` |
| `Recall` | `task="multiclass"`, `average="macro"` |
| `AUROC` | `task="multiclass"` |
| `ConfusionMatrix` | `task="multiclass"` |

### Binary Classification

| Metric Class | Common Params |
|--------------|---------------|
| `Accuracy` | `task="binary"` |
| `F1Score` | `task="binary"` |
| `AUROC` | `task="binary"` |
| `Precision` | `task="binary"` |
| `Recall` | `task="binary"` |

### Average Options

| Value | Description |
|-------|-------------|
| `"micro"` | Global average |
| `"macro"` | Unweighted class average |
| `"weighted"` | Weighted by class support |
| `"none"` | Per-class values |

---

## Full Example

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Define metrics
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="top5_accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "top_k": 5},
            stages=["val", "test"],
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="precision",
            backend="torchmetrics",
            metric_class="Precision",
            params={"task": "multiclass", "average": "macro"},
            stages=["test"],
        ),
        MetricConfig(
            name="recall",
            backend="torchmetrics",
            metric_class="Recall",
            params={"task": "multiclass", "average": "macro"},
            stages=["test"],
        ),
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
    )

    # Create model
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
            log_confusion_matrix=True,
        ),
    )

    # Trainer
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```
