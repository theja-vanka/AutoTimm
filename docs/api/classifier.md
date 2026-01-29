# ImageClassifier

End-to-end image classifier backed by a timm backbone.

## Overview

`ImageClassifier` is a PyTorch Lightning module that combines:

- A timm backbone (feature extractor)
- A classification head
- Configurable metrics
- Optimizer and scheduler setup

## API Reference

::: autotimm.ImageClassifier
    options:
      show_source: true
      members:
        - __init__
        - forward
        - training_step
        - validation_step
        - test_step
        - predict_step
        - configure_optimizers

## Usage Examples

### Basic Usage

```python
from autotimm import ImageClassifier, MetricConfig

metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
]

model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)
```

### With BackboneConfig

```python
from autotimm import BackboneConfig, ImageClassifier

cfg = BackboneConfig(
    model_name="vit_base_patch16_224",
    pretrained=True,
    drop_path_rate=0.1,
)

model = ImageClassifier(
    backbone=cfg,
    num_classes=100,
    metrics=metrics,
)
```

### With Enhanced Logging

```python
from autotimm import LoggingConfig

model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    logging_config=LoggingConfig(
        log_learning_rate=True,
        log_gradient_norm=True,
        log_confusion_matrix=True,
    ),
)
```

### With Custom Optimizer

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    optimizer="adamw",
    lr=1e-3,
    weight_decay=1e-4,
    optimizer_kwargs={"betas": (0.9, 0.999)},
)
```

### With Custom Scheduler

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    scheduler="onecycle",
    scheduler_kwargs={"max_lr": 1e-2},
)
```

### Frozen Backbone (Linear Probing)

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    freeze_backbone=True,
    lr=1e-2,
)
```

### With Regularization

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    label_smoothing=0.1,
    mixup_alpha=0.2,
    head_dropout=0.5,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone` | `str \| BackboneConfig` | Required | Model name or config |
| `num_classes` | `int` | Required | Number of target classes |
| `metrics` | `MetricManager \| list[MetricConfig]` | Required | Metrics configuration |
| `logging_config` | `LoggingConfig \| None` | `None` | Enhanced logging options |
| `lr` | `float` | `1e-3` | Learning rate |
| `weight_decay` | `float` | `1e-4` | Weight decay |
| `optimizer` | `str \| dict` | `"adamw"` | Optimizer name or config |
| `optimizer_kwargs` | `dict \| None` | `None` | Extra optimizer kwargs |
| `scheduler` | `str \| dict \| None` | `"cosine"` | Scheduler name or config |
| `scheduler_kwargs` | `dict \| None` | `None` | Extra scheduler kwargs |
| `head_dropout` | `float` | `0.0` | Dropout before classifier |
| `label_smoothing` | `float` | `0.0` | Label smoothing factor |
| `freeze_backbone` | `bool` | `False` | Freeze backbone weights |
| `mixup_alpha` | `float` | `0.0` | Mixup augmentation alpha |

## Supported Optimizers

**Torch:**
`adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`

**Timm:**
`adamp`, `sgdp`, `adabelief`, `radam`, `lamb`, `lars`, `madgrad`, `novograd`

**Custom:**
```python
optimizer={
    "class": "torch.optim.AdamW",
    "params": {"betas": (0.9, 0.999)},
}
```

## Supported Schedulers

**Torch:**
`cosine`, `step`, `multistep`, `exponential`, `onecycle`, `plateau`

**Timm:**
`cosine_with_restarts`

**Custom:**
```python
scheduler={
    "class": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
    "params": {"T_0": 10},
}
```

## Model Architecture

```
ImageClassifier
├── backbone (timm model, headless)
│   └── num_features outputs
├── head (ClassificationHead)
│   ├── dropout (if head_dropout > 0)
│   └── Linear(num_features, num_classes)
└── criterion (CrossEntropyLoss)
```

## Logged Metrics

| Metric | Stage | Condition |
|--------|-------|-----------|
| `{stage}/loss` | train, val, test | Always |
| `{stage}/{metric_name}` | As configured | Per MetricConfig |
| `train/lr` | train | `log_learning_rate=True` |
| `train/grad_norm` | train | `log_gradient_norm=True` |
| `train/weight_norm` | train | `log_weight_norm=True` |
| `val/confusion_matrix` | val | `log_confusion_matrix=True` |
