![AutoTimm](autotimm.png)

Automated deep learning image tasks powered by [timm](https://github.com/huggingface/pytorch-image-models) and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).

AutoTimm lets you train image classifiers with any of timm's 1000+ backbones in a few lines of Python. It features:

- **Configurable metrics** - Use torchmetrics, timm metrics, or custom metrics
- **Multiple logger backends** - TensorBoard, MLflow, W&B, CSV simultaneously
- **Auto-tuning** - Automatic learning rate and batch size finding
- **Enhanced logging** - Learning rate, gradient norms, confusion matrices
- **Flexible transforms** - Torchvision (PIL) or albumentations (OpenCV)

## Installation

```bash
# Core
pip install autotimm

# With albumentations (OpenCV-based transforms)
pip install autotimm[albumentations]

# With specific logger backends
pip install autotimm[tensorboard]
pip install autotimm[wandb]
pip install autotimm[mlflow]

# Everything
pip install autotimm[all]
```

For development:

```bash
git clone https://github.com/your-org/autotimm.git
cd autotimm
pip install -e ".[dev,all]"
```

## Quick Start

```python
from autotimm import (
    AutoTrainer, ImageClassifier, ImageDataModule,
    LoggerConfig, MetricConfig,
)

# Data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=64,
)

# Metrics (explicit configuration required)
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

# Model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    metrics=metrics,
    lr=1e-3,
)

# Trainer with logging
trainer = AutoTrainer(
    max_epochs=10,
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
    checkpoint_monitor="val/accuracy",
)

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

## Configurable Metrics

AutoTimm requires explicit metric configuration, supporting multiple backends:

### Torchmetrics (recommended)

```python
from autotimm import MetricConfig

metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
    MetricConfig(
        name="f1",
        backend="torchmetrics",
        metric_class="F1Score",
        params={"task": "multiclass", "average": "macro"},
        stages=["val", "test"],
    ),
    MetricConfig(
        name="top5_accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass", "top_k": 5},
        stages=["val", "test"],
    ),
]
```

### Timm metrics

```python
MetricConfig(
    name="timm_acc",
    backend="timm",
    metric_class="accuracy",
    params={"topk": (1, 5)},
    stages=["val"],
)
```

### Custom metrics

```python
MetricConfig(
    name="custom",
    backend="custom",
    metric_class="mypackage.metrics.CustomMetric",
    params={"param1": "value"},
    stages=["val"],
)
```

## Configurable Logging

### Single logger

```python
from autotimm import AutoTrainer, LoggerConfig

trainer = AutoTrainer(
    max_epochs=10,
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
)
```

### Multiple loggers

```python
from autotimm import LoggerConfig, LoggerManager

# Option 1: List of configs
trainer = AutoTrainer(
    logger=[
        LoggerConfig(backend="tensorboard", params={"save_dir": "logs/tb"}),
        LoggerConfig(backend="csv", params={"save_dir": "logs/csv"}),
        LoggerConfig(backend="wandb", params={"project": "my-project"}),
    ],
)

# Option 2: LoggerManager
manager = LoggerManager(configs=[
    LoggerConfig(backend="tensorboard", params={"save_dir": "logs"}),
    LoggerConfig(backend="mlflow", params={"experiment_name": "exp1"}),
])
trainer = AutoTrainer(logger=manager)
```

### Enhanced logging

```python
from autotimm import LoggingConfig

model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    metrics=metrics,
    logging_config=LoggingConfig(
        log_learning_rate=True,      # Log LR each step
        log_gradient_norm=True,      # Log gradient norms
        log_confusion_matrix=True,   # Log confusion matrix each epoch
    ),
)
```

## Auto-Tuning

Automatically find optimal learning rate and batch size:

```python
from autotimm import AutoTrainer, TunerConfig

# LR finding only
trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,
        auto_batch_size=False,
        lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0, "num_training": 100},
    ),
)

# Full auto-tuning (batch size + LR)
trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,
        auto_batch_size=True,
        lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0},
        batch_size_kwargs={"mode": "power", "init_val": 16},
    ),
)

trainer.fit(model, datamodule=data)  # Runs tuning before training
```

## Custom Dataset

Organize your images in ImageFolder format:

```
dataset/
  train/
    class_a/
      img1.jpg
    class_b/
      img2.jpg
  val/
    class_a/
      img3.jpg
    class_b/
      img4.jpg
```

Then:

```python
data = ImageDataModule(data_dir="./dataset", image_size=384, batch_size=16)
data.setup("fit")

model = ImageClassifier(
    backbone="efficientnet_b3",
    num_classes=data.num_classes,
    metrics=metrics,
)
```

If no `val/` directory exists, a fraction of the training data is held out automatically (controlled by `val_split`, default 10%).

## Transform Backends

### Torchvision (default, PIL-based)

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_backend="torchvision",     # default
    augmentation_preset="randaugment",   # "default", "autoaugment", "randaugment", "trivialaugment"
)
```

### Albumentations (OpenCV-based)

```bash
pip install autotimm[albumentations]
```

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_backend="albumentations",
    augmentation_preset="strong",   # "default" or "strong"
)
```

### Custom transforms

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

custom_train = A.Compose([
    A.RandomResizedCrop(size=(224, 224)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.8),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

data = ImageDataModule(
    data_dir="./dataset",
    transform_backend="albumentations",
    train_transforms=custom_train,
)
```

## Backbone Discovery

Browse the 1000+ backbones available through timm:

```python
import autotimm

# Search by pattern
autotimm.list_backbones("*convnext*")
autotimm.list_backbones("*efficientnet*", pretrained_only=True)

# Inspect a backbone
backbone = autotimm.create_backbone("resnet50")
print(f"Output features: {backbone.num_features}")
print(f"Parameters: {autotimm.count_parameters(backbone):,}")
```

### Advanced backbone configuration

```python
from autotimm import BackboneConfig, ImageClassifier

cfg = BackboneConfig(
    model_name="vit_base_patch16_224",
    pretrained=True,
    drop_path_rate=0.1,
)
model = ImageClassifier(backbone=cfg, num_classes=100, metrics=metrics)
```

## Training Features

### Freeze backbone for linear probing

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    freeze_backbone=True,
    lr=1e-2,
)
```

### Mixup augmentation

```python
model = ImageClassifier(backbone="resnet50", num_classes=10, metrics=metrics, mixup_alpha=0.2)
```

### Label smoothing

```python
model = ImageClassifier(backbone="resnet50", num_classes=10, metrics=metrics, label_smoothing=0.1)
```

### Learning rate schedulers

```python
model = ImageClassifier(backbone="resnet50", num_classes=10, metrics=metrics, scheduler="cosine")  # or "step", "none"
```

### Gradient accumulation, clipping, and mixed precision

```python
trainer = AutoTrainer(
    max_epochs=20,
    accumulate_grad_batches=4,
    gradient_clip_val=1.0,
    precision="bf16-mixed",
)
```

### Multi-GPU training

```python
trainer = AutoTrainer(
    max_epochs=10,
    accelerator="gpu",
    devices=2,              # Use 2 GPUs
    strategy="ddp",         # Distributed Data Parallel
    precision="bf16-mixed",
)
```

### Two-phase fine-tuning

```python
# Phase 1: Linear probe
model = ImageClassifier(
    backbone="vit_base_patch16_224",
    num_classes=data.num_classes,
    metrics=metrics,
    freeze_backbone=True,
    lr=1e-2,
)
trainer = AutoTrainer(max_epochs=5)
trainer.fit(model, datamodule=data)

# Phase 2: Full fine-tune
for param in model.backbone.parameters():
    param.requires_grad = True

model._lr = 1e-4
trainer = AutoTrainer(max_epochs=20, gradient_clip_val=1.0)
trainer.fit(model, datamodule=data)
```

## Data Module Features

### Balanced sampling for imbalanced datasets

```python
data = ImageDataModule(
    data_dir="./imbalanced_dataset",
    balanced_sampling=True,   # WeightedRandomSampler oversamples minority classes
)
```

### Dataset summary

```python
data = ImageDataModule(data_dir="./dataset", dataset_name="CIFAR10")
data.setup("fit")
print(data.summary())
```

## Examples

The [`examples/`](examples/) directory contains runnable scripts:

| Script | Description |
|---|---|
| [`classify_cifar10.py`](examples/classify_cifar10.py) | ResNet-18 on CIFAR-10 with metrics and auto-tuning |
| [`classify_custom_folder.py`](examples/classify_custom_folder.py) | EfficientNet on a custom folder dataset with W&B |
| [`timm_metrics.py`](examples/timm_metrics.py) | Using timm metrics alongside torchmetrics |
| [`multiple_loggers.py`](examples/multiple_loggers.py) | TensorBoard + CSV logging simultaneously |
| [`auto_tuning.py`](examples/auto_tuning.py) | Automatic LR and batch size finding |
| [`inference.py`](examples/inference.py) | Model inference and batch prediction |
| [`detailed_evaluation.py`](examples/detailed_evaluation.py) | Confusion matrix and per-class metrics |
| [`multi_gpu_training.py`](examples/multi_gpu_training.py) | Multi-GPU and distributed training |
| [`vit_finetuning.py`](examples/vit_finetuning.py) | Two-phase ViT fine-tuning |
| [`balanced_sampling.py`](examples/balanced_sampling.py) | Weighted sampling for imbalanced data |
| [`mlflow_tracking.py`](examples/mlflow_tracking.py) | MLflow experiment tracking |
| [`albumentations_cifar10.py`](examples/albumentations_cifar10.py) | Albumentations strong augmentation |
| [`albumentations_custom_folder.py`](examples/albumentations_custom_folder.py) | Custom albumentations pipeline |
| [`backbone_discovery.py`](examples/backbone_discovery.py) | Explore timm backbones |

## API Reference

### Core Classes

| Class | Description |
|---|---|
| `ImageClassifier` | Image classifier: timm backbone + classification head + training loop |
| `ImageDataModule` | Data module for ImageFolder and built-in datasets |
| `AutoTrainer` | `pl.Trainer` subclass with logger, checkpoint, and tuner support |

### Configuration Classes

| Class | Description |
|---|---|
| `MetricConfig` | Configuration for a single metric (backend, class, params, stages) |
| `MetricManager` | Manages multiple metrics across train/val/test stages |
| `LoggerConfig` | Configuration for a logger backend (tensorboard, mlflow, wandb, csv) |
| `LoggerManager` | Manages multiple loggers |
| `LoggingConfig` | Enhanced logging options (LR, gradients, confusion matrix) |
| `TunerConfig` | Auto-tuning configuration (LR finder, batch size finder) |
| `BackboneConfig` | Timm backbone configuration (model name, pretrained, dropout) |

### Utility Functions

| Function | Description |
|---|---|
| `create_backbone` | Create a headless timm model from a name or config |
| `list_backbones` | Search available timm model names by glob pattern |
| `count_parameters` | Count model parameters (total or trainable) |

## Built-in Datasets

`ImageDataModule` supports these torchvision datasets via the `dataset_name` parameter:

- `CIFAR10`
- `CIFAR100`
- `MNIST`
- `FashionMNIST`

## Augmentation Presets

| Backend | Preset | Description |
|---|---|---|
| `torchvision` | `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `torchvision` | `autoaugment` | AutoAugment (ImageNet policy) |
| `torchvision` | `randaugment` | RandAugment (2 ops, magnitude 9) |
| `torchvision` | `trivialaugment` | TrivialAugmentWide |
| `albumentations` | `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `albumentations` | `strong` | Affine, blur/noise, ColorJitter, CoarseDropout |

## License

Apache 2.0
