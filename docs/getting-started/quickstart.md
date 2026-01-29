# Quick Start

This guide walks you through training your first image classifier with AutoTimm.

## Basic Training

### 1. Import Required Classes

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
    MetricManager,
)
```

### 2. Set Up Data

AutoTimm supports built-in datasets (CIFAR10, CIFAR100, MNIST, FashionMNIST) and custom folder-based datasets.

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",  # Downloads automatically
    image_size=224,
    batch_size=64,
)
```

### 3. Define Metrics with MetricManager

AutoTimm requires explicit metric configuration using `MetricConfig` and `MetricManager`:

```python
metric_configs = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
]

# Create MetricManager for programmatic access
metric_manager = MetricManager(configs=metric_configs, num_classes=10)

# Access metrics by name
accuracy = metric_manager.get_metric_by_name("accuracy")

# Iterate over configs
for config in metric_manager:
    print(f"{config.name}: {config.stages}")
```

### 4. Create Model

Choose from 1000+ timm backbones:

```python
model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    metrics=metric_manager,
    lr=1e-3,
)
```

### 5. Train

```python
trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
```

### 6. Evaluate

```python
trainer.test(model, datamodule=data)
```

## Complete Example

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
    )

    # Metrics
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
            name="f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        lr=1e-3,
    )

    # Trainer with TensorBoard logging
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
        checkpoint_monitor="val/accuracy",
    )

    # Train and evaluate
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

## Using Custom Datasets

Organize your images in ImageFolder format:

```
dataset/
  train/
    class_a/
      img1.jpg
      img2.jpg
    class_b/
      img3.jpg
  val/
    class_a/
      img4.jpg
    class_b/
      img5.jpg
```

Then load:

```python
def main():
    data = ImageDataModule(
        data_dir="./dataset",
        image_size=224,
        batch_size=32,
    )
    data.setup("fit")

    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    model = ImageClassifier(
        backbone="efficientnet_b0",
        num_classes=data.num_classes,
        metrics=metric_manager,
    )


if __name__ == "__main__":
    main()
```

## Using Different Backbones

Browse available backbones:

```python
import autotimm

# Search by pattern
autotimm.list_backbones("*resnet*")
autotimm.list_backbones("*efficientnet*", pretrained_only=True)
autotimm.list_backbones("*vit*")
```

Popular choices:

| Backbone | Description |
|----------|-------------|
| `resnet18`, `resnet50` | Classic ResNet models |
| `efficientnet_b0` to `efficientnet_b7` | EfficientNet family |
| `vit_base_patch16_224` | Vision Transformer |
| `convnext_tiny` | ConvNeXt models |
| `swin_tiny_patch4_window7_224` | Swin Transformer |

## Next Steps

- [Data Loading](../user-guide/data-loading.md) - Learn about transforms and datasets
- [Models](../user-guide/models.md) - Backbone configuration and customization
- [Training](../user-guide/training.md) - Advanced training features
- [Examples](../examples/index.md) - More complete examples
