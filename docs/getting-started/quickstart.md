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

## Discovering Available Optimizers and Schedulers

AutoTimm provides utilities to discover all available optimizers and learning rate schedulers from both PyTorch and timm.

### List Optimizers

```python
import autotimm

# Get all optimizers from torch and timm
optimizers = autotimm.list_optimizers()
print("Torch optimizers:", optimizers["torch"])
print("Timm optimizers:", optimizers.get("timm", []))

# Get only torch optimizers
optimizers = autotimm.list_optimizers(include_timm=False)
```

**Available optimizers:**
- **PyTorch**: `adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`, `adadelta`, `adamax`, `asgd`
- **Timm**: `adamp`, `sgdp`, `adabelief`, `radam`, `lamb`, `lars`, `madgrad`, `novograd`

### List Schedulers

```python
# Get all schedulers from torch and timm
schedulers = autotimm.list_schedulers()
print("Torch schedulers:", schedulers["torch"])
print("Timm schedulers:", schedulers.get("timm", []))
```

**Available schedulers:**
- **PyTorch**: `cosineannealinglr`, `cosineannealingwarmrestarts`, `steplr`, `multisteplr`, `exponentiallr`, `onecyclelr`, `reducelronplateau`, and more (15 total)
- **Timm**: `cosinelrscheduler`, `multisteplrscheduler`, `plateaulrscheduler`, `steplrscheduler`, and more (6 total)

### Using Custom Optimizers and Schedulers

```python
# Use a timm optimizer
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    optimizer="adamw",  # or "adamp", "lamb", etc.
    lr=1e-3,
)

# Use a custom scheduler
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    lr=1e-3,
    scheduler="cosineannealinglr",
    scheduler_kwargs={"T_max": 10},
)
```

## Object Detection Quick Start

AutoTimm also supports object detection with FCOS-style anchor-free detectors.

### Basic Object Detection Example

```python
from autotimm import (
    AutoTrainer,
    DetectionDataModule,
    MetricConfig,
    ObjectDetector,
)


def main():
    # Data - COCO format detection dataset
    data = DetectionDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=16,
        augmentation_preset="default",
    )

    # Metrics
    metric_configs = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model - FCOS detector with any timm backbone
    model = ObjectDetector(
        backbone="resnet50",  # Try: swin_tiny, efficientnet_b3, etc.
        num_classes=80,  # COCO has 80 classes
        metrics=metric_configs,
        lr=1e-4,
    )

    # Train
    trainer = AutoTrainer(
        max_epochs=12,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

### Detection with Transformer Backbones

Use Vision Transformers for detection:

```python
# Swin Transformer (recommended for detection)
model = ObjectDetector(
    backbone="swin_tiny_patch4_window7_224",
    num_classes=80,
    metrics=metric_configs,
    lr=1e-4,
)

# Vision Transformer
model = ObjectDetector(
    backbone="vit_base_patch16_224",
    num_classes=80,
    metrics=metric_configs,
    lr=1e-4,
)
```

See [Object Detection Examples](../examples/tasks/object-detection.md#object-detection-on-coco) for more details.

### Alternative: RT-DETR

For end-to-end transformer detection without NMS:

```python
from transformers import RTDetrForObjectDetection
from autotimm import DetectionDataModule

# Use AutoTimm's data loading
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=4,
)

# RT-DETR model
model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd",
    num_labels=80,
)
```

See [RT-DETR Example](../examples/tasks/object-detection.md#rt-detr-real-time-detection-transformer) for complete integration.

## Semantic Segmentation Quick Start

AutoTimm provides DeepLabV3+ and FCN architectures for semantic segmentation.

### Basic Semantic Segmentation Example

```python
from autotimm import (
    AutoTrainer,
    SemanticSegmentor,
    SegmentationDataModule,
    MetricConfig,
)


def main():
    # Data - supports PNG, COCO, Cityscapes, Pascal VOC formats
    data = SegmentationDataModule(
        data_dir="./cityscapes",
        format="cityscapes",  # or "png", "coco", "voc"
        image_size=512,
        batch_size=8,
        augmentation_preset="default",
    )

    # Metrics
    metric_configs = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": 19,
                "average": "macro",
                "ignore_index": 255,
            },
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model - DeepLabV3+ with any timm backbone
    model = SemanticSegmentor(
        backbone="resnet50",  # Try: swin_tiny, efficientnet_b3, etc.
        num_classes=19,  # Cityscapes has 19 classes
        head_type="deeplabv3plus",  # or "fcn"
        loss_type="combined",  # CE + Dice
        metrics=metric_configs,
        lr=1e-4,
    )

    # Train
    trainer = AutoTrainer(
        max_epochs=100,
        precision="16-mixed",  # Mixed precision for faster training
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

### Using Import Aliases

AutoTimm supports cleaner imports:

```python
from autotimm.task import SemanticSegmentor
from autotimm.loss import DiceLoss, CombinedSegmentationLoss
from autotimm.head import DeepLabV3PlusHead
from autotimm.metric import MetricConfig

# Create model using aliases
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
)
```

See [Semantic Segmentation Guide](../user-guide/models/semantic-segmentation.md) for more details.

## Instance Segmentation Quick Start

AutoTimm supports instance segmentation with Mask R-CNN style architecture.

### Basic Instance Segmentation Example

```python
from autotimm import (
    AutoTrainer,
    InstanceSegmentor,
    InstanceSegmentationDataModule,
    MetricConfig,
)


def main():
    # Data - COCO format with masks
    data = InstanceSegmentationDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=4,
        augmentation_preset="default",
    )

    # Metrics
    metric_configs = [
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model - FCOS detection + mask head
    model = InstanceSegmentor(
        backbone="resnet50",
        num_classes=80,  # COCO has 80 classes
        metrics=metric_configs,
        lr=1e-4,
        mask_loss_weight=1.0,
    )

    # Train
    trainer = AutoTrainer(
        max_epochs=12,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

See [Instance Segmentation Guide](../user-guide/models/instance-segmentation.md) for more details.

## Next Steps

- [Data Loading](../user-guide/data-loading/index.md) - Learn about transforms and datasets
- [Models](../user-guide/models/index.md) - Backbone configuration and customization
- [Training](../user-guide/training/training.md) - Advanced training features
- [Examples](../examples/index.md) - More complete examples
