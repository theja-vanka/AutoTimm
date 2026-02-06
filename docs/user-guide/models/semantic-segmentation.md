# Semantic Segmentation

AutoTimm provides state-of-the-art semantic segmentation with DeepLabV3+ and FCN architectures.

## Overview

Semantic segmentation assigns a class label to every pixel in an image. AutoTimm supports:

- **Architectures**: DeepLabV3+ (ASPP + decoder), FCN
- **Backbones**: Any timm model with multi-scale features (ResNet, EfficientNet, etc.)
- **Losses**: CrossEntropy, Dice, Focal, Combined (CE + Dice), Tversky
- **Datasets**: PNG masks, COCO stuff, Cityscapes, Pascal VOC

## Quick Start

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig, AutoTrainer

# Data
data = SegmentationDataModule(
    data_dir="./cityscapes",
    format="cityscapes",  # or "png", "coco", "voc"
    image_size=512,
    batch_size=8,
    augmentation_preset="default",
)

# Metrics
metrics = [
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

# Model
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",  # or "fcn"
    loss_type="combined",        # CE + Dice
    dice_weight=1.0,
    metrics=metrics,
)

# Train
trainer = AutoTrainer(max_epochs=100)
trainer.fit(model, datamodule=data)
```

## Architectures

### DeepLabV3+

DeepLabV3+ combines ASPP (Atrous Spatial Pyramid Pooling) with a decoder for high-quality segmentation.

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=21,
    head_type="deeplabv3plus",
    # DeepLabV3+ uses both high-level (C5) and low-level (C2) features
)
```

**Features:**

- ASPP module with multiple dilation rates (6, 12, 18)
- Low-level feature fusion (C2 + ASPP output)
- Decoder with 3x3 convolutions
- Output stride: 4 (1/4 of input resolution)

### FCN (Fully Convolutional Network)

A simpler baseline architecture for comparison.

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=21,
    head_type="fcn",
    # FCN uses only the highest-level feature (C5)
)
```

**Features:**

- Single high-level feature processing
- Lightweight and fast
- Good baseline for simple datasets

## Loss Functions

### Cross-Entropy

Standard pixel-wise classification loss.

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_type="ce",
    ignore_index=255,  # Ignore unlabeled pixels
)
```

### Dice Loss

Overlap-based loss that handles class imbalance well.

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_type="dice",
)
```

**Formula:** `1 - (2 * |X ∩ Y|) / (|X| + |Y|)`

### Combined Loss

Best of both worlds: CE for pixel-wise accuracy + Dice for overlap.

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_type="combined",
    ce_weight=1.0,
    dice_weight=1.0,
)
```

### Focal Loss

Handles severe class imbalance by down-weighting easy examples.

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_type="focal",
)
```

### Tversky Loss

Generalization of Dice with separate control over false positives and false negatives.

```python
from autotimm.loss import TverskyLoss

# Use directly (not via loss_type parameter)
loss_fn = TverskyLoss(
    num_classes=19,
    alpha=0.3,  # Weight for false positives
    beta=0.7,   # Weight for false negatives
)
```

## Datasets

### PNG Format

Simple image + mask pairs.

```
data/
  train/
    images/
      img001.jpg
      img002.jpg
    masks/
      img001.png
      img002.png
  val/
    images/
    masks/
```

```python
data = SegmentationDataModule(
    data_dir="./data",
    format="png",
    image_size=512,
    batch_size=8,
)
```

### Cityscapes

```python
data = SegmentationDataModule(
    data_dir="./cityscapes",
    format="cityscapes",
    image_size=512,
    batch_size=8,
)
```

**Expected structure:**
```
cityscapes/
  leftImg8bit/
    train/
    val/
  gtFine/
    train/
    val/
```

### Pascal VOC

```python
data = SegmentationDataModule(
    data_dir="./VOC2012",
    format="voc",
    image_size=512,
    batch_size=8,
)
```

### COCO Stuff

```python
data = SegmentationDataModule(
    data_dir="./coco",
    format="coco",
    image_size=512,
    batch_size=8,
)
```

## Data Augmentation

### Presets

```python
# Light augmentation
data = SegmentationDataModule(
    data_dir="./data",
    format="png",
    image_size=512,
    augmentation_preset="light",
)

# Default augmentation
data = SegmentationDataModule(
    data_dir="./data",
    format="png",
    augmentation_preset="default",
)

# Strong augmentation
data = SegmentationDataModule(
    data_dir="./data",
    format="png",
    augmentation_preset="strong",
)
```

### Custom Transforms

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.RandomScale(scale_limit=0.5),
    A.RandomCrop(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

data = SegmentationDataModule(
    data_dir="./data",
    format="png",
    custom_train_transforms=transforms,
)
```

## Metrics

### IoU (Jaccard Index)

```python
MetricConfig(
    name="iou",
    backend="torchmetrics",
    metric_class="JaccardIndex",
    params={
        "task": "multiclass",
        "num_classes": 19,
        "average": "macro",  # or "micro", "weighted", None
        "ignore_index": 255,
    },
    stages=["val", "test"],
    prog_bar=True,
)
```

### Per-Class IoU

```python
MetricConfig(
    name="iou_per_class",
    backend="torchmetrics",
    metric_class="JaccardIndex",
    params={
        "task": "multiclass",
        "num_classes": 19,
        "average": None,  # Returns per-class scores
    },
    stages=["val"],
)
```

### Pixel Accuracy

```python
MetricConfig(
    name="accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={
        "task": "multiclass",
        "num_classes": 19,
        "ignore_index": 255,
    },
    stages=["val", "test"],
)
```

## Inference

### Predict on Images

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = SemanticSegmentor.load_from_checkpoint("best_model.ckpt")
model.eval()

# Load and preprocess image
image = Image.open("test.jpg")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    predictions = model.predict(image_tensor)

# predictions shape: [1, H, W] with class indices
```

### Batch Prediction

```python
images = torch.randn(4, 3, 512, 512)  # Batch of 4 images

with torch.no_grad():
    predictions = model.predict(images)
    # Shape: [4, H, W]
```

### Get Probabilities

```python
with torch.no_grad():
    logits = model.predict(images, return_logits=True)
    probs = torch.softmax(logits, dim=1)
    # Shape: [B, num_classes, H, W]
```

## Advanced Options

### Freeze Backbone

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    freeze_backbone=True,  # Only train the head
)
```

### Custom Optimizer

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    optimizer="adamw",
    lr=1e-4,
    weight_decay=1e-5,
)
```

### Learning Rate Scheduler

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    scheduler="cosine",
    scheduler_kwargs={
        "T_max": 100,  # epochs
    },
)
```

### Class Weights

For imbalanced datasets:

```python
import torch

class_weights = torch.tensor([1.0, 2.5, 1.5, ...])  # One per class

model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    loss_type="ce",
    # Note: Pass via loss initialization, not directly to model
)
```

## Best Practices

### 1. Choose the Right Architecture

- **DeepLabV3+**: Best quality, slower training
- **FCN**: Faster, good for simple datasets or baselines

### 2. Select Appropriate Loss

- **Combined (CE + Dice)**: Best overall performance
- **Dice only**: Good for class imbalance
- **CE only**: Fast, works well when classes are balanced
- **Focal**: Severe class imbalance

### 3. Handle Unlabeled Pixels

Always set `ignore_index=255` for datasets with unlabeled regions:

```python
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    ignore_index=255,
)
```

### 4. Use Appropriate Backbones

- **ResNet-50/101**: Good balance of speed and accuracy
- **EfficientNet**: More efficient for similar accuracy
- **MobileNet**: Fastest, lower accuracy
- **ConvNeXt/Swin**: Highest accuracy, slower

### 5. Image Size Considerations

- Larger images = better quality but slower and more memory
- Typical sizes: 512x512 (balanced), 768x768 (high quality), 1024x1024 (best quality)

```python
data = SegmentationDataModule(
    data_dir="./data",
    format="png",
    image_size=512,  # Adjust based on your needs
)
```

## Example: Cityscapes Training

Complete example for Cityscapes dataset:

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig, AutoTrainer

# Data
data = SegmentationDataModule(
    data_dir="./cityscapes",
    format="cityscapes",
    image_size=512,
    batch_size=8,
    num_workers=4,
    augmentation_preset="default",
)

# Metrics
metrics = [
    MetricConfig(
        name="mIoU",
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
    MetricConfig(
        name="pixel_acc",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={
            "task": "multiclass",
            "num_classes": 19,
            "ignore_index": 255,
        },
        stages=["val", "test"],
    ),
]

# Model
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    loss_type="combined",
    ce_weight=1.0,
    dice_weight=1.0,
    ignore_index=255,
    metrics=metrics,
    lr=1e-4,
    weight_decay=1e-5,
    optimizer="adamw",
    scheduler="cosine",
)

# Trainer
trainer = AutoTrainer(
    max_epochs=200,
    accelerator="auto",
    devices=1,
    precision="16-mixed",  # Mixed precision training
)

# Train
trainer.fit(model, datamodule=data)
```

## Troubleshooting

### Out of Memory

Reduce batch size or image size:

```python
data = SegmentationDataModule(
    data_dir="./data",
    format="png",
    image_size=384,  # Smaller
    batch_size=4,    # Smaller
)
```

### Slow Training

- Use smaller backbone (e.g., ResNet-18, MobileNet)
- Use FCN instead of DeepLabV3+
- Enable mixed precision training
- Reduce image size

### Poor Accuracy

- Try combined loss instead of CE only
- Increase training epochs
- Use stronger augmentation
- Try larger backbone
- Check for class imbalance

### Class Imbalance

- Use Dice or Focal loss
- Add class weights
- Use stronger augmentation for minority classes

## API Reference

See [SemanticSegmentor API](../../api/segmentation.md#semanticsegmentor) for complete parameter documentation.
