# Instance Segmentation

AutoTimm provides Mask R-CNN style instance segmentation combining FCOS object detection with mask prediction.

## Overview

Instance segmentation detects objects and predicts pixel-precise masks for each instance. AutoTimm supports:

- **Architecture**: FCOS detection + Mask Head (Mask R-CNN style)
- **Backbones**: Any timm model with FPN features
- **Losses**: Detection loss (classification + bbox + centerness) + Binary mask loss
- **Datasets**: COCO instance segmentation format
- **Metrics**: Mask mAP, bbox mAP (torchmetrics)

## Quick Start

```python
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule, MetricConfig, AutoTrainer

# Data
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=4,
)

# Metrics
metrics = [
    MetricConfig(
        name="mask_mAP",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "segm"},
        stages=["val", "test"],
        prog_bar=True,
    ),
]

# Model
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    lr=1e-4,
    mask_loss_weight=1.0,
)

# Train
trainer = AutoTrainer(max_epochs=12)
trainer.fit(model, datamodule=data)
```

## Architecture

### Detection Branch

InstanceSegmentor uses the same FCOS detection head as ObjectDetector:
- Classification head (80 classes for COCO)
- Bounding box regression head
- Centerness head

### Mask Branch

The mask head predicts pixel-precise masks for each detected instance:
- Takes ROI-aligned features from FPN
- Applies 4 conv layers → deconv → 1x1 conv
- Outputs [N, num_classes, mask_size, mask_size]
- Default mask_size: 28x28 (upsampled to bbox size during inference)

```python
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    fpn_channels=256,
    mask_size=28,  # ROI mask resolution
)
```

## Loss Functions

### Detection Loss

Same as ObjectDetector:
- **Focal Loss**: Classification with class imbalance handling
- **GIoU Loss**: Bounding box regression
- **Centerness Loss**: Quality estimation

### Mask Loss

Binary cross-entropy for per-instance masks:

```python
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    mask_loss_weight=1.0,  # Weight for mask loss
)
```

The mask loss is only computed for positive (detected) instances.

## Dataset Format

### COCO Instance Format

AutoTimm uses COCO JSON format with segmentation annotations:

```
coco/
  train2017/
    000000000001.jpg
    000000000002.jpg
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

The JSON annotations include:
- Bounding boxes (COCO format: [x, y, width, height])
- Segmentation masks (RLE or polygon format)
- Category IDs

```python
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=4,
)
```

### Mask Formats

COCO supports two mask formats:
1. **RLE (Run-Length Encoding)**: Compressed binary masks
2. **Polygon**: List of [x, y] coordinates

AutoTimm automatically decodes both formats using pycocotools.

## Data Augmentation

### Presets

```python
# Light augmentation
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    image_size=640,
    augmentation_preset="light",
)

# Default augmentation
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    augmentation_preset="default",
)

# Strong augmentation
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    augmentation_preset="strong",
)
```

### Custom Transforms

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.RandomScale(scale_limit=0.1),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

data = InstanceSegmentationDataModule(
    data_dir="./coco",
    custom_train_transforms=transforms,
)
```

**Note**: Masks are automatically transformed alongside boxes.

## Metrics

### Mask mAP

Primary metric for instance segmentation:

```python
MetricConfig(
    name="mask_mAP",
    backend="torchmetrics",
    metric_class="MeanAveragePrecision",
    params={"box_format": "xyxy", "iou_type": "segm"},
    stages=["val", "test"],
    prog_bar=True,
)
```

### Bbox mAP

Also track detection performance:

```python
MetricConfig(
    name="bbox_mAP",
    backend="torchmetrics",
    metric_class="MeanAveragePrecision",
    params={"box_format": "xyxy", "iou_type": "bbox"},
    stages=["val", "test"],
)
```

## Inference

### Predict on Images

```python
import torch
from PIL import Image
from torchvision import transforms as T

# Load model
model = InstanceSegmentor.load_from_checkpoint("best_model.ckpt")
model.eval()

# Preprocess image
image = Image.open("test.jpg")
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    predictions = model.predict(image_tensor)

# predictions is a list of dicts, one per image:
# {
#     'boxes': [N, 4],    # xyxy format
#     'labels': [N],      # class indices
#     'scores': [N],      # confidence scores
#     'masks': [N, H, W]  # binary masks (0 or 1)
# }

for i, pred in enumerate(predictions):
    print(f"Image {i}: {len(pred['boxes'])} instances")
    for box, label, score, mask in zip(pred['boxes'], pred['labels'], pred['scores'], pred['masks']):
        print(f"  Class {label}, score {score:.3f}, mask size {mask.shape}")
```

### Visualize Results

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_instance_segmentation(image, prediction, threshold=0.5):
    """Visualize instance segmentation results."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    masks = prediction['masks']

    # Filter by score threshold
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks = masks[keep]

    # Draw boxes and masks
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        x1, y1, x2, y2 = box

        # Draw box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

        # Draw label
        ax.text(x1, y1-5, f"Class {label}: {score:.2f}",
               color='white', fontsize=10,
               bbox=dict(facecolor='red', alpha=0.5))

        # Overlay mask
        mask_overlay = np.zeros((*mask.shape, 4))
        mask_overlay[mask > 0.5] = [1, 0, 0, 0.4]  # Red with alpha
        ax.imshow(mask_overlay)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Use it
with torch.no_grad():
    predictions = model.predict(image_tensor)
visualize_instance_segmentation(image, predictions[0])
```

## Advanced Options

### Freeze Backbone

```python
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    freeze_backbone=True,  # Only train detection + mask heads
)
```

### Custom Optimizer

```python
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    optimizer="adamw",
    lr=1e-4,
    weight_decay=1e-5,
)
```

### Detection Parameters

```python
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    score_thresh=0.05,     # Minimum confidence
    nms_thresh=0.5,        # NMS IoU threshold
    max_detections_per_image=100,  # Max instances per image
)
```

### torch.compile (PyTorch 2.0+)

**Enabled by default** for faster training and inference:

```python
# Default: torch.compile enabled
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
)

# Disable if needed
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    compile_model=False,
)

# Custom compile options
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    compile_kwargs={"mode": "reduce-overhead"},
)
```

**What gets compiled:** Backbone + FPN + Detection Head + Mask Head

See [ImageClassifier](image-classifier.md#performance-optimization) for compile mode details.

## Best Practices

### 1. Choose Appropriate Image Size

Larger images capture more detail but require more memory:

```python
# Balanced (default)
data = InstanceSegmentationDataModule(data_dir="./coco", image_size=640)

# High quality (more memory)
data = InstanceSegmentationDataModule(data_dir="./coco", image_size=800)

# Fast training (less accurate)
data = InstanceSegmentationDataModule(data_dir="./coco", image_size=512)
```

### 2. Adjust Mask Loss Weight

Balance detection and mask quality:

```python
# Emphasize mask quality
model = InstanceSegmentor(backbone="resnet50", num_classes=80, mask_loss_weight=2.0)

# Emphasize detection
model = InstanceSegmentor(backbone="resnet50", num_classes=80, mask_loss_weight=0.5)
```

### 3. Use Appropriate Backbones

- **ResNet-50/101**: Good balance
- **Swin Transformer**: Best accuracy
- **EfficientNet**: Memory efficient
- **ResNet-18**: Fast training/inference

### 4. Gradient Clipping

Instance segmentation can have unstable gradients:

```python
trainer = AutoTrainer(
    max_epochs=12,
    gradient_clip_val=1.0,  # Clip gradients
)
```

## Example: COCO Training

Complete example for COCO instance segmentation:

```python
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule, MetricConfig, AutoTrainer

# Data
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=4,
    num_workers=4,
    augmentation_preset="default",
)

# Metrics
metrics = [
    MetricConfig(
        name="mask_mAP",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "segm"},
        stages=["val", "test"],
        prog_bar=True,
    ),
    MetricConfig(
        name="bbox_mAP",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "bbox"},
        stages=["val", "test"],
    ),
]

# Model
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    fpn_channels=256,
    mask_size=28,
    mask_loss_weight=1.0,
    score_thresh=0.05,
    nms_thresh=0.5,
    metrics=metrics,
    lr=1e-4,
    weight_decay=1e-5,
    optimizer="adamw",
    scheduler="cosine",
)

# Trainer
trainer = AutoTrainer(
    max_epochs=12,
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    gradient_clip_val=1.0,
)

# Train
trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

## Troubleshooting

### Out of Memory

Reduce batch size or image size:

```python
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    image_size=512,  # Smaller
    batch_size=2,    # Smaller
)
```

### Slow Training

- Use smaller backbone (ResNet-18, EfficientNet-B0)
- Reduce image size
- Enable mixed precision training
- Reduce mask_size

### Poor Mask Quality

- Increase mask_loss_weight
- Use larger image_size
- Increase mask_size (e.g., 56 instead of 28)
- Train longer

### NaN Loss

- Enable gradient clipping
- Reduce learning rate
- Check dataset annotations are valid

## API Reference

See [InstanceSegmentor API](../../api/segmentation.md#instancesegmentor) for complete parameter documentation.
