# ObjectDetector

FCOS-style anchor-free object detector with timm backbones and Feature Pyramid Networks.

## Overview

`ObjectDetector` is a PyTorch Lightning module for object detection that combines:

- Any timm backbone for feature extraction
- Feature Pyramid Network (FPN) for multi-scale features
- FCOS-style detection head with classification, bbox regression, and centerness
- Focal Loss, GIoU Loss, and Centerness Loss
- NMS post-processing for inference
- Configurable optimizer and scheduler

## API Reference

::: autotimm.ObjectDetector
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
from autotimm import ObjectDetector, MetricConfig

metrics = [
    MetricConfig(
        name="mAP",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "bbox"},
        stages=["val", "test"],
        prog_bar=True,
    ),
]

model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    lr=1e-4,
)
```

### With FeatureBackboneConfig

```python
from autotimm import FeatureBackboneConfig, ObjectDetector

cfg = FeatureBackboneConfig(
    model_name="resnet50",
    pretrained=True,
    out_indices=(2, 3, 4),  # C3, C4, C5
)

model = ObjectDetector(
    backbone=cfg,
    num_classes=80,
    metrics=metrics,
)
```

### With Transformer Backbone

```python
model = ObjectDetector(
    backbone="swin_tiny_patch4_window7_224",
    num_classes=80,
    metrics=metrics,
    lr=1e-5,  # Lower LR for transformers
    fpn_channels=256,
)
```

### Custom FPN and Head

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    fpn_channels=256,      # FPN channels (128, 256, or 512)
    head_num_convs=4,      # Number of conv layers in head
)
```

### Custom Loss Configuration

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    focal_alpha=0.25,
    focal_gamma=2.0,
    cls_loss_weight=1.0,
    reg_loss_weight=1.0,
    centerness_loss_weight=1.0,
)
```

### Custom Inference Settings

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    score_thresh=0.05,              # Confidence threshold
    nms_thresh=0.5,                 # NMS IoU threshold
    max_detections_per_image=100,   # Max detections to keep
)
```

### With TransformConfig (Preprocessing)

Enable inference-time preprocessing with model-specific normalization:

```python
from autotimm import ObjectDetector, TransformConfig

model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    transform_config=TransformConfig(),  # Enable preprocess()
)

# Now you can preprocess raw images
from PIL import Image
image = Image.open("test.jpg")
tensor = model.preprocess(image)  # Returns preprocessed tensor
output = model(tensor)
```

### Get Model's Data Config

```python
model = ObjectDetector(
    backbone="swin_tiny_patch4_window7_224",
    num_classes=80,
    metrics=metrics,
    transform_config=TransformConfig(),
)

# Get normalization config
config = model.get_data_config()
print(f"Mean: {config['mean']}")
print(f"Std: {config['std']}")
print(f"Input size: {config['input_size']}")
```

### Frozen Backbone

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    freeze_backbone=True,  # Only train FPN and head
    lr=1e-3,               # Higher LR when backbone frozen
)
```

### With MultiStep Scheduler

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    lr=1e-4,
    scheduler="multistep",
    scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone` | `str \| FeatureBackboneConfig` | Required | Model name or config |
| `num_classes` | `int` | Required | Number of object classes |
| `metrics` | `MetricManager \| list[MetricConfig]` | Required | Metrics configuration |
| `logging_config` | `LoggingConfig \| None` | `None` | Enhanced logging options |
| `transform_config` | `TransformConfig \| None` | `None` | Transform config for preprocessing |
| `lr` | `float` | `1e-4` | Learning rate |
| `weight_decay` | `float` | `1e-4` | Weight decay |
| `optimizer` | `str \| dict` | `"adamw"` | Optimizer name or config |
| `optimizer_kwargs` | `dict \| None` | `None` | Extra optimizer kwargs |
| `scheduler` | `str \| dict \| None` | `"cosine"` | Scheduler name or config |
| `scheduler_kwargs` | `dict \| None` | `None` | Extra scheduler kwargs |
| `fpn_channels` | `int` | `256` | Number of FPN channels |
| `head_num_convs` | `int` | `4` | Conv layers in detection head |
| `focal_alpha` | `float` | `0.25` | Focal loss alpha |
| `focal_gamma` | `float` | `2.0` | Focal loss gamma |
| `cls_loss_weight` | `float` | `1.0` | Classification loss weight |
| `reg_loss_weight` | `float` | `1.0` | Regression loss weight |
| `centerness_loss_weight` | `float` | `1.0` | Centerness loss weight |
| `score_thresh` | `float` | `0.05` | Score threshold for detections |
| `nms_thresh` | `float` | `0.5` | NMS IoU threshold |
| `max_detections_per_image` | `int` | `100` | Max detections per image |
| `freeze_backbone` | `bool` | `False` | Freeze backbone weights |
| `strides` | `tuple[int, ...]` | `(8, 16, 32, 64, 128)` | FPN strides |
| `regress_ranges` | `tuple \| None` | `None` | Custom regression ranges |

## Model Architecture

```
ObjectDetector
├── backbone (timm feature extractor)
│   └── Multi-scale features: C3, C4, C5
├── fpn (Feature Pyramid Network)
│   └── Pyramid levels: P3, P4, P5, P6, P7
├── detection_head (DetectionHead)
│   ├── cls_subnet → classification logits
│   ├── bbox_subnet → bbox offsets (l, t, r, b)
│   └── centerness_subnet → centerness scores
└── loss_fn (FCOSLoss)
    ├── FocalLoss (classification)
    ├── GIoULoss (bbox regression)
    └── CenternessLoss (center-ness)
```

## FCOS Architecture

**Feature Pyramid Network (FPN):**
- Takes C3, C4, C5 features from backbone
- Builds pyramid levels P3-P7 via top-down and lateral connections
- Each pyramid level detects objects at different scales

**Regression Ranges:**
Objects are assigned to FPN levels based on their size:

| Level | Stride | Default Range | Object Size |
|-------|--------|---------------|-------------|
| P3 | 8 | (-1, 64) | Very small |
| P4 | 16 | (64, 128) | Small |
| P5 | 32 | (128, 256) | Medium |
| P6 | 64 | (256, 512) | Large |
| P7 | 128 | (512, ∞) | Very large |

**Detection Head:**
- Shared across all FPN levels
- 3 branches: classification, bbox regression, centerness
- Each branch has 4 conv layers (configurable via `head_num_convs`)

**Loss Functions:**
- **Focal Loss**: Handles class imbalance in one-stage detectors
- **GIoU Loss**: IoU-based metric for bbox regression
- **Centerness Loss**: Suppresses low-quality detections far from object centers

## Backbone Selection

### CNN Backbones

| Backbone | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| `resnet18` | Fast | Good | Quick experiments |
| `resnet50` | Medium | Better | Standard baseline |
| `efficientnet_b3` | Medium | Better | Efficiency |
| `convnext_tiny` | Medium | Best | Modern CNN |
| `resnet101` | Slow | Best | High accuracy |

### Transformer Backbones

| Backbone | Speed | Memory | Use Case |
|----------|-------|--------|----------|
| `swin_tiny_patch4_window7_224` | Fast | Medium | Balanced |
| `swin_small_patch4_window7_224` | Medium | Medium | Production |
| `swin_base_patch4_window7_224` | Slow | High | Maximum accuracy |
| `vit_base_patch16_224` | Slow | High | Research |

**Notes:**
- Swin Transformers work best for detection (hierarchical features)
- Use smaller batch sizes (8-16) with transformers
- Use lower learning rates (1e-5) with transformer backbones

## Logged Metrics

| Metric | Stage | Condition |
|--------|-------|-----------|
| `{stage}/loss` | train, val, test | Always |
| `{stage}/cls_loss` | train, val, test | Always |
| `{stage}/reg_loss` | train, val, test | Always |
| `{stage}/centerness_loss` | train, val, test | Always |
| `{stage}/{metric_name}` | As configured | Per MetricConfig |
| `train/lr` | train | `log_learning_rate=True` |
| `train/grad_norm` | train | `log_gradient_norm=True` |

## Training Tips

### Standard COCO Training

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    lr=1e-4,
    scheduler="multistep",
    scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
)

trainer = AutoTrainer(
    max_epochs=12,
    gradient_clip_val=1.0,
)
```

### Two-Phase Training (Recommended)

```python
# Phase 1: Train FPN and head only
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    freeze_backbone=True,
    lr=1e-3,
)
trainer = AutoTrainer(max_epochs=3)
trainer.fit(model, datamodule=data)

# Phase 2: Fine-tune entire model
for param in model.backbone.parameters():
    param.requires_grad = True
model._lr = 1e-4
trainer = AutoTrainer(max_epochs=12, gradient_clip_val=1.0)
trainer.fit(model, datamodule=data)
```

### Small Object Detection

For better small object detection:

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=num_classes,
    metrics=metrics,
    fpn_channels=256,  # More capacity
    head_num_convs=4,  # Deeper head
    # Adjust regression ranges to emphasize smaller levels
    regress_ranges=(
        (-1, 32),      # P3: extra small
        (32, 64),      # P4: very small
        (64, 128),     # P5: small
        (128, 256),    # P6: medium
        (256, float("inf")),  # P7: large
    ),
)
```
