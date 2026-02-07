# ObjectDetector

FCOS-style anchor-free object detector that combines timm backbones with Feature Pyramid Networks (FPN) and detection heads.

## Basic Usage

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
    num_classes=80,  # COCO has 80 classes
    metrics=metrics,
    lr=1e-4,
)
```

## Architecture

The ObjectDetector uses the FCOS (Fully Convolutional One-Stage) architecture:

1. **Backbone**: Any timm model extracts multi-scale features
2. **FPN (Feature Pyramid Network)**: Builds pyramid levels P3-P7
3. **Detection Head**: Predicts class, bounding box, and centerness for each location
4. **NMS**: Non-Maximum Suppression filters overlapping detections

## Backbone Selection

Any timm backbone works for object detection. Popular choices:

### CNN Backbones

| Backbone | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| `resnet18` | Fast | Good | Quick experiments |
| `resnet50` | Medium | Better | Standard baseline |
| `efficientnet_b3` | Medium | Better | Efficiency |
| `convnext_tiny` | Medium | Best | Modern CNN |
| `resnet101` | Slow | Best | High accuracy |

### Transformer Backbones

| Backbone | Speed | Accuracy | Memory | Use Case |
|----------|-------|----------|--------|----------|
| `swin_tiny_patch4_window7_224` | Fast | Good | Medium | Balanced performance |
| `swin_small_patch4_window7_224` | Medium | Better | Medium | Production use |
| `swin_base_patch4_window7_224` | Medium | Best | High | Maximum accuracy |
| `vit_base_patch16_224` | Slow | Best | High | Research |
| `deit_base_patch16_224` | Slow | Best | High | Limited data |

```python
# CNN backbones
model = ObjectDetector(backbone="resnet18", num_classes=80, metrics=metrics)
model = ObjectDetector(backbone="efficientnet_b3", num_classes=80, metrics=metrics)
model = ObjectDetector(backbone="convnext_tiny", num_classes=80, metrics=metrics)

# Transformer backbones (recommended: Swin Transformer)
model = ObjectDetector(backbone="swin_tiny_patch4_window7_224", num_classes=80, metrics=metrics)
model = ObjectDetector(backbone="vit_base_patch16_224", num_classes=80, metrics=metrics)
```

**Notes:**

- Swin Transformers work best for detection due to hierarchical features
- Transformers require smaller batch sizes (8-16) due to higher memory usage
- Use lower learning rates (1e-4 to 1e-5) with transformer backbones
- Two-phase training (freeze backbone → fine-tune) works well with transformers

See [Transformer-Based Detection Example](../../examples/tasks/object-detection.md#transformer-based-object-detection) for detailed usage.

## FPN Configuration

Customize the Feature Pyramid Network:

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    fpn_channels=256,      # Number of channels in FPN (128, 256, or 512)
    head_num_convs=4,      # Number of conv layers in detection head (3-5)
)
```

## Loss Configuration

FCOS uses three loss components:

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    # Focal loss parameters
    focal_alpha=0.25,      # Alpha for focal loss (0.25 is standard)
    focal_gamma=2.0,       # Gamma for focal loss (2.0 is standard)
    # Loss weights
    cls_loss_weight=1.0,   # Classification loss weight
    reg_loss_weight=1.0,   # Regression (bbox) loss weight
    centerness_loss_weight=1.0,  # Centerness loss weight
)
```

**Loss Functions:**

- **Focal Loss**: Classification with hard example mining
- **GIoU Loss**: Bounding box regression with IoU-based metric
- **Centerness Loss**: Predicts how centered a location is in its box

## Inference Configuration

Control detection behavior:

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    score_thresh=0.05,     # Minimum confidence score (0.05 for COCO)
    nms_thresh=0.5,        # IoU threshold for NMS (0.5 is standard)
    max_detections_per_image=100,  # Maximum detections to keep
)
```

## Optimizer and Scheduler

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    lr=1e-4,               # Learning rate (1e-4 for detection)
    weight_decay=1e-4,     # Weight decay
    optimizer="adamw",     # "adamw", "sgd", etc.
    scheduler="multistep", # Multi-step LR schedule for COCO
    scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
)
```

Common schedules for object detection:

- **MultiStep**: Drop LR at specific epochs (standard for COCO)
- **Cosine**: Smooth decay over training
- **Step**: Drop LR every N epochs

## Freeze Backbone

For faster training or transfer learning:

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    freeze_backbone=True,  # Only train FPN and detection head
    lr=1e-3,               # Higher LR when backbone is frozen
)
```

## Advanced: Custom Regression Ranges

FCOS assigns objects to FPN levels based on size. Customize these ranges:

```python
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    strides=(8, 16, 32, 64, 128),  # FPN strides (P3-P7)
    regress_ranges=(
        (-1, 64),      # P3: smallest objects
        (64, 128),     # P4
        (128, 256),    # P5
        (256, 512),    # P6
        (512, float("inf")),  # P7: largest objects
    ),
)
```

## Performance Optimization

### torch.compile (PyTorch 2.0+)

**Enabled by default** for automatic optimization:

```python
# Default: torch.compile enabled
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
)
```

Disable or customize:

```python
# Disable compilation
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    compile_model=False,
)

# Custom compile options
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    compile_kwargs={"mode": "reduce-overhead"},
)
```

**What gets compiled:** Backbone + FPN + Detection Head

See [ImageClassifier Performance Optimization](image-classifier.md#performance-optimization) for compile mode details.

## Full Parameter Reference

```python
ObjectDetector(
    backbone="resnet50",                    # Model name or FeatureBackboneConfig
    num_classes=80,                         # Number of object classes
    metrics=metrics,                        # MetricManager or list of MetricConfig
    logging_config=None,                    # LoggingConfig for enhanced logging
    lr=1e-4,                                # Learning rate
    weight_decay=1e-4,                      # Weight decay
    optimizer="adamw",                      # Optimizer name or dict
    optimizer_kwargs=None,                  # Extra optimizer kwargs
    scheduler="cosine",                     # Scheduler name, dict, or None
    scheduler_kwargs=None,                  # Extra scheduler kwargs
    fpn_channels=256,                       # Number of FPN channels
    head_num_convs=4,                       # Conv layers in detection head
    focal_alpha=0.25,                       # Focal loss alpha
    focal_gamma=2.0,                        # Focal loss gamma
    cls_loss_weight=1.0,                    # Classification loss weight
    reg_loss_weight=1.0,                    # Regression loss weight
    centerness_loss_weight=1.0,             # Centerness loss weight
    score_thresh=0.05,                      # Score threshold for detections
    nms_thresh=0.5,                         # NMS IoU threshold
    max_detections_per_image=100,           # Max detections per image
    freeze_backbone=False,                  # Freeze backbone weights
    strides=(8, 16, 32, 64, 128),          # FPN strides
    regress_ranges=None,                    # Custom regression ranges
    compile_model=True,                     # Enable torch.compile (PyTorch 2.0+)
    compile_kwargs=None,                    # Custom torch.compile options
)
```

## Usage Example

```python
from autotimm import AutoTrainer, DetectionDataModule, ObjectDetector, MetricConfig

# Data
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
)

# Model
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=[
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ],
    lr=1e-4,
    scheduler="multistep",
    scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
)

# Train
trainer = AutoTrainer(
    max_epochs=12,
    gradient_clip_val=1.0,
    checkpoint_monitor="val/map",
    checkpoint_mode="max",
)
trainer.fit(model, datamodule=data)
```

---

## Alternative Detection Architectures

While AutoTimm's built-in `ObjectDetector` uses FCOS architecture, you can also integrate other detection architectures with AutoTimm's data loading utilities.

### RT-DETR (Real-Time Detection Transformer)

RT-DETR is an end-to-end transformer-based detector that eliminates the need for NMS (Non-Maximum Suppression).

**Key Differences from FCOS:**

| Feature | RT-DETR | FCOS (AutoTimm) |
|---------|---------|-----------------|
| Architecture | Transformer-based | CNN-based |
| Detection paradigm | Query-based (like DETR) | Anchor-free points |
| NMS required | No (end-to-end) | Yes |
| Training | End-to-end differentiable | End-to-end |
| Inference speed | Real-time | Real-time |
| Memory usage | Higher | Lower |
| Small objects | Good | Excellent |
| Large objects | Excellent | Good |

**When to use RT-DETR:**

- Need end-to-end differentiable pipeline
- Want to avoid NMS post-processing
- Detecting large objects or complex scenes
- Have sufficient GPU memory
- Prefer transformer-based architecture

**When to use FCOS (ObjectDetector):**

- Need maximum efficiency
- Detecting small objects
- Limited GPU memory
- Want CNN-based architecture
- Need multi-scale detection (P3-P7)

**Example Integration:**

```python
from transformers import RTDetrForObjectDetection
from autotimm import DetectionDataModule

# Use AutoTimm for data loading
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=4,
)

# Use RT-DETR model from transformers library
model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd",
    num_labels=80,
)
```

See [RT-DETR Example](../../examples/tasks/object-detection.md#rt-detr-real-time-detection-transformer) for complete integration guide, model comparison tables, and when to use each architecture.

**Requirements:**
```bash
pip install transformers
```

## See Also

- [Object Detection Data](../data-loading/object-detection-data.md) - Data loading for detection
- [Object Detection Inference](../inference/object-detection-inference.md) - Making predictions
- [Image Classifier](image-classifier.md) - Classification models
- [Object Detection Examples](../../examples/tasks/object-detection.md) - Complete examples
