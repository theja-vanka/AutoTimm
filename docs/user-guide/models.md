# Models

AutoTimm provides two main model types:
- **ImageClassifier**: Image classification with any timm backbone
- **ObjectDetector**: FCOS-style anchor-free object detection with timm backbones

## ImageClassifier

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

### Backbone Selection

AutoTimm supports 1000+ backbones from timm. Browse available models:

```python
import autotimm

# List all backbones
all_models = autotimm.list_backbones()
print(f"Total models: {len(all_models)}")

# Search by pattern
autotimm.list_backbones("*resnet*")
autotimm.list_backbones("*efficientnet*")
autotimm.list_backbones("*vit*")

# Only pretrained models
autotimm.list_backbones("*convnext*", pretrained_only=True)
```

Popular backbone families:

| Family | Examples | Use Case |
|--------|----------|----------|
| ResNet | `resnet18`, `resnet50`, `resnet101` | General purpose |
| EfficientNet | `efficientnet_b0` to `efficientnet_b7` | Efficiency |
| ConvNeXt | `convnext_tiny`, `convnext_base` | Modern CNN |
| ViT | `vit_base_patch16_224`, `vit_large_patch16_224` | Transformers |
| Swin | `swin_tiny_patch4_window7_224` | Hierarchical ViT |
| DeiT | `deit_base_patch16_224` | Data-efficient ViT |

## BackboneConfig

For advanced backbone configuration:

```python
from autotimm import BackboneConfig, ImageClassifier

cfg = BackboneConfig(
    model_name="vit_base_patch16_224",
    pretrained=True,           # Load pretrained weights
    drop_rate=0.1,             # Dropout rate
    drop_path_rate=0.1,        # Stochastic depth
)

model = ImageClassifier(
    backbone=cfg,
    num_classes=100,
    metrics=metrics,
)
```

### BackboneConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"resnet50"` | timm model name |
| `pretrained` | `True` | Load pretrained weights |
| `num_classes` | `0` | Set to 0 for feature extractor |
| `drop_rate` | `0.0` | Dropout rate |
| `drop_path_rate` | `0.0` | Stochastic depth rate |
| `extra_kwargs` | `{}` | Additional timm.create_model kwargs |

## Backbone Utilities

### Inspect Backbone

```python
import autotimm

backbone = autotimm.create_backbone("resnet50")
print(f"Output features: {backbone.num_features}")
print(f"Parameters: {autotimm.count_parameters(backbone):,}")
```

### Count Parameters

```python
import autotimm

model = ImageClassifier(backbone="resnet50", num_classes=10, metrics=metrics)

# Trainable parameters only
trainable = autotimm.count_parameters(model)

# All parameters
total = autotimm.count_parameters(model, trainable_only=False)

print(f"Trainable: {trainable:,}")
print(f"Total: {total:,}")
```

## Model Configuration

### Optimizer

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    lr=1e-3,
    weight_decay=1e-4,
    optimizer="adamw",  # "adamw", "adam", "sgd", "rmsprop"
)
```

Available optimizers:

**Torch:**
- `adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`

**Timm (if installed):**
- `adamp`, `sgdp`, `adabelief`, `radam`, `lamb`, `lars`, `madgrad`, `novograd`

Custom optimizer:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    optimizer={
        "class": "torch.optim.AdamW",
        "params": {"betas": (0.9, 0.999)},
    },
)
```

### Scheduler

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    scheduler="cosine",  # "cosine", "step", "onecycle", "none"
)
```

Available schedulers:

**Torch:**
- `cosine`, `step`, `multistep`, `exponential`, `onecycle`, `plateau`

**Timm:**
- `cosine_with_restarts`

Custom scheduler:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    scheduler={
        "class": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params": {"T_0": 10, "T_mult": 2},
    },
)
```

### Label Smoothing

Regularization technique for better generalization:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    label_smoothing=0.1,
)
```

### Mixup Augmentation

Data augmentation that mixes samples:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    mixup_alpha=0.2,
)
```

### Head Dropout

Dropout before the classification layer:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    head_dropout=0.5,
)
```

## Freeze Backbone

For transfer learning / linear probing:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    freeze_backbone=True,  # Only train classification head
    lr=1e-2,               # Higher LR for head-only training
)
```

## Two-Phase Fine-Tuning

```python
# Phase 1: Linear probe (frozen backbone)
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

model._lr = 1e-4  # Lower LR for fine-tuning
trainer = AutoTrainer(max_epochs=20, gradient_clip_val=1.0)
trainer.fit(model, datamodule=data)
```

## Full Parameter Reference

```python
ImageClassifier(
    backbone="resnet50",           # Model name or BackboneConfig
    num_classes=10,                # Number of classes
    metrics=metrics,               # MetricManager or list of MetricConfig
    logging_config=None,           # LoggingConfig for enhanced logging
    lr=1e-3,                       # Learning rate
    weight_decay=1e-4,             # Weight decay
    optimizer="adamw",             # Optimizer name or dict
    optimizer_kwargs=None,         # Extra optimizer kwargs
    scheduler="cosine",            # Scheduler name, dict, or None
    scheduler_kwargs=None,         # Extra scheduler kwargs
    head_dropout=0.0,              # Dropout before classifier
    label_smoothing=0.0,           # Label smoothing factor
    freeze_backbone=False,         # Freeze backbone weights
    mixup_alpha=0.0,               # Mixup augmentation alpha
)
```

---

## ObjectDetector

FCOS-style anchor-free object detector that combines timm backbones with Feature Pyramid Networks (FPN) and detection heads.

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
    num_classes=80,  # COCO has 80 classes
    metrics=metrics,
    lr=1e-4,
)
```

### Architecture

The ObjectDetector uses the FCOS (Fully Convolutional One-Stage) architecture:

1. **Backbone**: Any timm model extracts multi-scale features
2. **FPN (Feature Pyramid Network)**: Builds pyramid levels P3-P7
3. **Detection Head**: Predicts class, bounding box, and centerness for each location
4. **NMS**: Non-Maximum Suppression filters overlapping detections

### Backbone Selection

Any timm backbone works for object detection. Popular choices:

**CNN Backbones:**

| Backbone | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| `resnet18` | Fast | Good | Quick experiments |
| `resnet50` | Medium | Better | Standard baseline |
| `efficientnet_b3` | Medium | Better | Efficiency |
| `convnext_tiny` | Medium | Best | Modern CNN |
| `resnet101` | Slow | Best | High accuracy |

**Transformer Backbones:**

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

See [Transformer-Based Detection Example](../examples/index.md#transformer-based-object-detection) for detailed usage.

### FPN Configuration

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

### Loss Configuration

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

### Inference Configuration

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

### Optimizer and Scheduler

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

### Freeze Backbone

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

### Advanced: Custom Regression Ranges

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

### Full Parameter Reference

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
)
```

### Usage Example

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

See [RT-DETR Example](../examples/index.md#rt-detr-real-time-detection-transformer) for complete integration guide.

**Available RT-DETR Models:**

| Model | Parameters | Best For |
|-------|------------|----------|
| `PekingU/rtdetr_r18vd` | 20M | Quick experiments |
| `PekingU/rtdetr_r34vd` | 31M | Balanced performance |
| `PekingU/rtdetr_r50vd` | 42M | Recommended (best balance) |
| `PekingU/rtdetr_r101vd` | 76M | Maximum accuracy |

**Requirements:**
```bash
pip install transformers
```

### Summary: Choosing a Detection Architecture

**Use FCOS (ObjectDetector) when:**
- Starting with object detection
- Want seamless AutoTimm integration
- Need efficient small object detection
- Limited computational resources
- Want to experiment with 1000+ timm backbones

**Use Transformer Backbones (with ObjectDetector) when:**
- Want to leverage vision transformers
- Have sufficient GPU memory
- Prefer hierarchical transformer features (Swin)
- Want modern CNN alternatives

**Use RT-DETR when:**
- Need end-to-end transformer detection
- Want to eliminate NMS
- Have high memory budget
- Detecting primarily large objects
- Want query-based detection paradigm

All approaches support COCO format datasets and can use AutoTimm's `DetectionDataModule` for efficient data loading.
