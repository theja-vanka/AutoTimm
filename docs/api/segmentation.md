# Segmentation API Reference

Complete API documentation for semantic and instance segmentation tasks.

## Task Models

### SemanticSegmentor

End-to-end semantic segmentation model with timm backbones, supporting multiple head architectures (DeepLabV3+, FCN) and loss functions (CE, Dice, Focal, Combined).

::: autotimm.SemanticSegmentor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward
        - predict
        - training_step
        - validation_step
        - test_step
        - predict_step
        - configure_optimizers

### InstanceSegmentor

End-to-end instance segmentation model combining FCOS-style object detection with per-instance mask prediction. Integrates timm backbones with FPN and dual-head architecture for boxes and masks.

::: autotimm.InstanceSegmentor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward
        - predict
        - training_step
        - validation_step
        - test_step
        - predict_step
        - configure_optimizers

## Data Modules

### SegmentationDataModule

PyTorch Lightning DataModule for semantic segmentation. Supports multiple dataset formats including PNG masks, Cityscapes, COCO, and Pascal VOC. Provides flexible augmentation presets and custom transform support.

::: autotimm.SegmentationDataModule
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader

### InstanceSegmentationDataModule

PyTorch Lightning DataModule for instance segmentation using COCO format. Handles both detection boxes and instance masks with built-in augmentation support.

::: autotimm.InstanceSegmentationDataModule
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader

## Segmentation Heads

### DeepLabV3PlusHead

DeepLabV3+ segmentation head with Atrous Spatial Pyramid Pooling (ASPP) and decoder with skip connections. Provides multi-scale context aggregation for accurate semantic segmentation.

::: autotimm.head.DeepLabV3PlusHead
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

### FCNHead

Fully Convolutional Network (FCN) head for semantic segmentation. Simple upsampling-based architecture suitable for baseline models.

::: autotimm.head.FCNHead
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

### MaskHead

ROI-based mask prediction head for instance segmentation. Predicts per-instance binary masks from ROI-aligned features.

::: autotimm.head.MaskHead
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

### ASPP

Atrous Spatial Pyramid Pooling module for multi-scale feature extraction. Core component of DeepLabV3+ architecture.

::: autotimm.head.ASPP
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

## Segmentation Losses

### DiceLoss

Dice loss for multi-class semantic segmentation. Optimizes directly for IoU-like metric. Formula: `1 - (2 * |X ∩ Y|) / (|X| + |Y|)`. Effective for handling class imbalance.

::: autotimm.loss.DiceLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

### CombinedSegmentationLoss

Combines Cross-Entropy and Dice losses with configurable weights. Leverages pixel-wise classification (CE) and region overlap optimization (Dice) for robust segmentation.

::: autotimm.loss.CombinedSegmentationLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

### FocalLossPixelwise

Focal loss for dense pixel-wise prediction. Down-weights easy examples to focus on hard pixels. Particularly effective for handling severe class imbalance in segmentation tasks.

::: autotimm.loss.FocalLossPixelwise
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

### TverskyLoss

Generalized Dice loss with configurable false positive/negative trade-off via alpha and beta parameters. Useful for highly imbalanced segmentation tasks.

::: autotimm.loss.TverskyLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

### MaskLoss

Binary cross-entropy loss for instance segmentation masks. Used in conjunction with detection losses for end-to-end instance segmentation training.

::: autotimm.loss.MaskLoss
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - forward

## Import Styles

AutoTimm supports multiple import styles for convenience:

### Direct Imports

```python
from autotimm import (
    SemanticSegmentor,
    InstanceSegmentor,
    SegmentationDataModule,
    DiceLoss,
    DeepLabV3PlusHead,
)
```

### Submodule Aliases

```python
# Use singular form aliases
from autotimm.task import SemanticSegmentor, InstanceSegmentor
from autotimm.loss import DiceLoss, CombinedSegmentationLoss
from autotimm.head import DeepLabV3PlusHead, MaskHead
from autotimm.metric import MetricConfig
```

### Original Imports

```python
# Original plural form still works
from autotimm.tasks import SemanticSegmentor
from autotimm.losses import DiceLoss
from autotimm.heads import DeepLabV3PlusHead
from autotimm.metrics import MetricConfig
```

### Namespace Access

```python
import autotimm

# Access via submodule aliases
model = autotimm.task.SemanticSegmentor(...)
loss = autotimm.loss.DiceLoss(...)
head = autotimm.head.DeepLabV3PlusHead(...)
```

## Parameters Reference

### SemanticSegmentor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone` | str \| FeatureBackboneConfig | Required | Timm model name or config |
| `num_classes` | int | Required | Number of segmentation classes |
| `head_type` | str | "deeplabv3plus" | Head architecture ("deeplabv3plus" or "fcn") |
| `loss_type` | str | "combined" | Loss function type |
| `dice_weight` | float | 1.0 | Weight for Dice loss in combined mode |
| `ce_weight` | float | 1.0 | Weight for CE loss in combined mode |
| `ignore_index` | int | 255 | Index to ignore in loss computation |
| `metrics` | list[MetricConfig] | None | Metric configurations |
| `lr` | float | 1e-4 | Learning rate |
| `weight_decay` | float | 1e-4 | Weight decay |
| `optimizer` | str \| dict | "adamw" | Optimizer name or config |
| `scheduler` | str \| dict | "cosine" | Scheduler name or config |
| `freeze_backbone` | bool | False | Whether to freeze backbone parameters |

### InstanceSegmentor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone` | str \| FeatureBackboneConfig | Required | Timm model name or config |
| `num_classes` | int | Required | Number of object classes (excluding background) |
| `fpn_channels` | int | 256 | Number of FPN channels |
| `head_num_convs` | int | 4 | Number of conv layers in detection head |
| `mask_size` | int | 28 | ROI mask resolution |
| `roi_pool_size` | int | 14 | ROI pooling output size |
| `mask_loss_weight` | float | 1.0 | Weight for mask loss |
| `focal_alpha` | float | 0.25 | Alpha parameter for focal loss |
| `focal_gamma` | float | 2.0 | Gamma parameter for focal loss |
| `cls_loss_weight` | float | 1.0 | Weight for classification loss |
| `reg_loss_weight` | float | 1.0 | Weight for regression loss |
| `centerness_loss_weight` | float | 1.0 | Weight for centerness loss |
| `score_thresh` | float | 0.05 | Score threshold for detections |
| `nms_thresh` | float | 0.5 | IoU threshold for NMS |
| `max_detections_per_image` | int | 100 | Maximum detections per image |
| `mask_threshold` | float | 0.5 | Threshold for binarizing predicted masks |
| `metrics` | list[MetricConfig] | None | Metric configurations |
| `logging_config` | LoggingConfig | None | Enhanced logging configuration |
| `lr` | float | 1e-4 | Learning rate |
| `weight_decay` | float | 1e-4 | Weight decay |
| `optimizer` | str \| dict | "adamw" | Optimizer name or config |
| `scheduler` | str \| dict | "cosine" | Scheduler name or config |
| `freeze_backbone` | bool | False | Whether to freeze backbone parameters |

### SegmentationDataModule Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str \| Path | Required | Root directory of dataset |
| `format` | str | "png" | Dataset format ("png", "coco", "cityscapes", "voc") |
| `image_size` | int | 512 | Target image size (square) |
| `batch_size` | int | 8 | Batch size |
| `num_workers` | int | 4 | Number of dataloader workers |
| `augmentation_preset` | str | "default" | Augmentation preset ("default", "strong", "light") |
| `custom_train_transforms` | Any | None | Custom training transforms (overrides preset) |
| `custom_val_transforms` | Any | None | Custom validation transforms |
| `class_mapping` | dict | None | Mapping from dataset class IDs to contiguous IDs |
| `ignore_index` | int | 255 | Index for ignored pixels (e.g., boundaries) |

### InstanceSegmentationDataModule Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str \| Path | Required | Root directory of COCO dataset |
| `image_size` | int | 640 | Target image size (square) |
| `batch_size` | int | 4 | Batch size |
| `num_workers` | int | 4 | Number of dataloader workers |
| `augmentation_preset` | str | "default" | Augmentation strength ("default", "strong", "light") |
| `custom_train_transforms` | Any | None | Custom training transforms (overrides preset) |
| `custom_val_transforms` | Any | None | Custom validation transforms |
| `min_keypoints` | int | 0 | Minimum keypoints for valid instance |
| `min_area` | float | 0.0 | Minimum area for valid instance |

## Examples

### Basic Semantic Segmentation

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig

# Setup data
data = SegmentationDataModule(
    data_dir="./data",
    format="png",
    image_size=512,
    batch_size=8,
)

# Create model
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=21,
    head_type="deeplabv3plus",
    loss_type="combined",
    ce_weight=1.0,
    dice_weight=1.0,
    metrics=[
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={"task": "multiclass", "num_classes": 21, "average": "macro"},
            stages=["val"],
            prog_bar=True,
        )
    ],
)
```

### Instance Segmentation

```python
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule, MetricConfig

# Setup data
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=4,
)

# Create model
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    mask_loss_weight=1.0,
    mask_size=28,
    roi_pool_size=14,
    metrics=[
        MetricConfig(
            name="mask_mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "segm"},
            stages=["val"],
            prog_bar=True,
        )
    ],
)
```

### Using Import Aliases

```python
from autotimm.task import SemanticSegmentor, InstanceSegmentor
from autotimm.loss import DiceLoss, CombinedSegmentationLoss, MaskLoss
from autotimm.head import DeepLabV3PlusHead, MaskHead
from autotimm.metric import MetricConfig

# Create model using aliases
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    loss_type="combined",
)

# Directly instantiate losses if needed
dice_loss = DiceLoss(num_classes=19, ignore_index=255)
combined_loss = CombinedSegmentationLoss(
    num_classes=19,
    ce_weight=1.0,
    dice_weight=1.0,
)
mask_loss = MaskLoss()
```

### Custom Loss Configuration

```python
from autotimm.loss import TverskyLoss

# Create custom Tversky loss for handling class imbalance
loss_fn = TverskyLoss(
    num_classes=19,
    alpha=0.3,  # Lower alpha emphasizes recall
    beta=0.7,   # Higher beta penalizes false negatives more
    ignore_index=255,
)

# Note: Use loss_type parameter in model for built-in losses
# For custom losses, you would need to subclass and override the loss
```

### Advanced Configuration

```python
from autotimm import SemanticSegmentor, LoggingConfig

# Model with enhanced logging
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    loss_type="combined",
    ce_weight=1.0,
    dice_weight=1.0,
    logging_config=LoggingConfig(
        log_learning_rate=True,
        log_gradient_norm=True,
        log_weight_norm=True,
    ),
    freeze_backbone=False,  # Set True to freeze backbone
)
```

## See Also

- [Semantic Segmentation Examples](../examples/semantic-segmentation.md)
- [Instance Segmentation Examples](../examples/instance-segmentation.md)
- [Training Guide](../user-guide/training.md)
- [Metrics Guide](../user-guide/metrics.md)
