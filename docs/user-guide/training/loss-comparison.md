# Loss Function Comparison

AutoTimm provides specialized loss functions for object detection and segmentation tasks. This guide compares available losses and helps you choose the right one for your use case.

## Detection Losses

AutoTimm implements FCOS-style detection losses optimized for anchor-free object detection.

### FocalLoss

Focal Loss addresses class imbalance by down-weighting well-classified examples and focusing on hard negatives.

```python
from autotimm import FocalLoss

loss_fn = FocalLoss(
    alpha=0.25,       # Weight for positive examples
    gamma=2.0,        # Focusing parameter (higher = more focus on hard examples)
    reduction="mean", # "mean", "sum", or "none"
)

# Usage
loss = loss_fn(predictions, targets)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.25 | Weighting factor for positive examples. Higher values give more weight to positives. |
| `gamma` | float | 2.0 | Focusing parameter. Higher values increase focus on hard examples. |
| `reduction` | str | "mean" | Reduction method: "none", "mean", or "sum" |

**When to Use:**

- Class imbalanced detection datasets
- When background overwhelms foreground examples
- Standard choice for anchor-free detectors

**Tuning Guidelines:**

- `gamma=2.0`: Standard for most cases
- `gamma=1.5`: Less aggressive focusing (for balanced datasets)
- `gamma=3.0`: More aggressive (for severe imbalance)
- `alpha=0.25`: Standard for detection
- `alpha=0.5`: Equal weighting

---

### GIoULoss

Generalized IoU Loss for bounding box regression. Provides better gradients than standard IoU loss when boxes don't overlap.

```python
from autotimm import GIoULoss

loss_fn = GIoULoss(
    reduction="mean",
    eps=1e-7,
)

# Usage: boxes in (x1, y1, x2, y2) format
loss = loss_fn(pred_boxes, target_boxes)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduction` | str | "mean" | Reduction method: "none", "mean", or "sum" |
| `eps` | float | 1e-7 | Small value for numerical stability |

**When to Use:**

- Bounding box regression in detection tasks
- When boxes may not overlap (early training)
- Better convergence than L1/L2 losses for boxes

**Comparison with Other Box Losses:**

| Loss | Handles Non-Overlap | Scale Invariant | Typical Use |
|------|--------------------| ----------------|-------------|
| L1/L2 Loss | Yes | No | Fast, simple |
| IoU Loss | No | Yes | Overlapping boxes |
| GIoU Loss | Yes | Yes | General detection |
| DIoU Loss | Yes | Yes | Center-aware |
| CIoU Loss | Yes | Yes | Aspect ratio-aware |

---

### CenternessLoss

Binary cross-entropy loss for FCOS centerness prediction. Centerness helps suppress low-quality predictions far from object centers.

```python
from autotimm import CenternessLoss

loss_fn = CenternessLoss(reduction="mean")

# Usage: centerness values in [0, 1]
loss = loss_fn(pred_centerness, target_centerness)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduction` | str | "mean" | Reduction method: "none", "mean", or "sum" |

**When to Use:**

- FCOS-style anchor-free detection
- Quality prediction for bounding boxes
- Combined with classification and regression losses

---

### FCOSLoss

Combined FCOS loss that integrates classification, regression, and centerness losses.

```python
from autotimm import FCOSLoss

loss_fn = FCOSLoss(
    num_classes=80,
    focal_alpha=0.25,
    focal_gamma=2.0,
    cls_weight=1.0,
    reg_weight=1.0,
    centerness_weight=1.0,
)

# Usage
losses = loss_fn(
    cls_preds, reg_preds, centerness_preds,
    cls_targets, reg_targets, centerness_targets,
)
# Returns: {"cls_loss", "reg_loss", "centerness_loss", "total_loss"}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | Required | Number of object classes (excluding background) |
| `focal_alpha` | float | 0.25 | Alpha for focal loss |
| `focal_gamma` | float | 2.0 | Gamma for focal loss |
| `cls_weight` | float | 1.0 | Weight for classification loss |
| `reg_weight` | float | 1.0 | Weight for regression loss |
| `centerness_weight` | float | 1.0 | Weight for centerness loss |

**When to Use:**

- Complete FCOS training pipeline
- Anchor-free object detection
- Multi-scale detection

**Tuning Guidelines:**

- Default weights (1:1:1) work well for most cases
- Increase `reg_weight` if localization is poor
- Increase `cls_weight` if classification accuracy is low

---

## Detection Loss Comparison

| Loss | Purpose | Pros | Cons | Best For |
|------|---------|------|------|----------|
| **FocalLoss** | Classification | Handles imbalance, proven performance | Sensitive to alpha/gamma | Class-imbalanced datasets |
| **GIoULoss** | Box regression | Scale invariant, handles non-overlap | Slightly slower than L1 | General detection |
| **CenternessLoss** | Quality prediction | Improves NMS, filters poor boxes | Requires centerness targets | FCOS-style detectors |
| **FCOSLoss** | Combined | All-in-one, balanced training | Fixed architecture | Complete FCOS training |

---

## Segmentation Losses

AutoTimm provides losses for both semantic and instance segmentation tasks.

### DiceLoss

Dice Loss measures overlap between predicted and ground truth masks. Effective for imbalanced segmentation.

```python
from autotimm import DiceLoss

loss_fn = DiceLoss(
    num_classes=21,
    smooth=1.0,
    ignore_index=255,
    reduction="mean",
)

# Usage
loss = loss_fn(logits, targets)  # [B, C, H, W], [B, H, W]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | Required | Number of segmentation classes |
| `smooth` | float | 1.0 | Smoothing constant to avoid division by zero |
| `ignore_index` | int | 255 | Index to ignore in loss computation |
| `reduction` | str | "mean" | Reduction method: "none", "mean", or "sum" |

**When to Use:**

- Class-imbalanced segmentation
- Small object segmentation
- Medical image segmentation

**Tuning Guidelines:**

- `smooth=1.0`: Standard, prevents NaN for empty masks
- `smooth=0.01`: Less smoothing, sharper gradients
- `ignore_index=255`: Standard for Pascal VOC/Cityscapes

---

### FocalLossPixelwise

Pixel-wise focal loss for dense prediction. Handles class imbalance at the pixel level.

```python
from autotimm import FocalLossPixelwise

loss_fn = FocalLossPixelwise(
    alpha=0.25,
    gamma=2.0,
    ignore_index=255,
    reduction="mean",
)

# Usage
loss = loss_fn(logits, targets)  # [B, C, H, W], [B, H, W]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.25 | Weighting factor for positive/rare classes |
| `gamma` | float | 2.0 | Focusing parameter |
| `ignore_index` | int | 255 | Index to ignore in loss computation |
| `reduction` | str | "mean" | Reduction method |

**When to Use:**

- Severely imbalanced pixel distributions
- When some classes are much rarer than others
- Scene parsing with many small objects

---

### TverskyLoss

Generalization of Dice loss with separate control over false positives and false negatives.

```python
from autotimm import TverskyLoss

loss_fn = TverskyLoss(
    num_classes=21,
    alpha=0.5,  # Weight for false positives
    beta=0.5,   # Weight for false negatives
    smooth=1.0,
    ignore_index=255,
    reduction="mean",
)

# Usage
loss = loss_fn(logits, targets)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | Required | Number of segmentation classes |
| `alpha` | float | 0.5 | Weight for false positives |
| `beta` | float | 0.5 | Weight for false negatives |
| `smooth` | float | 1.0 | Smoothing constant |
| `ignore_index` | int | 255 | Index to ignore |
| `reduction` | str | "mean" | Reduction method |

**When to Use:**

- Control precision vs recall trade-off
- Medical imaging (minimize false negatives)
- Small object detection (minimize false positives)

**Tuning Guidelines:**

| Setting | alpha | beta | Effect |
|---------|-------|------|--------|
| Dice Loss (equal) | 0.5 | 0.5 | Balanced |
| Precision focus | 0.7 | 0.3 | Penalize false positives more |
| Recall focus | 0.3 | 0.7 | Penalize false negatives more |

---

### MaskLoss

Binary cross-entropy loss for instance segmentation masks.

```python
from autotimm import MaskLoss

loss_fn = MaskLoss(
    reduction="mean",
    pos_weight=1.0,  # Weight for positive pixels
)

# Usage: per-instance binary masks
loss = loss_fn(pred_masks, target_masks)  # [N, H, W], [N, H, W]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduction` | str | "mean" | Reduction method |
| `pos_weight` | float | None | Weight for positive (foreground) pixels |

**When to Use:**

- Instance segmentation tasks
- Per-object binary mask prediction
- Mask R-CNN style architectures

---

### CombinedSegmentationLoss

Combines cross-entropy and Dice loss for robust semantic segmentation training.

```python
from autotimm import CombinedSegmentationLoss
import torch

loss_fn = CombinedSegmentationLoss(
    num_classes=21,
    ce_weight=1.0,
    dice_weight=1.0,
    ignore_index=255,
    class_weights=None,  # Optional per-class weights
)

# With class weights for imbalanced data
class_weights = torch.tensor([1.0, 2.0, 2.0, ...])  # One per class
loss_fn = CombinedSegmentationLoss(
    num_classes=21,
    ce_weight=1.0,
    dice_weight=1.0,
    class_weights=class_weights,
)

# Usage
loss = loss_fn(logits, targets)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | Required | Number of segmentation classes |
| `ce_weight` | float | 1.0 | Weight for cross-entropy loss |
| `dice_weight` | float | 1.0 | Weight for Dice loss |
| `ignore_index` | int | 255 | Index to ignore |
| `class_weights` | Tensor | None | Per-class weights for CE loss |

**When to Use:**

- Default choice for semantic segmentation
- Combines pixel-wise accuracy (CE) with region overlap (Dice)
- Robust to class imbalance

**Tuning Guidelines:**

- Start with equal weights (1:1)
- Increase `dice_weight` for better IoU scores
- Increase `ce_weight` for better pixel accuracy
- Use `class_weights` for severely imbalanced datasets

---

## Segmentation Loss Comparison

| Loss | Handles Imbalance | Best For | Typical Use Case |
|------|------------------|----------|------------------|
| **CrossEntropy** | No (without weights) | Balanced datasets | General segmentation |
| **DiceLoss** | Yes | Region overlap, small objects | Medical imaging |
| **FocalLossPixelwise** | Yes | Severe pixel imbalance | Scene parsing |
| **TverskyLoss** | Yes | Precision/recall control | Domain-specific needs |
| **MaskLoss** | Optional (pos_weight) | Binary masks | Instance segmentation |
| **CombinedSegmentationLoss** | Yes | General semantic segmentation | Default choice |

---

## Task-Specific Recommendations

### Image Classification

Use `CrossEntropyLoss` (built into task classes):

```python
from autotimm import ImageClassifier

# CrossEntropy is used automatically
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)
```

### Object Detection

```python
from autotimm import ObjectDetector

# FCOSLoss is used automatically with these defaults
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    # Loss weights can be adjusted via model parameters if needed
)
```

### Semantic Segmentation

```python
from autotimm import SemanticSegmentor, CombinedSegmentationLoss

# Default: Combined CE + Dice
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=21,
    metrics=metrics,
)

# Custom loss configuration
loss_fn = CombinedSegmentationLoss(
    num_classes=21,
    ce_weight=0.5,
    dice_weight=1.5,  # Emphasize Dice for better IoU
)
```

### Instance Segmentation

```python
from autotimm import InstanceSegmentor

# Uses FCOSLoss for detection + MaskLoss for masks
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
)
```

---

## Complete Example: Custom Loss Configuration

```python
from autotimm import (
    AutoTrainer,
    CombinedSegmentationLoss,
    LoggerConfig,
    MetricConfig,
    SegmentationDataModule,
    SemanticSegmentor,
)
import torch


def main():
    # Data
    data = SegmentationDataModule(
        data_dir="./cityscapes",
        image_size=512,
        batch_size=8,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={"task": "multiclass", "num_classes": 19},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Class weights for Cityscapes (example: road and building are more common)
    class_weights = torch.ones(19)
    class_weights[0] = 0.5   # road (common)
    class_weights[1] = 0.5   # sidewalk (common)
    class_weights[11] = 2.0  # person (less common)
    class_weights[12] = 2.0  # rider (rare)

    # Custom loss
    loss_fn = CombinedSegmentationLoss(
        num_classes=19,
        ce_weight=1.0,
        dice_weight=1.5,
        class_weights=class_weights,
    )

    # Model
    model = SemanticSegmentor(
        backbone="resnet50",
        num_classes=19,
        metrics=metrics,
        lr=1e-4,
    )

    # Trainer
    trainer = AutoTrainer(
        max_epochs=50,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
        checkpoint_monitor="val/iou",
        checkpoint_mode="max",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## See Also

- [Training Guide](training.md) - Complete training documentation
- [Metric Selection](../evaluation/metric-selection.md) - Choosing metrics for your task
- [API Reference: Losses](../../api/losses.md) - Full API documentation
