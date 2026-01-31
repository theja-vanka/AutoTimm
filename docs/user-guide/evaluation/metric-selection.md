# Metric Selection Guide

This guide helps you choose the right metrics for your computer vision task and configure them properly in AutoTimm.

## Quick Reference

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Classification (balanced) | Accuracy | Top-5 Accuracy |
| Classification (imbalanced) | F1 Score (macro) | Precision, Recall, AUROC |
| Binary Classification | AUROC | F1, Precision, Recall |
| Object Detection | mAP@[0.5:0.95] | mAP@0.5, Average Recall |
| Semantic Segmentation | mIoU | Pixel Accuracy, Dice |
| Instance Segmentation | mAP (bbox + mask) | AP per class |

---

## Classification Metrics

### Accuracy

Best for balanced datasets with equal class distribution.

```python
from autotimm import MetricConfig

accuracy = MetricConfig(
    name="accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={"task": "multiclass"},
    stages=["train", "val", "test"],
    prog_bar=True,
)
```

**When to Use:**
- Balanced class distribution
- All classes equally important
- Simple model comparison

**When NOT to Use:**
- Imbalanced datasets (accuracy can be misleading)
- When false positives/negatives have different costs

### Top-K Accuracy

Measures if correct class is in top K predictions. Useful for fine-grained classification.

```python
top5_accuracy = MetricConfig(
    name="top5_accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={"task": "multiclass", "top_k": 5},
    stages=["val", "test"],
)
```

**When to Use:**
- Many similar classes (ImageNet-style tasks)
- Fine-grained classification
- When near-misses are acceptable

### F1 Score

Harmonic mean of precision and recall. Better for imbalanced data.

```python
# Macro F1 (average across classes, equal weight to each class)
f1_macro = MetricConfig(
    name="f1_macro",
    backend="torchmetrics",
    metric_class="F1Score",
    params={"task": "multiclass", "average": "macro"},
    stages=["val", "test"],
    prog_bar=True,
)

# Weighted F1 (weighted by class support)
f1_weighted = MetricConfig(
    name="f1_weighted",
    backend="torchmetrics",
    metric_class="F1Score",
    params={"task": "multiclass", "average": "weighted"},
    stages=["val", "test"],
)

# Per-class F1 (for detailed analysis)
f1_per_class = MetricConfig(
    name="f1_per_class",
    backend="torchmetrics",
    metric_class="F1Score",
    params={"task": "multiclass", "average": "none"},
    stages=["test"],
)
```

**When to Use:**
- Imbalanced datasets
- When both precision and recall matter
- Medical diagnosis, fraud detection

### Precision and Recall

```python
precision = MetricConfig(
    name="precision",
    backend="torchmetrics",
    metric_class="Precision",
    params={"task": "multiclass", "average": "macro"},
    stages=["val", "test"],
)

recall = MetricConfig(
    name="recall",
    backend="torchmetrics",
    metric_class="Recall",
    params={"task": "multiclass", "average": "macro"},
    stages=["val", "test"],
)
```

**Precision** (minimize false positives): Use when false alarms are costly.
- Spam detection: Don't mark legitimate emails as spam
- Product recommendations: Don't recommend irrelevant items

**Recall** (minimize false negatives): Use when missing cases is costly.
- Disease detection: Don't miss any positive cases
- Security threats: Don't miss any actual threats

### AUROC (Area Under ROC Curve)

Threshold-independent metric for classification quality.

```python
# Binary classification
auroc_binary = MetricConfig(
    name="auroc",
    backend="torchmetrics",
    metric_class="AUROC",
    params={"task": "binary"},
    stages=["val", "test"],
)

# Multiclass classification
auroc_multiclass = MetricConfig(
    name="auroc",
    backend="torchmetrics",
    metric_class="AUROC",
    params={"task": "multiclass", "num_classes": 10, "average": "macro"},
    stages=["val", "test"],
)
```

**When to Use:**
- Binary classification
- When you need to compare models at different thresholds
- Medical diagnosis (sensitivity vs specificity trade-off)

---

## Detection Metrics

### Mean Average Precision (mAP)

The standard metric for object detection.

```python
map_metric = MetricConfig(
    name="mAP",
    backend="torchmetrics",
    metric_class="MeanAveragePrecision",
    params={
        "box_format": "xyxy",
        "iou_type": "bbox",
    },
    stages=["val", "test"],
    prog_bar=True,
)
```

**Parameters Explained:**

| Parameter | Options | Description |
|-----------|---------|-------------|
| `box_format` | "xyxy", "xywh", "cxcywh" | Bounding box format |
| `iou_type` | "bbox", "segm" | IoU type for matching |
| `iou_thresholds` | List[float] | Custom IoU thresholds |
| `class_metrics` | bool | Compute per-class metrics |

**mAP Variants:**

```python
# COCO-style mAP (average over IoU 0.5:0.95)
# This is the default behavior

# Pascal VOC mAP (single IoU threshold 0.5)
map_50 = MetricConfig(
    name="mAP50",
    backend="torchmetrics",
    metric_class="MeanAveragePrecision",
    params={
        "box_format": "xyxy",
        "iou_thresholds": [0.5],
    },
    stages=["val", "test"],
)

# Strict mAP at IoU 0.75
map_75 = MetricConfig(
    name="mAP75",
    backend="torchmetrics",
    metric_class="MeanAveragePrecision",
    params={
        "box_format": "xyxy",
        "iou_thresholds": [0.75],
    },
    stages=["val", "test"],
)
```

### Average Recall

Measures detection coverage at different numbers of detections.

```python
# Average Recall is computed automatically with MeanAveragePrecision
# Access via metric output keys: mar_1, mar_10, mar_100
```

**Interpretation:**
- `mar_1`: Recall with max 1 detection per image
- `mar_10`: Recall with max 10 detections per image
- `mar_100`: Recall with max 100 detections per image

---

## Segmentation Metrics

### IoU / Jaccard Index

Intersection over Union - the standard segmentation metric.

```python
iou = MetricConfig(
    name="iou",
    backend="torchmetrics",
    metric_class="JaccardIndex",
    params={
        "task": "multiclass",
        "num_classes": 21,
        "average": "macro",  # mIoU
    },
    stages=["val", "test"],
    prog_bar=True,
)

# Per-class IoU for detailed analysis
iou_per_class = MetricConfig(
    name="iou_per_class",
    backend="torchmetrics",
    metric_class="JaccardIndex",
    params={
        "task": "multiclass",
        "num_classes": 21,
        "average": "none",
    },
    stages=["test"],
)
```

**When to Use:**
- Semantic segmentation evaluation
- Standard benchmark comparison
- When region overlap matters

### Dice Coefficient

Similar to IoU but with different weighting. Common in medical imaging.

```python
dice = MetricConfig(
    name="dice",
    backend="torchmetrics",
    metric_class="Dice",
    params={
        "num_classes": 21,
        "average": "macro",
    },
    stages=["val", "test"],
)
```

**Dice vs IoU:**

| Metric | Formula | Range | Relationship |
|--------|---------|-------|--------------|
| IoU | TP / (TP + FP + FN) | [0, 1] | IoU = Dice / (2 - Dice) |
| Dice | 2*TP / (2*TP + FP + FN) | [0, 1] | Dice = 2*IoU / (1 + IoU) |

Dice is always >= IoU for the same prediction.

### Pixel Accuracy

Simple metric counting correctly classified pixels.

```python
pixel_accuracy = MetricConfig(
    name="pixel_accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={
        "task": "multiclass",
        "num_classes": 21,
    },
    stages=["val", "test"],
)
```

**Limitations:**
- Biased toward dominant classes
- Background often dominates, inflating scores
- Use mIoU for more balanced evaluation

---

## MetricConfig Deep Dive

### Full Parameter Reference

```python
MetricConfig(
    name="metric_name",           # Unique identifier for logging
    backend="torchmetrics",       # "torchmetrics" or "custom"
    metric_class="ClassName",     # Class name or full path
    params={},                    # Constructor parameters
    stages=["train", "val"],      # When to compute
    log_on_step=False,            # Log each step (default: False)
    log_on_epoch=True,            # Log each epoch (default: True)
    prog_bar=False,               # Show in progress bar (default: False)
)
```

### Stage Configuration

| Stage | When Computed | Typical Use |
|-------|---------------|-------------|
| `train` | Every training step/epoch | Monitor training progress |
| `val` | Every validation epoch | Model selection, early stopping |
| `test` | Final evaluation | Reporting final results |

**Recommendations:**
- `train`: Accuracy only (fast metrics)
- `val`: Primary metrics for checkpointing
- `test`: All metrics for comprehensive evaluation

### Logging Configuration

```python
# Metric computed and logged every step (useful for debugging)
fast_metric = MetricConfig(
    name="train_accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={"task": "multiclass"},
    stages=["train"],
    log_on_step=True,
    log_on_epoch=True,
    prog_bar=True,
)

# Metric computed every epoch only (recommended for expensive metrics)
slow_metric = MetricConfig(
    name="val_f1",
    backend="torchmetrics",
    metric_class="F1Score",
    params={"task": "multiclass", "average": "macro"},
    stages=["val"],
    log_on_step=False,
    log_on_epoch=True,
    prog_bar=True,
)
```

---

## Torchmetrics Integration

AutoTimm uses [torchmetrics](https://torchmetrics.readthedocs.io/) for metric computation.

### Available Metric Classes

**Classification:**
- `Accuracy` - Classification accuracy
- `F1Score` - F1 score
- `Precision` - Precision
- `Recall` - Recall
- `AUROC` - Area under ROC curve
- `AveragePrecision` - Average precision
- `ConfusionMatrix` - Confusion matrix
- `CohenKappa` - Cohen's kappa
- `MatthewsCorrCoef` - Matthews correlation

**Detection:**
- `MeanAveragePrecision` - mAP for detection

**Segmentation:**
- `JaccardIndex` - IoU / Jaccard index
- `Dice` - Dice coefficient

### Custom Metric Classes

You can use any torchmetrics class or create custom ones:

```python
# Using a custom torchmetrics class from your package
custom_metric = MetricConfig(
    name="custom_metric",
    backend="custom",
    metric_class="mypackage.metrics.CustomMetric",
    params={"threshold": 0.5},
    stages=["val"],
)
```

See [Advanced Customization](../training/advanced-customization.md) for creating custom metrics.

---

## Task-Specific Recommendations

### Image Classification

#### Balanced Dataset

```python
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
        name="top5_accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass", "top_k": 5},
        stages=["val", "test"],
    ),
]
```

#### Imbalanced Dataset

```python
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
        name="f1_macro",
        backend="torchmetrics",
        metric_class="F1Score",
        params={"task": "multiclass", "average": "macro"},
        stages=["val", "test"],
        prog_bar=True,
    ),
    MetricConfig(
        name="precision",
        backend="torchmetrics",
        metric_class="Precision",
        params={"task": "multiclass", "average": "macro"},
        stages=["test"],
    ),
    MetricConfig(
        name="recall",
        backend="torchmetrics",
        metric_class="Recall",
        params={"task": "multiclass", "average": "macro"},
        stages=["test"],
    ),
]
```

#### Binary Classification

```python
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "binary"},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
    MetricConfig(
        name="auroc",
        backend="torchmetrics",
        metric_class="AUROC",
        params={"task": "binary"},
        stages=["val", "test"],
        prog_bar=True,
    ),
    MetricConfig(
        name="f1",
        backend="torchmetrics",
        metric_class="F1Score",
        params={"task": "binary"},
        stages=["val", "test"],
    ),
]
```

### Object Detection

```python
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
```

### Semantic Segmentation

```python
metrics = [
    MetricConfig(
        name="iou",
        backend="torchmetrics",
        metric_class="JaccardIndex",
        params={"task": "multiclass", "num_classes": 21, "average": "macro"},
        stages=["val", "test"],
        prog_bar=True,
    ),
    MetricConfig(
        name="dice",
        backend="torchmetrics",
        metric_class="Dice",
        params={"num_classes": 21, "average": "macro"},
        stages=["val", "test"],
    ),
]
```

### Instance Segmentation

```python
metrics = [
    # Bounding box mAP
    MetricConfig(
        name="mAP_bbox",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "bbox"},
        stages=["val", "test"],
        prog_bar=True,
    ),
    # Mask mAP
    MetricConfig(
        name="mAP_segm",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "segm"},
        stages=["val", "test"],
    ),
]
```

---

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
    # Define comprehensive metrics for imbalanced classification
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
            name="f1_macro",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="f1_weighted",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "weighted"},
            stages=["test"],
        ),
        MetricConfig(
            name="precision",
            backend="torchmetrics",
            metric_class="Precision",
            params={"task": "multiclass", "average": "macro"},
            stages=["test"],
        ),
        MetricConfig(
            name="recall",
            backend="torchmetrics",
            metric_class="Recall",
            params={"task": "multiclass", "average": "macro"},
            stages=["test"],
        ),
    ]

    # Create MetricManager
    manager = MetricManager(configs=metric_configs, num_classes=10)

    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
    )

    # Model
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        metrics=manager,
    )

    # Trainer - use F1 for checkpointing (better for imbalanced data)
    trainer = AutoTrainer(
        max_epochs=50,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
        checkpoint_monitor="val/f1_macro",
        checkpoint_mode="max",
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## See Also

- [Metrics Guide](metrics.md) - MetricConfig and MetricManager usage
- [Training Guide](../training/training.md) - Training configuration
- [Advanced Customization](../training/advanced-customization.md) - Custom metrics
