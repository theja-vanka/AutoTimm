# Metric Calculation Issues

Problems with metric configuration and values.

## Unexpected Metric Values

**Problem:** Metrics return 0, NaN, or unexpected values

**Solutions:**

```python
from autotimm import MetricConfig

# 1. Verify metric configuration matches task
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={
            "task": "multiclass",  # Must match: binary, multiclass, multilabel
            "num_classes": 10,
        },
        stages=["train", "val"],
    )
]

# 2. Check prediction format
def debug_predictions(model, batch):
    outputs = model(batch["images"])
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Label shape: {batch['labels'].shape}")
    print(f"Label range: [{batch['labels'].min()}, {batch['labels'].max()}]")

# 3. Verify label encoding
# For classification: labels should be integers 0 to num_classes-1
# For detection: check bbox format (xyxy, xywh, cxcywh)
```

## Detection mAP Issues

```python
# Common issue: bbox format mismatch
from autotimm import ObjectDetector

# Specify bbox format explicitly
model = ObjectDetector(
    backbone="resnet50",
    num_classes=10,
    bbox_format="xyxy",  # Options: xyxy, xywh, cxcywh
)

# Verify annotation format
# COCO format uses [x, y, width, height]
# Model expects format specified in bbox_format parameter
```

## Related Issues

- [Data Loading](../data/data-loading.md) - Data format issues
- [Convergence](../training/convergence.md) - Training issues affecting metrics
