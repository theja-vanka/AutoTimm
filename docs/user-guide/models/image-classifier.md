# ImageClassifier

Image classification with any timm backbone and flexible training options.

## Basic Usage

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

## Backbone Selection

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

## Performance Optimization

### torch.compile (PyTorch 2.0+)

**Enabled by default** for automatic optimization:

```python
# Default: torch.compile enabled
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)
```

Disable if needed:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    compile_model=False,
)
```

Custom compile options:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    compile_kwargs={
        "mode": "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
        "fullgraph": True,           # Try to compile entire graph
        "dynamic": False,            # Dynamic shapes
    },
)
```

**Compile Modes:**

- `"default"` - Balanced performance (recommended)
- `"reduce-overhead"` - Lower latency, better for smaller batches or inference
- `"max-autotune"` - Maximum optimization with longer compile time

**What gets compiled:**

- Backbone network
- Classification head

**Note:** First training/inference run will be slower due to compilation overhead. Subsequent runs benefit from optimization. Falls back gracefully on PyTorch < 2.0.

## Multi-Label Classification

For tasks where each image can belong to multiple classes simultaneously (e.g., content tagging, attribute recognition, medical imaging), enable multi-label mode:

```python
from autotimm import ImageClassifier, MetricConfig

model = ImageClassifier(
    backbone="resnet50",
    num_classes=4,          # number of labels
    multi_label=True,       # switches to BCEWithLogitsLoss + sigmoid
    threshold=0.5,          # prediction threshold for sigmoid outputs
    metrics=[
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="MultilabelAccuracy",
            params={"num_labels": 4},
            stages=["train", "val"],
            prog_bar=True,
        ),
    ],
)
```

**What changes with `multi_label=True`:**

| Aspect | Single-label (default) | Multi-label |
|--------|----------------------|-------------|
| Loss | `CrossEntropyLoss` | `BCEWithLogitsLoss` |
| Predictions | `argmax` (one class) | `sigmoid > threshold` (multiple) |
| `predict_step` output | `softmax` (sums to 1) | `sigmoid` (independent per label) |
| Targets | Integer class indices | Multi-hot float vectors |
| Confusion matrix | Supported | Skipped |
| Label smoothing | Supported | Not supported (raises `ValueError`) |

### Multi-Label Data

Use `MultiLabelImageDataModule` with CSV files:

```python
from autotimm import MultiLabelImageDataModule

# CSV format:
#   image_path,cat,dog,outdoor,indoor
#   img1.jpg,1,0,1,0
#   img2.jpg,0,1,0,1

data = MultiLabelImageDataModule(
    train_csv="train.csv",
    image_dir="./images",
    val_csv="val.csv",
    image_size=224,
    batch_size=32,
)
```

### Multi-Label Metrics

Use `torchmetrics.classification.Multilabel*` metrics:

```python
from autotimm import MetricConfig

metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="MultilabelAccuracy",
        params={"num_labels": 4},
        stages=["train", "val"],
        prog_bar=True,
    ),
    MetricConfig(
        name="f1",
        backend="torchmetrics",
        metric_class="MultilabelF1Score",
        params={"num_labels": 4, "average": "macro"},
        stages=["val"],
    ),
]
```

Common multilabel metrics: `MultilabelAccuracy`, `MultilabelF1Score`, `MultilabelPrecision`, `MultilabelRecall`, `MultilabelAUROC`, `MultilabelHammingDistance`.

---

## Full Parameter Reference

```python
ImageClassifier(
    backbone="resnet50",           # Model name or BackboneConfig
    num_classes=10,                # Number of classes (or labels for multi-label)
    multi_label=False,             # Enable multi-label classification
    threshold=0.5,                 # Prediction threshold (multi-label only)
    metrics=metrics,               # MetricManager or list of MetricConfig
    logging_config=None,           # LoggingConfig for enhanced logging
    lr=1e-3,                       # Learning rate
    weight_decay=1e-4,             # Weight decay
    optimizer="adamw",             # Optimizer name or dict
    optimizer_kwargs=None,         # Extra optimizer kwargs
    scheduler="cosine",            # Scheduler name, dict, or None
    scheduler_kwargs=None,         # Extra scheduler kwargs
    head_dropout=0.0,              # Dropout before classifier
    label_smoothing=0.0,           # Label smoothing (not for multi-label)
    freeze_backbone=False,         # Freeze backbone weights
    mixup_alpha=0.0,               # Mixup augmentation alpha
    compile_model=True,            # Enable torch.compile (PyTorch 2.0+)
    compile_kwargs=None,           # Custom torch.compile options
)
```

## See Also

- [Image Classification Data](../data-loading/image-classification-data.md) - Data loading for classification
- [Classification Inference](../inference/classification-inference.md) - Making predictions
- [Object Detector](object-detector.md) - Object detection models
- [Classification Examples](../../examples/tasks/classification.md) - Complete examples
- [Backbone Utilities](../../examples/utilities/backbone-utilities.md) - Backbone discovery and utilities
