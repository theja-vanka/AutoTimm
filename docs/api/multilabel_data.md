# MultiLabelImageDataModule

Lightning data module for multi-label image classification from CSV files.

## Overview

`MultiLabelImageDataModule` loads multi-label datasets where each image can belong to multiple classes. It reads CSV files with binary label columns and pairs with `ImageClassifier(multi_label=True)`.

Also provides `MultiLabelImageDataset` for custom data loading.

## API Reference

::: autotimm.MultiLabelImageDataModule
    options:
      show_source: true
      members:
        - __init__
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader
        - summary
        - num_labels
        - label_names

## Usage Examples

### Basic Usage

```python
from autotimm import MultiLabelImageDataModule

data = MultiLabelImageDataModule(
    train_csv="train.csv",
    image_dir="./images",
    val_csv="val.csv",
    image_size=224,
    batch_size=32,
)
data.setup("fit")
print(f"Labels: {data.num_labels}")
print(f"Label names: {data.label_names}")
```

### With Auto Validation Split

```python
data = MultiLabelImageDataModule(
    train_csv="train.csv",
    image_dir="./images",
    val_split=0.2,
)
```

### With Albumentations

```python
data = MultiLabelImageDataModule(
    train_csv="train.csv",
    image_dir="./images",
    transform_backend="albumentations",
    augmentation_preset="strong",
)
```

### With Explicit Label Columns

```python
data = MultiLabelImageDataModule(
    train_csv="train.csv",
    image_dir="./images",
    label_columns=["cat", "dog"],
    image_column="filepath",
)
```

### Full Training Pipeline

```python
from autotimm import AutoTrainer, ImageClassifier, MetricConfig

data = MultiLabelImageDataModule(
    train_csv="train.csv",
    image_dir="./images",
    val_csv="val.csv",
    image_size=224,
    batch_size=32,
)
data.setup("fit")

model = ImageClassifier(
    backbone="resnet50",
    num_classes=data.num_labels,
    multi_label=True,
    metrics=[
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="MultilabelAccuracy",
            params={"num_labels": data.num_labels},
            stages=["train", "val"],
            prog_bar=True,
        ),
    ],
)

trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_csv` | `str \| Path` | Required | Path to training CSV |
| `image_dir` | `str \| Path` | `"."` | Root directory for images |
| `val_csv` | `str \| Path \| None` | `None` | Validation CSV |
| `test_csv` | `str \| Path \| None` | `None` | Test CSV |
| `label_columns` | `list[str] \| None` | `None` | Label columns (auto-detected) |
| `image_column` | `str \| None` | `None` | Image column (first column) |
| `image_size` | `int` | `224` | Target image size |
| `batch_size` | `int` | `32` | Batch size |
| `num_workers` | `int` | `4` | Data loading workers |
| `val_split` | `float` | `0.1` | Validation split fraction |
| `train_transforms` | `Callable \| None` | `None` | Custom train transforms |
| `eval_transforms` | `Callable \| None` | `None` | Custom eval transforms |
| `augmentation_preset` | `str \| None` | `None` | Preset name |
| `transform_backend` | `str` | `"torchvision"` | `"torchvision"` or `"albumentations"` |
| `transform_config` | `TransformConfig \| None` | `None` | Transform configuration |
| `backbone` | `str \| nn.Module \| None` | `None` | Backbone for model normalization |
| `pin_memory` | `bool` | `True` | Pin memory for GPU |
| `persistent_workers` | `bool` | `False` | Keep workers alive |
| `prefetch_factor` | `int \| None` | `None` | Prefetch batches |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_labels` | `int \| None` | Number of labels (after setup) |
| `label_names` | `list[str] \| None` | Label column names (after setup) |
| `train_dataset` | `Dataset \| None` | Training dataset (after setup) |
| `val_dataset` | `Dataset \| None` | Validation dataset (after setup) |
| `test_dataset` | `Dataset \| None` | Test dataset (after setup) |

## CSV Format

```
image_path,cat,dog,outdoor,indoor
img1.jpg,1,0,1,0
img2.jpg,0,1,0,1
img3.jpg,1,1,1,0
```

## See Also

- [CSV Data Loading API](csv_data.md#multilabelimagedataset) - Direct dataset API for multi-label CSV data
- [ImageDataModule](data.md) - Single-label classification data module
- [CSV Data Loading Guide](../user-guide/data-loading/csv-data.md) - Complete guide with examples
- [TransformConfig](transforms.md) - Unified transform configuration
