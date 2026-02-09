# ImageDataModule

Lightning data module for image classification datasets.

## Overview

`ImageDataModule` is a PyTorch Lightning data module that supports:

- Built-in torchvision datasets (CIFAR10, CIFAR100, MNIST, FashionMNIST)
- Custom ImageFolder datasets
- Torchvision and albumentations transform backends
- Automatic validation splits
- Balanced sampling for imbalanced datasets

## API Reference

::: autotimm.ImageDataModule
    options:
      show_source: true
      members:
        - __init__
        - prepare_data
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader
        - summary

## Usage Examples

### Built-in Dataset

```python
from autotimm import ImageDataModule

data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=64,
)
```

### Custom Folder Dataset

```python
data = ImageDataModule(
    data_dir="./my_dataset",
    image_size=384,
    batch_size=32,
)
data.setup("fit")
print(f"Classes: {data.num_classes}")
print(f"Class names: {data.class_names}")
```

### With Albumentations

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_backend="albumentations",
    augmentation_preset="strong",
)
```

### With Augmentation Preset

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    augmentation_preset="randaugment",
)
```

### With Custom Transforms

```python
from torchvision import transforms

custom_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data = ImageDataModule(
    data_dir="./dataset",
    train_transforms=custom_train,
)
```

### With Balanced Sampling

```python
data = ImageDataModule(
    data_dir="./imbalanced_dataset",
    balanced_sampling=True,
)
```

### Performance Optimization

```python
data = ImageDataModule(
    data_dir="./dataset",
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
```

### With TransformConfig (Model-Specific Normalization)

Use `TransformConfig` with a backbone to get model-specific normalization:

```python
from autotimm import ImageDataModule, TransformConfig

# Create shared config
config = TransformConfig(
    preset="randaugment",
    image_size=384,
    use_timm_config=True,  # Use model's pretrained mean/std
)

data = ImageDataModule(
    data_dir="./dataset",
    transform_config=config,
    backbone="efficientnet_b4",  # Required for model-specific normalization
)
```

### Shared Config Between Model and Data

```python
from autotimm import ImageClassifier, ImageDataModule, TransformConfig, MetricConfig

# Shared config ensures same preprocessing
config = TransformConfig(preset="randaugment", image_size=384)
backbone_name = "efficientnet_b4"

# DataModule uses model's normalization
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_config=config,
    backbone=backbone_name,
)
data.setup("fit")

# Model uses same config for inference preprocessing
metrics = [MetricConfig(
    name="accuracy",
    backend="torchmetrics",
    metric_class="Accuracy",
    params={"task": "multiclass"},
    stages=["val"],
)]

model = ImageClassifier(
    backbone=backbone_name,
    num_classes=data.num_classes,
    metrics=metrics,
    transform_config=config,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str \| Path` | `"./data"` | Root directory |
| `dataset_name` | `str \| None` | `None` | Built-in dataset name |
| `image_size` | `int` | `224` | Target image size |
| `batch_size` | `int` | `32` | Batch size |
| `num_workers` | `int` | `4` | Data loading workers |
| `val_split` | `float` | `0.1` | Validation split fraction |
| `train_transforms` | `Callable \| None` | `None` | Custom train transforms |
| `eval_transforms` | `Callable \| None` | `None` | Custom eval transforms |
| `augmentation_preset` | `str \| None` | `None` | Preset name |
| `transform_backend` | `str` | `"torchvision"` | `"torchvision"` or `"albumentations"` |
| `transform_config` | `TransformConfig \| None` | `None` | Unified transform configuration |
| `backbone` | `str \| nn.Module \| None` | `None` | Backbone for model-specific normalization |
| `pin_memory` | `bool` | `True` | Pin memory for GPU |
| `persistent_workers` | `bool` | `False` | Keep workers alive |
| `prefetch_factor` | `int \| None` | `None` | Prefetch batches |
| `balanced_sampling` | `bool` | `False` | Weighted sampling |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_classes` | `int \| None` | Number of classes (after setup) |
| `class_names` | `list[str] \| None` | Class names (after setup) |
| `train_dataset` | `Dataset \| None` | Training dataset (after setup) |
| `val_dataset` | `Dataset \| None` | Validation dataset (after setup) |
| `test_dataset` | `Dataset \| None` | Test dataset (after setup) |

## Built-in Datasets

| Name | Classes | Image Size |
|------|---------|------------|
| `CIFAR10` | 10 | 32x32 |
| `CIFAR100` | 100 | 32x32 |
| `MNIST` | 10 | 28x28 |
| `FashionMNIST` | 10 | 28x28 |

## Augmentation Presets

### Torchvision

| Preset | Description |
|--------|-------------|
| `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `autoaugment` | AutoAugment (ImageNet policy) |
| `randaugment` | RandAugment (2 ops, magnitude 9) |
| `trivialaugment` | TrivialAugmentWide |

### Albumentations

| Preset | Description |
|--------|-------------|
| `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `strong` | Affine, blur/noise, ColorJitter, CoarseDropout |

## Folder Structure

```
dataset/
├── train/
│   ├── class_a/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_b/
│       └── img3.jpg
├── val/           # Optional (uses val_split if missing)
│   ├── class_a/
│   │   └── img4.jpg
│   └── class_b/
│       └── img5.jpg
└── test/          # Optional
    ├── class_a/
    │   └── img6.jpg
    └── class_b/
        └── img7.jpg
```

## Summary Output

```python
data.setup("fit")
print(data.summary())
```

```
┌─────────────────────┬──────────┐
│ Field               │ Value    │
├─────────────────────┼──────────┤
│ Data dir            │ ./data   │
│ Dataset             │ CIFAR10  │
│ Image size          │ 224      │
│ Batch size          │ 32       │
│ Num classes         │ 10       │
│ Train samples       │ 45000    │
│ Val samples         │ 5000     │
│ Test samples        │ 10000    │
│ Balanced sampling   │ False    │
│ Class: airplane     │ 4500     │
│ Class: automobile   │ 4500     │
│ ...                 │ ...      │
└─────────────────────┴──────────┘
```

---

## CSV Data Loading

For CSV-based data loading, `ImageDataModule` supports `train_csv`, `val_csv`, and `test_csv` parameters for single-label classification.

For multi-label classification from CSV files, see [MultiLabelImageDataModule](multilabel_data.md).

For direct CSV dataset usage (without DataModules), see the [CSV Data Loading API](csv_data.md) documentation.

### CSV Classification Example

```python
from autotimm import ImageDataModule

data = ImageDataModule(
    train_csv="train.csv",
    val_csv="val.csv",
    image_dir="./images",
    image_size=224,
    batch_size=32,
)
```

**CSV Format:**
```csv
image_path,label
img001.jpg,cat
img002.jpg,dog
```

See [CSV Data API](csv_data.md#csvimagedataset) for detailed CSV format specification.

