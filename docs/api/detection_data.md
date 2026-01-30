# DetectionDataModule

Lightning data module for object detection datasets in COCO format.

## Overview

`DetectionDataModule` is a PyTorch Lightning data module for object detection that supports:

- COCO format datasets with automatic annotation loading
- Torchvision and albumentations transform backends
- Built-in augmentation presets optimized for detection
- Efficient collation for variable-sized objects per image
- Multi-worker data loading with prefetching

## API Reference

::: autotimm.DetectionDataModule
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

### Basic COCO Dataset

```python
from autotimm import DetectionDataModule

data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
)
```

### With Augmentation Preset

```python
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    augmentation_preset="strong",  # Enhanced augmentation
)
```

### With Albumentations

```python
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    transform_backend="albumentations",
    augmentation_preset="strong",
)
```

### With Custom Transforms

```python
from torchvision import transforms as T

custom_train = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    train_transforms=custom_train,
)
```

### Performance Optimization

```python
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str \| Path` | `"./coco"` | Root directory |
| `image_size` | `int` | `640` | Target image size |
| `batch_size` | `int` | `16` | Batch size |
| `num_workers` | `int` | `4` | Data loading workers |
| `train_transforms` | `Callable \| None` | `None` | Custom train transforms |
| `val_transforms` | `Callable \| None` | `None` | Custom validation transforms |
| `augmentation_preset` | `str \| None` | `"default"` | Preset name |
| `transform_backend` | `str` | `"torchvision"` | `"torchvision"` or `"albumentations"` |
| `pin_memory` | `bool` | `True` | Pin memory for GPU |
| `persistent_workers` | `bool` | `False` | Keep workers alive |
| `prefetch_factor` | `int \| None` | `None` | Prefetch batches |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_classes` | `int \| None` | Number of object classes (after setup) |
| `train_dataset` | `Dataset \| None` | Training dataset (after setup) |
| `val_dataset` | `Dataset \| None` | Validation dataset (after setup) |
| `test_dataset` | `Dataset \| None` | Test dataset (after setup) |

## COCO Format

The data directory should follow this structure:

```
coco/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_test2017.json  # Optional
├── train2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
├── val2017/
│   ├── 000000000001.jpg
│   └── ...
└── test2017/          # Optional
    └── ...
```

### Annotation Format

COCO annotations should have this structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "000000000001.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "person"
    }
  ]
}
```

## Augmentation Presets

### Torchvision

| Preset | Description |
|--------|-------------|
| `default` | RandomHorizontalFlip, ColorJitter, ToTensor |
| `strong` | Default + RandomPhotometricDistort |

### Albumentations

| Preset | Description |
|--------|-------------|
| `default` | HorizontalFlip, ColorJitter |
| `strong` | HorizontalFlip, RandomBrightnessContrast, HueSaturationValue, Blur, Noise |

## Data Output

Each batch contains:

```python
batch = {
    "image": Tensor,      # Shape: (B, 3, H, W)
    "boxes": List[Tensor],   # List of (N, 4) tensors in [x1, y1, x2, y2] format
    "labels": List[Tensor],  # List of (N,) tensors with class indices
}
```

## Summary Output

```python
data.setup("fit")
print(data.summary())
```

```
┌─────────────────────┬──────────────┐
│ Field               │ Value        │
├─────────────────────┼──────────────┤
│ Data dir            │ ./coco       │
│ Image size          │ 640          │
│ Batch size          │ 16           │
│ Num classes         │ 80           │
│ Train samples       │ 118287       │
│ Val samples         │ 5000         │
│ Test samples        │ 0            │
│ Transform backend   │ torchvision  │
│ Augmentation        │ default      │
└─────────────────────┴──────────────┘
```
