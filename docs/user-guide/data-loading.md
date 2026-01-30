# Data Loading

AutoTimm provides two data modules:
- **ImageDataModule**: Image classification datasets
- **DetectionDataModule**: Object detection datasets in COCO format

## ImageDataModule

### Built-in Datasets

Load standard datasets automatically:

```python
from autotimm import ImageDataModule

data = ImageDataModule(
    data_dir="./data",          # Download location
    dataset_name="CIFAR10",     # Dataset name
    image_size=224,
    batch_size=64,
)
```

Supported datasets:

| Dataset | Classes | Size |
|---------|---------|------|
| `CIFAR10` | 10 | 32x32 |
| `CIFAR100` | 100 | 32x32 |
| `MNIST` | 10 | 28x28 |
| `FashionMNIST` | 10 | 28x28 |

### Custom Folder Datasets

Organize images in ImageFolder format:

```
dataset/
  train/
    class_a/
      img1.jpg
      img2.jpg
    class_b/
      img3.jpg
  val/
    class_a/
      img4.jpg
    class_b/
      img5.jpg
  test/           # Optional
    class_a/
      img6.jpg
```

Load with:

```python
data = ImageDataModule(
    data_dir="./dataset",
    image_size=384,
    batch_size=16,
)
data.setup("fit")
print(f"Classes: {data.num_classes}")
print(f"Class names: {data.class_names}")
```

### Auto Validation Split

If no `val/` directory exists, a fraction of training data is held out:

```python
data = ImageDataModule(
    data_dir="./dataset",
    val_split=0.1,   # 10% for validation (default)
)
```

## Transform Backends

### Torchvision (Default)

PIL-based transforms using torchvision:

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_backend="torchvision",  # Default
    augmentation_preset="default",
)
```

Available presets:

| Preset | Description |
|--------|-------------|
| `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `autoaugment` | AutoAugment (ImageNet policy) |
| `randaugment` | RandAugment (2 ops, magnitude 9) |
| `trivialaugment` | TrivialAugmentWide |

```python
# Using RandAugment
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    augmentation_preset="randaugment",
)
```

### Albumentations

OpenCV-based transforms (faster for some operations):

```bash
pip install autotimm[albumentations]
```

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_backend="albumentations",
    augmentation_preset="default",
)
```

Available presets:

| Preset | Description |
|--------|-------------|
| `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `strong` | Affine, blur/noise, ColorJitter, CoarseDropout |

```python
# Strong augmentation for better generalization
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_backend="albumentations",
    augmentation_preset="strong",
)
```

### Custom Transforms

#### Torchvision Custom

```python
from torchvision import transforms

custom_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data = ImageDataModule(
    data_dir="./dataset",
    train_transforms=custom_train,
)
```

#### Albumentations Custom

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

custom_train = A.Compose([
    A.RandomResizedCrop(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

data = ImageDataModule(
    data_dir="./dataset",
    transform_backend="albumentations",
    train_transforms=custom_train,
)
```

## Balanced Sampling

For imbalanced datasets, use weighted sampling:

```python
data = ImageDataModule(
    data_dir="./imbalanced_dataset",
    balanced_sampling=True,  # Oversamples minority classes
)
```

This uses `WeightedRandomSampler` to ensure each class is sampled equally during training.

## DataLoader Options

Fine-tune data loading performance:

```python
data = ImageDataModule(
    data_dir="./dataset",
    batch_size=64,
    num_workers=8,            # Parallel data loading
    pin_memory=True,          # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=4,        # Batches to prefetch per worker
)
```

## Dataset Summary

Get a summary of your data:

```python
data = ImageDataModule(data_dir="./dataset", dataset_name="CIFAR10")
data.setup("fit")
print(data.summary())
```

Output (Rich table):

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
└─────────────────────┴──────────┘
```

## Full Parameter Reference

```python
ImageDataModule(
    data_dir="./data",           # Root directory
    dataset_name=None,           # Built-in dataset name
    image_size=224,              # Target image size
    batch_size=32,               # Batch size
    num_workers=4,               # Data loading workers
    val_split=0.1,               # Validation split fraction
    train_transforms=None,       # Custom train transforms
    eval_transforms=None,        # Custom eval transforms
    augmentation_preset=None,    # "default", "autoaugment", etc.
    transform_backend="torchvision",  # "torchvision" or "albumentations"
    pin_memory=True,             # Pin memory for GPU
    persistent_workers=False,    # Keep workers alive
    prefetch_factor=None,        # Prefetch batches per worker
    balanced_sampling=False,     # Weighted sampling
)
```

---

## DetectionDataModule

For object detection tasks, use `DetectionDataModule` which loads COCO-format datasets.

### COCO Dataset Format

Expected directory structure:

```
coco/
  train2017/              # Training images
    000000000001.jpg
    000000000002.jpg
    ...
  val2017/                # Validation images
    000000000001.jpg
    ...
  test2017/               # Test images (optional)
    000000000001.jpg
    ...
  annotations/
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json  # Optional
```

The annotation files follow the standard COCO JSON format with image metadata, categories, and bounding box annotations.

### Basic Usage

```python
from autotimm import DetectionDataModule

data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,         # Standard COCO size
    batch_size=16,
    num_workers=4,
)
```

### Augmentation Presets

DetectionDataModule supports albumentations-based augmentation presets:

```python
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    augmentation_preset="default",  # or "strong"
)
```

Available presets:

| Preset | Description |
|--------|-------------|
| `default` | RandomResizedCrop (80-100%), HorizontalFlip |
| `strong` | Affine, blur, noise, brightness/contrast adjustments |

### Custom Transforms

Use custom albumentations pipelines with bbox-aware transforms:

```python
import albumentations as A

custom_train = A.Compose(
    [
        A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
)

data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    train_transforms=custom_train,
)
```

**Important**: Always include `bbox_params` when using custom transforms to ensure bounding boxes are properly transformed.

### Filter by Class

Train on a subset of classes:

```python
# Only detect person, bicycle, and car
selected_classes = [1, 2, 3]

data = DetectionDataModule(
    data_dir="./coco",
    image_size=512,
    batch_size=8,
    class_ids=selected_classes,
)
```

This filters the dataset to only include images containing these classes and remaps class IDs to 0-based indexing.

### Filter Small Boxes

Remove very small bounding boxes:

```python
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    min_bbox_area=32.0,  # Minimum box area in pixels
)
```

This helps avoid training on boxes that are too small to detect reliably.

### Dataset Summary

Get information about your detection dataset:

```python
data = DetectionDataModule(data_dir="./coco", image_size=640)
data.setup("fit")
print(data.summary())
```

Output:

```
┌─────────────────────┬──────────┐
│ Field               │ Value    │
├─────────────────────┼──────────┤
│ Data dir            │ ./coco   │
│ Image size          │ 640      │
│ Batch size          │ 16       │
│ Num classes         │ 80       │
│ Train samples       │ 118287   │
│ Val samples         │ 5000     │
│ Test samples        │ 0        │
└─────────────────────┴──────────┘
```

### DataLoader Options

Same as ImageDataModule:

```python
data = DetectionDataModule(
    data_dir="./coco",
    batch_size=16,
    num_workers=8,            # Parallel data loading
    pin_memory=True,          # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=4,        # Batches to prefetch per worker
)
```

### Full Parameter Reference

```python
DetectionDataModule(
    data_dir="./coco",           # COCO dataset root
    image_size=640,              # Target image size
    batch_size=16,               # Batch size
    num_workers=4,               # Data loading workers
    train_transforms=None,       # Custom albumentations transforms
    eval_transforms=None,        # Custom eval transforms
    augmentation_preset=None,    # "default" or "strong"
    class_ids=None,              # Filter to specific classes
    min_bbox_area=0.0,           # Minimum bounding box area
    pin_memory=True,             # Pin memory for GPU
    persistent_workers=False,    # Keep workers alive
    prefetch_factor=None,        # Prefetch batches per worker
)
```

### Complete Example

```python
from autotimm import AutoTrainer, DetectionDataModule, ObjectDetector, MetricConfig

# Data
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    num_workers=4,
    augmentation_preset="default",
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
)

# Train
trainer = AutoTrainer(max_epochs=12, gradient_clip_val=1.0)
trainer.fit(model, datamodule=data)
```
