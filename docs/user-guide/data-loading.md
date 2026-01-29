# Data Loading

AutoTimm provides `ImageDataModule`, a flexible data loading solution that supports both built-in torchvision datasets and custom folder-based datasets.

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
