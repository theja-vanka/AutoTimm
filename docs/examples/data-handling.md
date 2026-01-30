# Data Handling Examples

This page demonstrates data loading, augmentation, and balanced sampling techniques.

## Balanced Sampling

Handle imbalanced datasets with weighted sampling.

```python
from autotimm import ImageDataModule


def main():
    data = ImageDataModule(
        data_dir="./imbalanced_dataset",
        balanced_sampling=True,  # Oversamples minority classes
    )


if __name__ == "__main__":
    main()
```

This feature automatically computes class weights and uses a `WeightedRandomSampler` to ensure balanced training batches even when your dataset has imbalanced class distributions.

---

## Albumentations (Strong Augmentation)

Use strong augmentations with albumentations preset.

```python
from autotimm import ImageDataModule


def main():
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        transform_backend="albumentations",
        augmentation_preset="strong",
    )


if __name__ == "__main__":
    main()
```

The `strong` preset includes:
- Random resized crop
- Horizontal flip
- Color jitter
- Random brightness/contrast
- Random rotation
- Cutout/CoarseDropout
- Normalization

---

## Custom Albumentations Pipeline

Define your own custom albumentations pipeline for maximum control.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

from autotimm import ImageDataModule


def main():
    custom_train = A.Compose([
        A.RandomResizedCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    custom_val = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    data = ImageDataModule(
        data_dir="./dataset",
        transform_backend="albumentations",
        train_transforms=custom_train,
        val_transforms=custom_val,
        test_transforms=custom_val,
    )


if __name__ == "__main__":
    main()
```

**Popular Albumentations Transforms:**

| Transform | Use Case | Typical Parameters |
|-----------|----------|-------------------|
| `RandomResizedCrop` | Training | `height=224, width=224, scale=(0.8, 1.0)` |
| `HorizontalFlip` | Training | `p=0.5` |
| `VerticalFlip` | Specific domains | `p=0.5` |
| `ColorJitter` | Robustness to color | `brightness=0.2, contrast=0.2, saturation=0.2` |
| `GaussianBlur` | Robustness to blur | `blur_limit=(3, 7), p=0.3` |
| `RandomRotate90` | Rotation invariance | `p=0.5` |
| `ShiftScaleRotate` | Geometric transforms | `shift_limit=0.1, scale_limit=0.1, rotate_limit=15` |
| `CoarseDropout` | Regularization | `max_holes=8, max_height=8, max_width=8` |
| `GridDistortion` | Advanced augmentation | `p=0.3` |
| `ElasticTransform` | Advanced augmentation | `alpha=1, sigma=50, p=0.3` |

---

## Data Module Configuration

Full example showing all data module options.

```python
from autotimm import ImageDataModule


def main():
    data = ImageDataModule(
        # Dataset
        data_dir="./data",
        dataset_name="CIFAR10",  # or None for custom folder structure
        
        # Image settings
        image_size=224,
        
        # Batch settings
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        
        # Augmentation
        transform_backend="albumentations",  # or "torchvision"
        augmentation_preset="strong",  # or "default", "light", None
        train_transforms=None,  # Custom transforms override preset
        val_transforms=None,
        test_transforms=None,
        
        # Sampling
        balanced_sampling=False,  # Enable for imbalanced datasets
        
        # Splits (for folder datasets)
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
    )
    
    # Setup and inspect
    data.setup("fit")
    print(f"Number of classes: {data.num_classes}")
    print(f"Class names: {data.classes}")
    print(f"Training samples: {len(data.train_dataset)}")
    print(f"Validation samples: {len(data.val_dataset)}")


if __name__ == "__main__":
    main()
```

---

## Running Data Handling Examples

Clone the repository and run examples:

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[all]"

# Run balanced sampling example
python examples/balanced_sampling.py

# Run albumentations example
python examples/albumentations_cifar10.py

# Run custom albumentations example
python examples/albumentations_custom_folder.py
```
