# TransformConfig

Unified transform configuration for models and data modules.

## Overview

`TransformConfig` provides a single configuration interface for:

- Transform presets (default, randaugment, autoaugment, etc.)
- Model-specific normalization from timm
- Inference-time preprocessing via `model.preprocess()`
- Shared configuration between models and data modules

## API Reference

::: autotimm.TransformConfig
    options:
      show_source: true
      members:
        - __init__
        - with_overrides
        - to_dict

## Usage Examples

### Basic Usage

```python
from autotimm import ImageClassifier, TransformConfig, MetricConfig

# Create model with transform config
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=[
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        ),
    ],
    transform_config=TransformConfig(),  # Uses model's pretrained normalization
)

# Preprocess raw images for inference
from PIL import Image
image = Image.open("test.jpg")
tensor = model.preprocess(image)  # Returns (1, 3, 224, 224)
output = model(tensor)
```

### With Custom Image Size

```python
config = TransformConfig(
    image_size=384,
    use_timm_config=True,  # Use model's mean/std
)

model = ImageClassifier(
    backbone="efficientnet_b4",
    num_classes=100,
    metrics=metrics,
    transform_config=config,
)

# Preprocessing now uses 384x384
tensor = model.preprocess(image)  # Returns (1, 3, 384, 384)
```

### With Augmentation Preset

```python
config = TransformConfig(
    preset="randaugment",
    image_size=224,
    randaugment_num_ops=2,
    randaugment_magnitude=9,
)

# For training with the same config
datamodule = ImageDataModule(
    data_dir="./data",
    transform_config=config,
    backbone="resnet50",
)
```

### Custom Normalization

```python
# Override model's normalization (not recommended for pretrained models)
config = TransformConfig(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    use_timm_config=False,  # Use our mean/std instead
)
```

### Shared Config for Model and DataModule

```python
from autotimm import ImageClassifier, ImageDataModule, TransformConfig

# Create shared config
config = TransformConfig(
    preset="randaugment",
    image_size=384,
    use_timm_config=True,
)

backbone_name = "efficientnet_b4"

# DataModule uses the same transforms as model
datamodule = ImageDataModule(
    data_dir="./data",
    transform_config=config,
    backbone=backbone_name,
)

# Model uses same preprocessing
model = ImageClassifier(
    backbone=backbone_name,
    num_classes=datamodule.num_classes,
    metrics=metrics,
    transform_config=config,
)
```

### Get Model's Data Config

```python
model = ImageClassifier(
    backbone="vit_base_patch16_224",
    num_classes=100,
    metrics=metrics,
    transform_config=TransformConfig(),
)

# Get the normalization config
data_config = model.get_data_config()
print(f"Mean: {data_config['mean']}")      # (0.5, 0.5, 0.5) for ViT
print(f"Std: {data_config['std']}")        # (0.5, 0.5, 0.5) for ViT
print(f"Input size: {data_config['input_size']}")  # (3, 224, 224)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | `str` | `"default"` | Augmentation preset name |
| `backend` | `str` | `"torchvision"` | `"torchvision"` or `"albumentations"` |
| `image_size` | `int` | `224` | Target image size (square) |
| `use_timm_config` | `bool` | `True` | Use model's pretrained mean/std |
| `mean` | `tuple[float, ...]` | `None` | Override normalization mean |
| `std` | `tuple[float, ...]` | `None` | Override normalization std |
| `interpolation` | `str` | `"bicubic"` | Resize interpolation mode |
| `crop_pct` | `float` | `0.875` | Center crop percentage for eval |
| `min_bbox_area` | `float` | `0.0` | Detection: min bbox area |
| `min_visibility` | `float` | `0.0` | Detection: min visibility |
| `bbox_format` | `str` | `"coco"` | Detection: bbox format |
| `ignore_index` | `int` | `255` | Segmentation: ignore index |
| `randaugment_num_ops` | `int` | `2` | RandAugment: number of ops |
| `randaugment_magnitude` | `int` | `9` | RandAugment: magnitude |

---

## Augmentation Presets

### Torchvision Backend

| Preset | Description |
|--------|-------------|
| `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `autoaugment` | AutoAugment (ImageNet policy) |
| `randaugment` | RandAugment with configurable ops/magnitude |
| `trivialaugment` | TrivialAugmentWide |
| `light` | RandomResizedCrop, HorizontalFlip only |

### Albumentations Backend

| Preset | Description |
|--------|-------------|
| `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `strong` | Affine, blur/noise, ColorJitter, CoarseDropout |
| `light` | RandomResizedCrop, HorizontalFlip only |

---

## Model Preprocessing Methods

When a model is created with `transform_config`, it gains these methods:

### `model.preprocess(images, is_train=False)`

Preprocess raw images for model inference.

```python
from PIL import Image
import numpy as np

# Single PIL image
image = Image.open("test.jpg")
tensor = model.preprocess(image)  # (1, 3, H, W)

# List of PIL images
images = [Image.open(f"img{i}.jpg") for i in range(4)]
tensor = model.preprocess(images)  # (4, 3, H, W)

# Numpy array
img_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
tensor = model.preprocess(img_np)  # (1, 3, H, W)

# Tensor (passes through unchanged)
tensor_in = torch.randn(2, 3, 224, 224)
tensor_out = model.preprocess(tensor_in)  # (2, 3, 224, 224)
```

### `model.get_data_config()`

Get the model's normalization configuration.

```python
config = model.get_data_config()
# Returns:
# {
#     'mean': (0.485, 0.456, 0.406),
#     'std': (0.229, 0.224, 0.225),
#     'input_size': (3, 224, 224),
#     'interpolation': 'bicubic',
#     'crop_pct': 0.875,
# }
```

### `model.get_transform(is_train=False)`

Get the transform pipeline directly.

```python
eval_transform = model.get_transform(is_train=False)
train_transform = model.get_transform(is_train=True)
```

---

## Utility Functions

### resolve_backbone_data_config

Get model-specific preprocessing config from timm.

```python
from autotimm import resolve_backbone_data_config

config = resolve_backbone_data_config("efficientnet_b0")
print(config["mean"])       # (0.485, 0.456, 0.406)
print(config["std"])        # (0.229, 0.224, 0.225)
print(config["input_size"]) # (3, 224, 224)

# With overrides
config = resolve_backbone_data_config(
    "efficientnet_b0",
    override_mean=(0.5, 0.5, 0.5),
    override_std=(0.5, 0.5, 0.5),
)
```

### get_transforms_from_backbone

Create transforms using model-specific normalization.

```python
from autotimm import get_transforms_from_backbone, TransformConfig

config = TransformConfig(preset="randaugment", image_size=384)

train_transforms = get_transforms_from_backbone(
    backbone="efficientnet_b4",
    transform_config=config,
    is_train=True,
)

eval_transforms = get_transforms_from_backbone(
    backbone="efficientnet_b4",
    transform_config=config,
    is_train=False,
)
```

### create_inference_transform

Convenience function for creating inference transforms.

```python
from autotimm import create_inference_transform

# Simple usage
transform = create_inference_transform("resnet50")
tensor = transform(pil_image)

# With custom config
config = TransformConfig(image_size=384)
transform = create_inference_transform("resnet50", transform_config=config)
```

### list_transform_presets

List available transform presets for a given backend.

```python
from autotimm import list_transform_presets

# List all torchvision presets
presets = list_transform_presets(backend="torchvision")
print(presets)
# ['default', 'autoaugment', 'randaugment', 'trivialaugment', 'light']

# List albumentations presets
presets = list_transform_presets(backend="albumentations")
print(presets)
# ['default', 'strong', 'light']

# Get preset details
presets = list_transform_presets(backend="torchvision", verbose=True)
for name, description in presets:
    print(f"{name}: {description}")
# default: RandomResizedCrop, HorizontalFlip, ColorJitter
# autoaugment: AutoAugment (ImageNet policy)
# randaugment: RandAugment with configurable ops/magnitude
# trivialaugment: TrivialAugmentWide
# light: RandomResizedCrop, HorizontalFlip only
```

---

## Integration with DataModules

All AutoTimm DataModules support `transform_config` and `backbone` parameters:

```python
from autotimm import (
    ImageDataModule,
    DetectionDataModule,
    SegmentationDataModule,
    InstanceSegmentationDataModule,
    TransformConfig,
)

config = TransformConfig(preset="randaugment", image_size=384)

# Classification
data = ImageDataModule(
    data_dir="./data",
    transform_config=config,
    backbone="efficientnet_b4",
)

# Detection
data = DetectionDataModule(
    data_dir="./coco",
    transform_config=config,
    backbone="resnet50",
)

# Segmentation
data = SegmentationDataModule(
    data_dir="./cityscapes",
    format="cityscapes",
    transform_config=config,
    backbone="resnet50",
)

# Instance Segmentation
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    transform_config=config,
    backbone="resnet50",
)
```

---

## Best Practices

### 1. Use `use_timm_config=True` (Default)

Always use the model's pretrained normalization for best results:

```python
# Good - uses model's pretrained normalization
config = TransformConfig(use_timm_config=True)

# Not recommended - may hurt pretrained model performance
config = TransformConfig(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    use_timm_config=False,
)
```

### 2. Share Config Between Model and DataModule

```python
config = TransformConfig(preset="randaugment", image_size=384)
backbone = "efficientnet_b4"

# Same normalization for training and inference
datamodule = ImageDataModule(..., transform_config=config, backbone=backbone)
model = ImageClassifier(..., transform_config=config)
```

### 3. Use preprocess() for Inference

```python
# Simple and correct - uses model's exact preprocessing
model = ImageClassifier(..., transform_config=TransformConfig())
tensor = model.preprocess(pil_image)
output = model(tensor)
```

---

## See Also

- [Transforms User Guide](../user-guide/data-loading/transforms.md) - Comprehensive transforms guide
- [ImageClassifier](classifier.md) - Classification model with preprocessing
- [ImageDataModule](data.md) - Data module with TransformConfig support
- [Classification Inference](../user-guide/inference/classification-inference.md) - Inference guide
