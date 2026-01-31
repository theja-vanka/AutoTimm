# Transforms

AutoTimm provides a flexible transform system for image preprocessing and augmentation, supporting both torchvision and albumentations backends with model-specific normalization.

## Overview

The transform system consists of three main components:

1. **TransformConfig**: Unified configuration dataclass for all transform settings
2. **Augmentation Presets**: Built-in transform pipelines for training and evaluation
3. **Model-Specific Normalization**: Automatic normalization using timm's pretrained statistics

---

## TransformConfig

`TransformConfig` is the central configuration class for all transforms in AutoTimm. It provides a consistent interface across models and data modules.

### Basic Usage

```python
from autotimm import TransformConfig

# Default configuration
config = TransformConfig()

# Custom configuration
config = TransformConfig(
    preset="randaugment",
    backend="torchvision",
    image_size=384,
    use_timm_config=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | str | "default" | Augmentation preset name |
| `backend` | str | "torchvision" | Transform backend ("torchvision" or "albumentations") |
| `image_size` | int | 224 | Target image size (square) |
| `use_timm_config` | bool | True | Use model's pretrained normalization |
| `mean` | tuple | None | Override normalization mean |
| `std` | tuple | None | Override normalization std |
| `interpolation` | str | "bicubic" | Resize interpolation mode |
| `crop_pct` | float | 0.875 | Center crop percentage for eval |

#### Detection-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_bbox_area` | float | 0.0 | Minimum bbox area to keep |
| `min_visibility` | float | 0.0 | Minimum visibility ratio (0.0-1.0) |
| `bbox_format` | str | "coco" | Bbox format ("coco", "pascal_voc", "yolo") |

#### Segmentation-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ignore_index` | int | 255 | Label index to ignore in masks |

#### RandAugment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `randaugment_num_ops` | int | 2 | Number of augmentation operations |
| `randaugment_magnitude` | int | 9 | Magnitude of augmentations (0-30) |

### Configuration Methods

```python
# Create config with overrides
base_config = TransformConfig(image_size=224)
large_config = base_config.with_overrides(image_size=384)

# Convert to dictionary
config_dict = config.to_dict()
```

---

## Augmentation Presets

AutoTimm provides several built-in augmentation presets for different training scenarios.

### Torchvision Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | RandomResizedCrop, HorizontalFlip, ColorJitter | General training |
| `autoaugment` | AutoAugment (ImageNet policy) | Proven augmentation |
| `randaugment` | RandAugment (configurable ops/magnitude) | Flexible augmentation |
| `trivialaugment` | TrivialAugmentWide | Simple but effective |
| `light` | RandomResizedCrop, HorizontalFlip only | Minimal augmentation |

### Albumentations Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | RandomResizedCrop, HorizontalFlip, ColorJitter | General training |
| `strong` | Affine, blur/noise, ColorJitter, CoarseDropout | Heavy augmentation |
| `light` | RandomResizedCrop, HorizontalFlip only | Minimal augmentation |

### Preset Examples

```python
from autotimm import ImageDataModule, TransformConfig

# Standard training
data = ImageDataModule(
    data_dir="./data",
    transform_config=TransformConfig(preset="default"),
    backbone="resnet50",
)

# Strong augmentation with RandAugment
data = ImageDataModule(
    data_dir="./data",
    transform_config=TransformConfig(
        preset="randaugment",
        randaugment_num_ops=3,
        randaugment_magnitude=12,
    ),
    backbone="resnet50",
)

# Heavy augmentation with albumentations
data = ImageDataModule(
    data_dir="./data",
    transform_config=TransformConfig(
        preset="strong",
        backend="albumentations",
    ),
    backbone="resnet50",
)
```

---

## Default Transform Pipelines

### Training Transforms (default preset)

```
1. RandomResizedCrop(image_size)
2. RandomHorizontalFlip(p=0.5)
3. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
4. ToTensor()
5. Normalize(mean, std)
```

### Evaluation Transforms

```
1. Resize(image_size / crop_pct)  # e.g., 256 for 224 with 0.875 crop_pct
2. CenterCrop(image_size)
3. ToTensor()
4. Normalize(mean, std)
```

### AutoAugment Pipeline

```
1. RandomResizedCrop(image_size)
2. RandomHorizontalFlip(p=0.5)
3. AutoAugment(policy=IMAGENET)
4. ToTensor()
5. Normalize(mean, std)
```

### RandAugment Pipeline

```
1. RandomResizedCrop(image_size)
2. RandomHorizontalFlip(p=0.5)
3. RandAugment(num_ops=2, magnitude=9)
4. ToTensor()
5. Normalize(mean, std)
```

### Strong Albumentations Pipeline

```
1. RandomResizedCrop(image_size)
2. HorizontalFlip(p=0.5)
3. Affine(translate, scale, rotate, p=0.5)
4. OneOf([MotionBlur, GaussianBlur, GaussNoise], p=0.3)
5. ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
6. CoarseDropout(num_holes=1-3, p=0.3)
7. Normalize(mean, std)
8. ToTensorV2()
```

---

## Model-Specific Normalization

Different pretrained models use different normalization statistics. AutoTimm automatically uses the correct normalization for each model.

### How It Works

1. When `use_timm_config=True` (default), AutoTimm queries timm for the model's pretrained data config
2. The normalization mean/std are extracted from this config
3. Transforms are created with the correct normalization

### Common Model Normalizations

| Model Family | Mean | Std |
|-------------|------|-----|
| ResNet, EfficientNet | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| ViT (CLIP) | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) |
| Inception | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) |

### Example: Get Model Data Config

```python
from autotimm.data import resolve_backbone_data_config

# Get config for a specific model
config = resolve_backbone_data_config("efficientnet_b4")
print(f"Mean: {config['mean']}")
print(f"Std: {config['std']}")
print(f"Input size: {config['input_size']}")
print(f"Interpolation: {config['interpolation']}")
print(f"Crop percentage: {config['crop_pct']}")
```

### Override Normalization

```python
# Use ImageNet normalization regardless of model
config = TransformConfig(
    use_timm_config=False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

# Or just override specific values
config = TransformConfig(
    mean=(0.5, 0.5, 0.5),  # Override mean, get std from model
)
```

---

## Using Transforms with DataModules

### Method 1: TransformConfig (Recommended)

```python
from autotimm import ImageDataModule, TransformConfig

config = TransformConfig(
    preset="randaugment",
    image_size=384,
)

data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_config=config,
    backbone="efficientnet_b4",  # Required for model-specific normalization
)
```

### Method 2: Augmentation Preset Only

```python
# Simple preset selection
data = ImageDataModule(
    data_dir="./data",
    augmentation_preset="randaugment",
    image_size=224,
)
```

### Method 3: Custom Transforms

```python
from torchvision import transforms

custom_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

custom_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data = ImageDataModule(
    data_dir="./data",
    train_transforms=custom_train,
    eval_transforms=custom_eval,
)
```

---

## Using Transforms with Models

Models can use TransformConfig for inference-time preprocessing:

```python
from autotimm import ImageClassifier, TransformConfig, MetricConfig
from PIL import Image

# Create model with TransformConfig
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=[MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["val"],
    )],
    transform_config=TransformConfig(),
)

# Preprocess raw images
image = Image.open("test.jpg")
tensor = model.preprocess(image)  # Uses model's normalization

# Batch preprocessing
images = [Image.open(f"img{i}.jpg") for i in range(4)]
batch = model.preprocess(images)  # Returns (4, 3, 224, 224)

# Get model's data config
config = model.get_data_config()
print(f"Mean: {config['mean']}")
print(f"Std: {config['std']}")
```

---

## Shared Config for Model and Data

Ensure consistent preprocessing between training and inference:

```python
from autotimm import (
    ImageClassifier,
    ImageDataModule,
    TransformConfig,
    MetricConfig,
)

# Shared configuration
config = TransformConfig(
    preset="randaugment",
    image_size=384,
)
backbone_name = "efficientnet_b4"

# DataModule uses model's normalization during training
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_config=config,
    backbone=backbone_name,
)
data.setup("fit")

# Model uses same config for inference preprocessing
model = ImageClassifier(
    backbone=backbone_name,
    num_classes=data.num_classes,
    metrics=[MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["val"],
    )],
    transform_config=config,
)
```

---

## Utility Functions

### get_transforms_from_backbone

Create transforms using model-specific normalization:

```python
from autotimm.data import get_transforms_from_backbone, TransformConfig

config = TransformConfig(preset="randaugment", image_size=384)

# Training transforms
train_transform = get_transforms_from_backbone(
    backbone="efficientnet_b4",
    transform_config=config,
    is_train=True,
)

# Evaluation transforms
eval_transform = get_transforms_from_backbone(
    backbone="efficientnet_b4",
    transform_config=config,
    is_train=False,
)
```

### create_inference_transform

Convenience function for inference:

```python
from autotimm.data import create_inference_transform

# Quick inference transform with model normalization
transform = create_inference_transform("resnet50")
tensor = transform(pil_image)

# With custom config
from autotimm import TransformConfig
config = TransformConfig(image_size=384)
transform = create_inference_transform("efficientnet_b4", config)
```

### resolve_backbone_data_config

Get model's pretrained data configuration:

```python
from autotimm.data import resolve_backbone_data_config

config = resolve_backbone_data_config("vit_base_patch16_224")
print(config)
# {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'input_size': (3, 224, 224), ...}

# With overrides
config = resolve_backbone_data_config(
    "resnet50",
    override_mean=(0.5, 0.5, 0.5),
    override_input_size=(3, 384, 384),
)
```

---

## Backend Comparison

### Torchvision

**Pros:**
- Native PyTorch integration
- No additional dependencies
- PIL-based (good for web images)
- Faster for simple transforms

**Cons:**
- Limited augmentation variety
- No spatial transforms that preserve bboxes

**Best for:** Standard image classification

### Albumentations

**Pros:**
- Rich augmentation library
- Faster for complex transforms (OpenCV backend)
- Built-in bbox/mask handling
- Better for detection/segmentation

**Cons:**
- Additional dependency
- Requires `pip install autotimm[albumentations]`

**Best for:** Object detection, segmentation, advanced augmentation

### Switching Backends

```python
# Torchvision (default)
config = TransformConfig(backend="torchvision")

# Albumentations
config = TransformConfig(backend="albumentations")
```

---

## Best Practices

### 1. Use TransformConfig for Consistency

```python
# Good: Shared config ensures same preprocessing
config = TransformConfig(preset="randaugment", image_size=384)
data = ImageDataModule(..., transform_config=config, backbone="resnet50")
model = ImageClassifier(..., transform_config=config)
```

### 2. Match Training and Inference Preprocessing

```python
# Always use the same normalization for training and inference
model = ImageClassifier(
    backbone="efficientnet_b4",
    transform_config=TransformConfig(use_timm_config=True),  # Uses model's pretrained stats
)
```

### 3. Choose Appropriate Augmentation Strength

```python
# Light augmentation for small datasets or transfer learning
config = TransformConfig(preset="light")

# Standard augmentation for most cases
config = TransformConfig(preset="default")

# Strong augmentation for large datasets or preventing overfitting
config = TransformConfig(preset="strong", backend="albumentations")
```

### 4. Consider Image Size Trade-offs

```python
# Smaller size: faster training, less GPU memory
config = TransformConfig(image_size=224)

# Larger size: better accuracy, slower training
config = TransformConfig(image_size=384)
```

---

## Troubleshooting

### Wrong Predictions After Training

**Problem:** Model predictions don't match training performance

**Solution:** Ensure inference uses the same normalization:

```python
# Correct: Use model's preprocess method
tensor = model.preprocess(image)

# Or manually match normalization
config = model.get_data_config()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std']),
])
```

### Albumentations Not Found

**Problem:** `ImportError: Albumentations is required`

**Solution:** Install with extras:

```bash
pip install autotimm[albumentations]
```

### Bounding Boxes Not Preserved

**Problem:** Detection bboxes become invalid after augmentation

**Solution:** Use albumentations with proper bbox params:

```python
config = TransformConfig(
    backend="albumentations",
    bbox_format="coco",
    min_visibility=0.3,  # Filter bboxes with <30% visibility
)
```

---

## See Also

- [TransformConfig API](../../api/transforms.md) - Full API reference
- [ImageDataModule](../../api/data.md) - Data loading documentation
- [Image Classification Data](image-classification-data.md) - Classification data guide
- [Object Detection Data](object-detection-data.md) - Detection data guide
