![AutoTimm](autotimm.png)

Automated deep learning image tasks powered by [timm](https://github.com/huggingface/pytorch-image-models) and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).

AutoTimm lets you train image classifiers with any of timm's 1000+ backbones in a few lines of Python. It supports both torchvision (PIL) and albumentations (OpenCV) transform pipelines, and comes with built-in logging for TensorBoard, MLflow, and Weights & Biases.

## Installation

```bash
# Core
pip install autotimm

# With albumentations (OpenCV-based transforms)
pip install autotimm[albumentations]

# With specific logger backends
pip install autotimm[tensorboard]
pip install autotimm[wandb]
pip install autotimm[mlflow]

# Everything
pip install autotimm[all]
```

For development:

```bash
git clone https://github.com/your-org/autotimm.git
cd autotimm
pip install -e ".[dev,all]"
```

## Quick Start

```python
from autotimm import ImageClassifier, ImageDataModule, create_trainer

data = ImageDataModule(data_dir="./data", dataset_name="CIFAR10", image_size=224, batch_size=64)
model = ImageClassifier(backbone="resnet18", num_classes=10, lr=1e-3)
trainer = create_trainer(max_epochs=10, logger="tensorboard")

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

## Custom Dataset

Organize your images in ImageFolder format:

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
  test/          # optional
    class_a/
      img6.jpg
    class_b/
      img7.jpg
```

Then:

```python
from autotimm import ImageClassifier, ImageDataModule, create_trainer

data = ImageDataModule(data_dir="./dataset", image_size=384, batch_size=16)
data.setup("fit")

model = ImageClassifier(backbone="efficientnet_b3", num_classes=data.num_classes, lr=3e-4)
trainer = create_trainer(max_epochs=20, precision="bf16-mixed")

trainer.fit(model, datamodule=data)
```

If no `val/` directory exists, a fraction of the training data is held out automatically (controlled by `val_split`, default 10%).

## Transform Backends

AutoTimm supports two transform backends. Choose via `transform_backend`:

### Torchvision (default, PIL-based)

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_backend="torchvision",     # default
    augmentation_preset="randaugment",   # "default", "autoaugment", "randaugment", "trivialaugment"
)
```

### Albumentations (OpenCV-based)

```bash
pip install autotimm[albumentations]
```

```python
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    transform_backend="albumentations",
    augmentation_preset="strong",   # "default" or "strong"
)
```

When `transform_backend="albumentations"` is used with folder datasets, images are loaded with OpenCV (`cv2.imread`) instead of PIL. Built-in datasets (CIFAR10, etc.) automatically convert PIL images to numpy arrays for the pipeline.

### Custom albumentations pipeline

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from autotimm import ImageDataModule

custom_train = A.Compose([
    A.RandomResizedCrop(size=(224, 224)),
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-20, 20), scale=(0.8, 1.2), p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.8),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

data = ImageDataModule(
    data_dir="./dataset",
    transform_backend="albumentations",
    train_transforms=custom_train,
)
```

### Custom torchvision pipeline

```python
from torchvision import transforms
from autotimm import ImageDataModule

custom_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data = ImageDataModule(data_dir="./data", dataset_name="CIFAR10", train_transforms=custom_train)
```

## Backbone Discovery

Browse the 1000+ backbones available through timm:

```python
import autotimm

# Search by pattern
autotimm.list_backbones("*convnext*")
autotimm.list_backbones("*efficientnet*", pretrained_only=True)

# Inspect a backbone
backbone = autotimm.create_backbone("resnet50")
print(f"Output features: {backbone.num_features}")
print(f"Parameters: {autotimm.count_parameters(backbone):,}")
```

### Advanced backbone configuration

```python
from autotimm import BackboneConfig, ImageClassifier

cfg = BackboneConfig(
    model_name="vit_base_patch16_224",
    pretrained=True,
    drop_path_rate=0.1,
)
model = ImageClassifier(backbone=cfg, num_classes=100)
```

## Logging

Switch between logging backends by changing one argument:

```python
from autotimm import create_trainer

# TensorBoard (default)
trainer = create_trainer(logger="tensorboard")

# Weights & Biases
trainer = create_trainer(logger="wandb", logger_kwargs={"project": "my-project"})

# MLflow
trainer = create_trainer(logger="mlflow", logger_kwargs={"experiment_name": "my-exp"})

# CSV (no extra dependencies)
trainer = create_trainer(logger="csv")

# No logging
trainer = create_trainer(logger="none")
```

## Data Module Features

### Balanced sampling for imbalanced datasets

```python
data = ImageDataModule(
    data_dir="./imbalanced_dataset",
    balanced_sampling=True,   # WeightedRandomSampler oversamples minority classes
)
```

### Dataset summary

```python
data = ImageDataModule(data_dir="./dataset", dataset_name="CIFAR10")
data.setup("fit")
print(data.summary())
```

Output:

```
ImageDataModule Summary
  Data dir       : data
  Dataset        : CIFAR10
  Image size     : 224
  Batch size     : 32
  Num workers    : 4
  Backend        : torchvision
  Num classes    : 10
  Train samples  : 45000
  Val samples    : 5000
  Balanced sampling : False
  Class distribution (train):
    airplane: 4527
    automobile: 4512
    ...
```

### Persistent workers and prefetching

```python
data = ImageDataModule(
    data_dir="./dataset",
    num_workers=8,
    persistent_workers=True,  # keep workers alive between epochs
    prefetch_factor=4,        # prefetch 4 batches per worker
)
```

## Training Features

### Freeze backbone for linear probing

```python
model = ImageClassifier(backbone="resnet50", num_classes=10, freeze_backbone=True, lr=1e-2)
```

### Mixup augmentation

```python
model = ImageClassifier(backbone="resnet50", num_classes=10, mixup_alpha=0.2)
```

### Label smoothing

```python
model = ImageClassifier(backbone="resnet50", num_classes=10, label_smoothing=0.1)
```

### Learning rate schedulers

```python
# Cosine annealing (default)
model = ImageClassifier(backbone="resnet50", num_classes=10, scheduler="cosine")

# Step decay
model = ImageClassifier(backbone="resnet50", num_classes=10, scheduler="step")

# No scheduler
model = ImageClassifier(backbone="resnet50", num_classes=10, scheduler="none")
```

### Gradient accumulation, clipping, and mixed precision

```python
trainer = create_trainer(
    max_epochs=20,
    accumulate_grad_batches=4,
    gradient_clip_val=1.0,
    precision="bf16-mixed",
)
```

### Two-phase fine-tuning (linear probe then full)

```python
from autotimm import ImageClassifier, ImageDataModule, create_trainer

data = ImageDataModule(data_dir="./dataset", image_size=224, batch_size=32)
data.setup("fit")

# Phase 1: Linear probe
model = ImageClassifier(
    backbone="vit_base_patch16_224",
    num_classes=data.num_classes,
    freeze_backbone=True,
    lr=1e-2,
)
trainer = create_trainer(max_epochs=5)
trainer.fit(model, datamodule=data)

# Phase 2: Full fine-tune with lower LR
for param in model.backbone.parameters():
    param.requires_grad = True

model._lr = 1e-4
trainer = create_trainer(max_epochs=20, gradient_clip_val=1.0)
trainer.fit(model, datamodule=data)
```

## Examples

The [`examples/`](examples/) directory contains runnable scripts:

| Script | Description |
|---|---|
| [`classify_cifar10.py`](examples/classify_cifar10.py) | ResNet-18 on CIFAR-10 with TensorBoard |
| [`classify_custom_folder.py`](examples/classify_custom_folder.py) | EfficientNet on a custom folder dataset with W&B |
| [`albumentations_cifar10.py`](examples/albumentations_cifar10.py) | CIFAR-10 with albumentations strong augmentation |
| [`albumentations_custom_folder.py`](examples/albumentations_custom_folder.py) | Custom albumentations pipeline with OpenCV loading |
| [`vit_finetuning.py`](examples/vit_finetuning.py) | Two-phase ViT fine-tuning (linear probe then full) |
| [`balanced_sampling.py`](examples/balanced_sampling.py) | Weighted sampling for class-imbalanced data |
| [`mlflow_tracking.py`](examples/mlflow_tracking.py) | CIFAR-100 training with MLflow experiment tracking |
| [`backbone_discovery.py`](examples/backbone_discovery.py) | Explore and inspect available timm backbones |

## API Reference

### Core

| Symbol | Type | Description |
|---|---|---|
| `ImageClassifier` | `LightningModule` | Image classifier: timm backbone + classification head + training loop |
| `ImageDataModule` | `LightningDataModule` | Data module for ImageFolder and built-in datasets with torchvision/albumentations transforms |
| `create_trainer` | `function` | Factory for `pl.Trainer` with logger and checkpoint wiring |

### Backbone

| Symbol | Type | Description |
|---|---|---|
| `BackboneConfig` | `dataclass` | Configuration for timm backbone (model name, pretrained, dropout, etc.) |
| `create_backbone` | `function` | Create a headless timm model from a name or config |
| `list_backbones` | `function` | Search available timm model names by glob pattern |

### Heads and Utilities

| Symbol | Type | Description |
|---|---|---|
| `ClassificationHead` | `nn.Module` | Linear head with optional dropout |
| `create_logger` | `function` | Factory for TensorBoard / MLflow / W&B / CSV loggers |
| `count_parameters` | `function` | Count model parameters (total or trainable) |

### Data (via `autotimm.data`)

| Symbol | Type | Description |
|---|---|---|
| `ImageFolderCV2` | `Dataset` | ImageFolder dataset that loads images with OpenCV |
| `get_train_transforms` | `function` | Get transforms by preset name and backend |
| `albu_default_train_transforms` | `function` | Default albumentations training transforms |
| `albu_strong_train_transforms` | `function` | Strong albumentations augmentation pipeline |
| `albu_default_eval_transforms` | `function` | Default albumentations eval transforms |

## Built-in Datasets

`ImageDataModule` supports these torchvision datasets via the `dataset_name` parameter:

- `CIFAR10`
- `CIFAR100`
- `MNIST`
- `FashionMNIST`

## Augmentation Presets

| Backend | Preset | Description |
|---|---|---|
| `torchvision` | `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `torchvision` | `autoaugment` | AutoAugment (ImageNet policy) |
| `torchvision` | `randaugment` | RandAugment (2 ops, magnitude 9) |
| `torchvision` | `trivialaugment` | TrivialAugmentWide |
| `albumentations` | `default` | RandomResizedCrop, HorizontalFlip, ColorJitter |
| `albumentations` | `strong` | Affine, blur/noise, ColorJitter, CoarseDropout |

## License

Apache 2.0
