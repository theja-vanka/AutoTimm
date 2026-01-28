# AutoTimm

Automated deep learning image tasks powered by [timm](https://github.com/huggingface/pytorch-image-models) and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).

AutoTimm lets you train image classifiers with any timm backbone in a few lines of Python, with built-in support for TensorBoard, MLflow, and Weights & Biases.

## Installation

```bash
# Core (TensorBoard works out of the box)
pip install autotimm

# With specific logger backends
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

## Backbone Discovery

Browse the 1000+ backbones available through timm:

```python
import autotimm

# Search by pattern
autotimm.list_backbones("*convnext*")
autotimm.list_backbones("*efficientnet*", pretrained_only=True)

# Advanced configuration
from autotimm import BackboneConfig

cfg = BackboneConfig(
    model_name="vit_base_patch16_224",
    pretrained=True,
    drop_path_rate=0.1,
)
model = autotimm.ImageClassifier(backbone=cfg, num_classes=100)
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

## Advanced Configuration

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

### Custom transforms

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

### Gradient accumulation and clipping

```python
trainer = create_trainer(
    max_epochs=20,
    accumulate_grad_batches=4,
    gradient_clip_val=1.0,
)
```

## API Reference

| Symbol | Type | Description |
|---|---|---|
| `ImageClassifier` | `LightningModule` | Image classifier composing a timm backbone and classification head |
| `ImageDataModule` | `LightningDataModule` | Data module supporting ImageFolder and built-in datasets (CIFAR10, etc.) |
| `create_trainer` | `function` | Factory for `pl.Trainer` with logger and checkpoint wiring |
| `create_logger` | `function` | Factory for TensorBoard / MLflow / W&B / CSV loggers |
| `BackboneConfig` | `dataclass` | Configuration for timm backbone creation |
| `create_backbone` | `function` | Create a headless timm model from a name or config |
| `list_backbones` | `function` | Search available timm model names by glob pattern |
| `ClassificationHead` | `nn.Module` | Linear head with optional dropout |
| `count_parameters` | `function` | Count model parameters |

## Built-in Datasets

`ImageDataModule` supports these torchvision datasets out of the box via the `dataset_name` parameter:

- `CIFAR10`
- `CIFAR100`
- `MNIST`
- `FashionMNIST`

## License

Apache 2.0
