<p align="center">
  <img src="autotimm.png" alt="AutoTimm" width="400">
</p>

<p align="center">
  <strong>Train state-of-the-art vision models with minimal code</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/autotimm/"><img src="https://img.shields.io/pypi/v/autotimm?color=blue&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/autotimm/"><img src="https://img.shields.io/pypi/pyversions/autotimm?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/theja-vanka/AutoTimm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://github.com/theja-vanka/AutoTimm/stargazers"><img src="https://img.shields.io/github/stars/theja-vanka/AutoTimm?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="https://theja-vanka.github.io/AutoTimm/">Documentation</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/getting-started/quickstart/">Quick Start</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/examples/">Examples</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/api/">API Reference</a>
</p>

---

AutoTimm combines the power of [timm](https://github.com/huggingface/pytorch-image-models) (1000+ pretrained models) with [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for a seamless training experience. Train image classifiers, object detectors, and segmentation models with any timm backbone. Go from idea to trained model in minutes, not hours.

## Highlights

| | |
|---|---|
| **4 Vision Tasks** | Classification, Object Detection, Semantic Segmentation, Instance Segmentation |
| **1000+ Backbones** | Access ResNet, EfficientNet, ViT, ConvNeXt, Swin, and more from timm |
| **Advanced Architectures** | DeepLabV3+, FCOS, Mask R-CNN style heads with feature pyramids |
| **Explicit Metrics** | Configure exactly what you track with MetricManager and torchmetrics |
| **Multi-Logger Support** | TensorBoard, MLflow, Weights & Biases, CSV — use them all at once |
| **Auto-Tuning** | Automatic learning rate and batch size finding before training |
| **Flexible Transforms** | Choose between torchvision (PIL) or albumentations (OpenCV) |
| **Production Ready** | Mixed precision, multi-GPU, gradient accumulation out of the box |

## Installation

```bash
pip install autotimm
```

<details>
<summary><strong>More installation options</strong></summary>

```bash
# With specific extras
pip install autotimm[albumentations]  # OpenCV-based transforms
pip install autotimm[segmentation]    # Segmentation tasks (includes albumentations + pycocotools)
pip install autotimm[tensorboard]     # TensorBoard logging
pip install autotimm[wandb]           # Weights & Biases
pip install autotimm[mlflow]          # MLflow tracking

# Everything
pip install autotimm[all]

# Development
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[dev,all]"
```

</details>

## Quick Start

### Image Classification

```python
from autotimm import AutoTrainer, ImageClassifier, ImageDataModule, MetricConfig

# Data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=64,
)

# Metrics
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
]

# Model & Train
model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    metrics=metrics,
    lr=1e-3,
)

trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
```

### Semantic Segmentation

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig

# Data
data = SegmentationDataModule(
    data_dir="./cityscapes",
    format="cityscapes",  # or "png", "coco", "voc"
    image_size=512,
    batch_size=8,
)

# Metrics
metrics = [
    MetricConfig(
        name="iou",
        backend="torchmetrics",
        metric_class="JaccardIndex",
        params={
            "task": "multiclass",
            "num_classes": 19,
            "average": "macro",
            "ignore_index": 255,
        },
        stages=["val", "test"],
        prog_bar=True,
    ),
]

# Model & Train
model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",  # or "fcn"
    loss_type="combined",        # CE + Dice
    dice_weight=1.0,
    metrics=metrics,
)

trainer = AutoTrainer(max_epochs=100)
trainer.fit(model, datamodule=data)
```

### Object Detection

```python
from autotimm import ObjectDetector, DetectionDataModule, MetricConfig

# Data
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=4,
)

# Metrics
metrics = [
    MetricConfig(
        name="mAP",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "bbox"},
        stages=["val", "test"],
        prog_bar=True,
    ),
]

# Model & Train
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
)

trainer = AutoTrainer(max_epochs=100)
trainer.fit(model, datamodule=data)
```

### Instance Segmentation

```python
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule, MetricConfig

# Data
data = InstanceSegmentationDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=4,
)

# Metrics
metrics = [
    MetricConfig(
        name="mask_mAP",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "segm"},
        stages=["val", "test"],
        prog_bar=True,
    ),
]

# Model & Train
model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    mask_loss_weight=1.0,
    metrics=metrics,
)

trainer = AutoTrainer(max_epochs=100)
trainer.fit(model, datamodule=data)
```

**[See the full documentation for more examples and features →](https://theja-vanka.github.io/AutoTimm/)**

## Import Styles

AutoTimm supports flexible import styles for convenience:

```python
# Direct imports
from autotimm import SemanticSegmentor, DiceLoss, MetricConfig

# Submodule aliases (NEW!)
from autotimm.task import SemanticSegmentor, InstanceSegmentor
from autotimm.loss import DiceLoss, CombinedSegmentationLoss
from autotimm.metric import MetricConfig, MetricManager
from autotimm.head import DeepLabV3PlusHead, MaskHead

# Namespace access
import autotimm
model = autotimm.task.SemanticSegmentor(...)
loss = autotimm.loss.DiceLoss(...)

# Original imports (still supported)
from autotimm.losses import DiceLoss
from autotimm.metrics import MetricConfig
from autotimm.tasks import SemanticSegmentor
```

## Supported Tasks & Architectures

### Classification
- **Models**: Any timm backbone (1000+ models)
- **Head**: Linear classification head with dropout
- **Losses**: CrossEntropy with label smoothing, Mixup support
- **Datasets**: Torchvision datasets, ImageFolder, custom loaders

### Object Detection
- **Architecture**: FCOS-style anchor-free detection
- **Components**: FPN, Detection Head (classification + bbox regression + centerness)
- **Losses**: Focal Loss, GIoU Loss, Centerness Loss
- **Datasets**: COCO format, custom annotations

### Semantic Segmentation
- **Architectures**: DeepLabV3+ (ASPP + decoder), FCN
- **Losses**: CrossEntropy, Dice, Focal, Combined (CE + Dice), Tversky
- **Datasets**: PNG masks, COCO stuff, Cityscapes, Pascal VOC
- **Metrics**: IoU (Jaccard Index), pixel accuracy, per-class metrics

### Instance Segmentation
- **Architecture**: FCOS detection + Mask R-CNN style mask head
- **Components**: FPN, Detection Head, Mask Head with ROI Align
- **Losses**: Detection losses + Binary mask loss
- **Datasets**: COCO instance segmentation format
- **Metrics**: Mask mAP, bbox mAP

## Examples

Ready-to-run scripts in the [`examples/`](examples/) directory:

| Example | Description |
|---------|-------------|
| [classify_cifar10.py](examples/classify_cifar10.py) | Basic classification with MetricManager and auto-tuning |
| [classify_custom_folder.py](examples/classify_custom_folder.py) | Train on your own dataset |
| [object_detection_coco.py](examples/object_detection_coco.py) | FCOS-style object detection on COCO dataset |
| [object_detection_transformers.py](examples/object_detection_transformers.py) | Transformer-based detection (ViT, Swin, DeiT) |
| [object_detection_rtdetr.py](examples/object_detection_rtdetr.py) | RT-DETR end-to-end detection (no NMS required) |
| [semantic_segmentation_cityscapes.py](examples/semantic_segmentation_cityscapes.py) | DeepLabV3+ segmentation on Cityscapes |
| [instance_segmentation_coco.py](examples/instance_segmentation_coco.py) | Mask R-CNN style instance segmentation |
| [vit_finetuning.py](examples/vit_finetuning.py) | Two-phase Vision Transformer fine-tuning |
| [multi_gpu_training.py](examples/multi_gpu_training.py) | Distributed training with DDP |
| [mlflow_tracking.py](examples/mlflow_tracking.py) | Experiment tracking with MLflow |

**[Browse all examples →](https://theja-vanka.github.io/AutoTimm/examples/)**

## Documentation

| Section | Description |
|---------|-------------|
| [Quick Start](https://theja-vanka.github.io/AutoTimm/getting-started/quickstart/) | Get up and running in 5 minutes |
| [User Guide](https://theja-vanka.github.io/AutoTimm/user-guide/data-loading/) | In-depth guides for all features |
| [API Reference](https://theja-vanka.github.io/AutoTimm/api/) | Complete API documentation |
| [Examples](https://theja-vanka.github.io/AutoTimm/examples/) | Runnable code examples |

## Explore Backbones

```python
import autotimm

# Search 1000+ models
autotimm.list_backbones("*efficientnet*", pretrained_only=True)
autotimm.list_backbones("*vit*")

# Inspect a model
backbone = autotimm.create_backbone("convnext_tiny")
print(f"Features: {backbone.num_features}, Params: {autotimm.count_parameters(backbone):,}")
```

## Key Features

### Multiple Loss Functions

**Classification**
- CrossEntropy with label smoothing
- Mixup augmentation

**Detection**
- Focal Loss (handles class imbalance)
- GIoU Loss (bbox regression)
- Centerness Loss (prediction quality)

**Segmentation**
- Dice Loss (overlap-based)
- Combined Loss (CE + Dice)
- Focal Loss (class imbalance)
- Tversky Loss (FP/FN weighting)

### Flexible Data Loading

- **Torchvision**: PIL-based transforms (fast CPU)
- **Albumentations**: OpenCV-based transforms (advanced augmentations)
- **Multiple Formats**: COCO, Cityscapes, Pascal VOC, ImageFolder, PNG masks
- **Custom Datasets**: Easy integration with PyTorch DataLoaders

### Advanced Training Features

- **Auto-tuning**: LR finder and batch size finder
- **Multi-GPU**: Distributed training with DDP
- **Mixed Precision**: Automatic mixed precision (AMP)
- **Gradient Accumulation**: Train larger batch sizes
- **Early Stopping**: Prevent overfitting
- **Checkpointing**: Save best models automatically

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

```bash
# Setup development environment
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[dev,all]"

# Run tests
pytest tests/ -v
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_losses.py
pytest tests/test_segmentation.py

# With coverage
pytest tests/ --cov=autotimm --cov-report=html
```

## Citation

If you use AutoTimm in your research, please cite:

```bibtex
@software{autotimm,
  author = {Krishnatheja Vanka},
  title = {AutoTimm: Automated Deep Learning for Computer Vision},
  url = {https://github.com/theja-vanka/AutoTimm},
  year = {2026}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with <a href="https://github.com/huggingface/pytorch-image-models">timm</a> and <a href="https://github.com/Lightning-AI/pytorch-lightning">PyTorch Lightning</a>
</p>
