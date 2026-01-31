# Models

AutoTimm provides two main model architectures for computer vision tasks. Choose the model that best fits your use case.

## Available Models

### [ImageClassifier](image-classifier.md)

Image classification with any timm backbone and flexible training options.

**Key Features:**

- 1000+ pretrained backbones from timm
- Transfer learning with backbone freezing
- Optimizers: AdamW, SGD, Adam, and more
- Schedulers: Cosine, Step, OneCycle
- Regularization: Label smoothing, Mixup, Dropout
- Two-phase fine-tuning for transformers

**Use Cases:**

- Image categorization (CIFAR-10, ImageNet, custom datasets)
- Transfer learning from pretrained models
- Fine-tuning vision transformers (ViT, Swin, DeiT)
- Multi-class classification tasks

[Learn more about ImageClassifier →](image-classifier.md)

### [ObjectDetector](object-detector.md)

FCOS-style anchor-free object detection with Feature Pyramid Networks.

**Key Features:**

- Any timm backbone (CNNs and transformers)
- FCOS detection architecture (anchor-free)
- Feature Pyramid Network (FPN) for multi-scale detection
- Focal loss for hard example mining
- Configurable inference thresholds and NMS
- Support for transformer backbones (Swin, ViT)

**Use Cases:**

- Object detection on COCO or custom datasets
- Real-time detection applications
- Multi-scale object detection
- Transformer-based detection

[Learn more about ObjectDetector →](object-detector.md)

## Quick Comparison

| Feature | ImageClassifier | ObjectDetector |
|---------|-----------------|----------------|
| Task | Image classification | Object detection |
| Output | Class labels + confidence | Bounding boxes + classes |
| Architecture | Backbone + Classifier head | Backbone + FPN + Detection head |
| Backbones | 1000+ timm models | 1000+ timm models |
| Loss | CrossEntropy (+ Label smoothing) | Focal + GIoU + Centerness |
| Training speed | Fast | Moderate |
| Inference speed | Very fast | Fast |
| Memory usage | Low | Moderate |

## Common Concepts

Both models share several key concepts:

### Backbone Selection

Both models support 1000+ backbones from timm:

```python
import autotimm

# List all available backbones
autotimm.list_backbones()

# Search for specific backbones
autotimm.list_backbones("*resnet*")
autotimm.list_backbones("*vit*")
```

Popular backbone families: ResNet, EfficientNet, ConvNeXt, ViT, Swin, DeiT.

### Transfer Learning

Both models support transfer learning with backbone freezing:

```python
model = ImageClassifier(
    backbone="resnet50",
    freeze_backbone=True,  # Only train classifier head
    lr=1e-2,
)
```

### Optimizer & Scheduler Configuration

Both models share the same optimizer and scheduler configuration:

```python
model = ImageClassifier(
    backbone="resnet50",
    optimizer="adamw",
    lr=1e-3,
    scheduler="cosine",
)
```

### Metrics Integration

Both models use the same metrics configuration system:

```python
from autotimm import MetricConfig

metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
    ),
]
```

## See Also

- [Image Classification Data](../data-loading/image-classification-data.md) - Data loading for classification
- [Object Detection Data](../data-loading/object-detection-data.md) - Data loading for detection
- [Classification Inference](../inference/classification-inference.md) - Making predictions with classifiers
- [Object Detection Inference](../inference/object-detection-inference.md) - Making predictions with detectors
- [Training Guide](../training/training.md) - Training models with AutoTrainer
- [Metrics](../evaluation/metrics.md) - Configuring metrics and logging
