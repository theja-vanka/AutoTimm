# AutoTimm
![AutoTimm](autotimm.png){ width="500" }

Automated deep learning image tasks powered by [timm](https://github.com/huggingface/pytorch-image-models) and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).

AutoTimm lets you train image classifiers, object detectors, and segmentation models with any of timm's 1000+ backbones in a few lines of Python.

## AutoTimm Workflow

```mermaid
graph TD
    A[Raw Data] --> A1[Data Preprocessing]
    A1 -->|ImageDataModule| A2[Transforms & Augmentation]
    A2 --> A3[DataLoaders]
    
    A3 --> B[Select Backbone]
    B --> B1[Choose from 1000+ timm models]
    B1 --> B2[Build Model]
    B2 -->|ImageClassifier<br/>ObjectDetector<br/>SemanticSegmentor| C[Configure Training]
    
    C --> C1[Set Hyperparameters]
    C1 -->|AutoTrainer| C2[Initialize Callbacks]
    C2 --> D[Training Loop]
    
    D --> D1[Forward Pass]
    D1 --> D2[Loss Calculation]
    D2 --> D3[Backward Pass]
    D3 --> D4[Weight Update]
    D4 --> E{Validation}
    
    E -->|Compute Metrics| E1[Accuracy/mAP/IoU]
    E1 --> E2[Save Best Model]
    E2 --> E3{Training Complete?}
    E3 -->|No| D
    E3 -->|Yes| F[Test Model]
    
    F --> F1[Final Metrics]
    F1 --> G[Export Model]
    G -->|TorchScript<br/>ONNX<br/>TensorRT| H[Production Deployment]
    
    style A fill:#2196F3,stroke:#1976D2,color:#fff
    style A2 fill:#42A5F5,stroke:#1976D2,color:#fff
    style B1 fill:#2196F3,stroke:#1976D2,color:#fff
    style C1 fill:#42A5F5,stroke:#1976D2,color:#fff
    style D fill:#2196F3,stroke:#1976D2,color:#fff
    style D2 fill:#42A5F5,stroke:#1976D2,color:#fff
    style E1 fill:#2196F3,stroke:#1976D2,color:#fff
    style F1 fill:#42A5F5,stroke:#1976D2,color:#fff
    style G fill:#2196F3,stroke:#1976D2,color:#fff
    style H fill:#42A5F5,stroke:#1976D2,color:#fff
```

## Features

- **4 vision tasks** - Image classification, object detection, semantic segmentation, and instance segmentation
- **1000+ backbones** - Any timm model works: CNNs (ResNet, EfficientNet, ConvNeXt) and Transformers (ViT, Swin, DeiT)
- **Flexible architectures** - Built-in FCOS detector, DeepLabV3+ segmentation, Mask R-CNN style instance segmentation
- **Advanced losses** - Focal, Dice, Tversky, Combined CE+Dice, GIoU for bbox regression
- **Configurable metrics** - Use torchmetrics or custom metrics
- **Multiple logger backends** - TensorBoard, MLflow, W&B, CSV simultaneously
- **Auto-tuning** - Automatic learning rate and batch size finding
- **Enhanced logging** - Learning rate, gradient norms, confusion matrices
- **Flexible transforms** - Torchvision (PIL) or albumentations (OpenCV) with bbox and mask support (both included by default)

## Quick Example

```python
from autotimm import (
    AutoTrainer, ImageClassifier, ImageDataModule,
    LoggerConfig, MetricConfig,
)

# Data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=64,
)

# Metrics (explicit configuration required)
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

# Model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    metrics=metrics,
    lr=1e-3,
)

# Trainer with logging
trainer = AutoTrainer(
    max_epochs=10,
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
    checkpoint_monitor="val/accuracy",
)

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

## Why AutoTimm?

| Feature | AutoTimm | Raw PyTorch | Lightning |
|---------|----------|-------------|-----------|
| 1000+ backbones | Yes | Manual | Manual |
| Configurable metrics | Yes | Manual | Manual |
| Multi-logger support | Yes | Manual | Partial |
| Auto LR/batch finding | Yes | No | Yes |
| Lines of code | ~20 | ~200+ | ~100 |

## Next Steps

- [Installation](getting-started/installation.md) - Get AutoTimm installed
- [Quick Start](getting-started/quickstart.md) - Train your first model
- [User Guide](user-guide/data-loading/index.md) - Deep dive into features
- [API Reference](api/index.md) - Full API documentation
- [Examples](examples/index.md) - Runnable example scripts
