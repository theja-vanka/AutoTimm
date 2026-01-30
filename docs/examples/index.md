# Examples

The [`examples/`](https://github.com/theja-vanka/AutoTimm/tree/main/examples) directory contains runnable scripts demonstrating AutoTimm features.

## Quick Reference

| Script | Description |
|--------|-------------|
| [`classify_cifar10.py`](#cifar-10-classification) | ResNet-18 on CIFAR-10 with MetricManager and auto-tuning |
| [`classify_custom_folder.py`](#custom-folder-dataset) | EfficientNet on a custom folder dataset with W&B |
| [`object_detection_coco.py`](#object-detection-on-coco) | FCOS-style object detection on COCO dataset |
| [`object_detection_transformers.py`](#transformer-based-object-detection) | Vision Transformers for object detection (ViT, Swin, DeiT) |
| [`object_detection_rtdetr.py`](#rt-detr-real-time-detection-transformer) | RT-DETR end-to-end transformer detection (no NMS) |
| [`multiple_loggers.py`](#multiple-loggers) | TensorBoard + CSV logging simultaneously |
| [`auto_tuning.py`](#auto-tuning) | Automatic LR and batch size finding |
| [`inference.py`](#inference) | Model inference and batch prediction |
| [`detailed_evaluation.py`](#detailed-evaluation) | Confusion matrix and per-class metrics |
| [`multi_gpu_training.py`](#multi-gpu-training) | Multi-GPU and distributed training |
| [`vit_finetuning.py`](#vit-fine-tuning) | Two-phase ViT fine-tuning |
| [`balanced_sampling.py`](#balanced-sampling) | Weighted sampling for imbalanced data |
| [`mlflow_tracking.py`](#mlflow-tracking) | MLflow experiment tracking |
| [`albumentations_cifar10.py`](#albumentations) | Albumentations strong augmentation |
| [`albumentations_custom_folder.py`](#custom-albumentations) | Custom albumentations pipeline |
| [`backbone_discovery.py`](#backbone-discovery) | Explore timm backbones |

---

## CIFAR-10 Classification

Basic training with ResNet-18 on CIFAR-10 using MetricManager.

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
    )

    # Metrics
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        lr=1e-3,
    )

    # Train
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
        checkpoint_monitor="val/accuracy",
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## Custom Folder Dataset

Training on your own images organized in folders.

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Your data structure:
    # dataset/
    #   train/
    #     class_a/...
    #     class_b/...
    #   val/
    #     class_a/...
    #     class_b/...

    data = ImageDataModule(
        data_dir="./dataset",
        image_size=384,
        batch_size=16,
    )
    data.setup("fit")

    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    model = ImageClassifier(
        backbone="efficientnet_b3",
        num_classes=data.num_classes,
        metrics=metric_manager,
    )

    trainer = AutoTrainer(
        max_epochs=20,
        logger=[LoggerConfig(backend="wandb", params={"project": "my-project"})],
        checkpoint_monitor="val/accuracy",
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## Object Detection on COCO

FCOS-style anchor-free object detection with timm backbones and Feature Pyramid Networks.

```python
from autotimm import (
    AutoTrainer,
    DetectionDataModule,
    LoggerConfig,
    MetricConfig,
    ObjectDetector,
)


def main():
    # Data - COCO format detection dataset
    data = DetectionDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=16,
        num_workers=4,
        augmentation_preset="default",  # or "strong" for more augmentation
    )

    # Metrics - MeanAveragePrecision for object detection
    metric_configs = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model - FCOS-style detector with timm backbone
    model = ObjectDetector(
        backbone="resnet50",  # Any timm backbone works
        num_classes=80,  # COCO has 80 classes
        metrics=metric_configs,
        fpn_channels=256,
        head_num_convs=4,
        lr=1e-4,
        scheduler="multistep",
        scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
    )

    # Train
    trainer = AutoTrainer(
        max_epochs=12,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
        checkpoint_monitor="val/map",
        checkpoint_mode="max",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Key Features:**

- **FCOS architecture**: Anchor-free detection with Feature Pyramid Network (P3-P7)
- **Any timm backbone**: ResNet, EfficientNet, ConvNeXt, etc.
- **Focal Loss + GIoU Loss**: State-of-the-art detection losses
- **COCO-compatible**: Supports standard COCO JSON format
- **Flexible augmentation**: Built-in presets or custom albumentations pipelines

---

## Transformer-Based Object Detection

Use Vision Transformers (ViT, Swin, DeiT) as backbones for object detection.

```python
from autotimm import AutoTrainer, DetectionDataModule, LoggerConfig, MetricConfig, ObjectDetector


def main():
    # Data
    data = DetectionDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=8,  # Smaller batch for transformers
        num_workers=4,
        augmentation_preset="default",
    )

    # Metrics
    metric_configs = [
        MetricConfig(
            name="mAP",
            backend="torchmetrics",
            metric_class="MeanAveragePrecision",
            params={"box_format": "xyxy", "iou_type": "bbox"},
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Option 1: Swin Transformer (recommended for detection)
    model_swin = ObjectDetector(
        backbone="swin_tiny_patch4_window7_224",
        num_classes=80,
        metrics=metric_configs,
        fpn_channels=256,
        head_num_convs=4,
        lr=1e-4,
        scheduler="multistep",
        scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
    )

    # Option 2: Vision Transformer (highest accuracy)
    model_vit = ObjectDetector(
        backbone="vit_base_patch16_224",
        num_classes=80,
        metrics=metric_configs,
        fpn_channels=256,
        head_num_convs=4,
        lr=1e-4,
        scheduler="cosine",
    )

    # Option 3: Two-phase training (recommended)
    model = ObjectDetector(
        backbone="swin_base_patch4_window7_224",
        num_classes=80,
        metrics=metric_configs,
        fpn_channels=256,
        freeze_backbone=True,  # Phase 1: freeze backbone
        lr=1e-3,
    )

    # Phase 1: Train detection head
    trainer_phase1 = AutoTrainer(
        max_epochs=5,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs/phase1"})],
        gradient_clip_val=1.0,
    )
    trainer_phase1.fit(model, datamodule=data)

    # Phase 2: Fine-tune entire model
    for param in model.backbone.parameters():
        param.requires_grad = True
    model._lr = 1e-5  # Lower LR for fine-tuning

    trainer_phase2 = AutoTrainer(
        max_epochs=15,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs/phase2"})],
        gradient_clip_val=1.0,
    )
    trainer_phase2.fit(model, datamodule=data)
    trainer_phase2.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Transformer Backbone Comparison:**

| Backbone | Speed | Accuracy | Memory | Best For |
|----------|-------|----------|--------|----------|
| `vit_tiny_patch16_224` | Medium | Good | Low | Quick experiments |
| `vit_base_patch16_224` | Slow | Best | High | Maximum accuracy |
| `swin_tiny_patch4_window7_224` | Fast | Good | Medium | Balanced performance |
| `swin_base_patch4_window7_224` | Medium | Better | Medium-High | Production use |
| `deit_base_patch16_224` | Slow | Best | High | Limited data |

**Tips:**
- Use smaller batch sizes (8-16) - transformers need more memory
- Two-phase training works very well with transformers
- Gradient clipping (1.0) is important for stability
- Lower learning rates (1e-4 to 1e-5) than CNNs
- Swin Transformers are best for detection (hierarchical features)

---

## RT-DETR (Real-Time Detection Transformer)

End-to-end transformer-based object detection with no NMS required.

```python
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from autotimm import AutoTrainer, DetectionDataModule, LoggerConfig


class RTDetrModule(torch.nn.Module):
    """Wrapper for RT-DETR model."""

    def __init__(self, model_name="PekingU/rtdetr_r50vd", num_classes=80, lr=1e-4):
        super().__init__()
        self.model = RTDetrForObjectDetection.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        self.processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.lr = lr

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        labels = self._convert_targets(targets, images.shape[0])
        outputs = self(pixel_values=images, labels=labels)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main():
    # Data - Use AutoTimm's DetectionDataModule
    data = DetectionDataModule(
        data_dir="./coco",
        image_size=640,
        batch_size=4,
        num_workers=4,
        augmentation_preset="default",
    )

    # Model - RT-DETR with ResNet-50 backbone
    model = RTDetrModule(
        model_name="PekingU/rtdetr_r50vd",
        num_classes=80,
        lr=1e-4,
    )

    # Train
    trainer = AutoTrainer(
        max_epochs=12,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs/rtdetr"})],
        gradient_clip_val=0.1,
        precision="bf16-mixed",
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**RT-DETR vs FCOS:**

| Feature | RT-DETR | FCOS (AutoTimm) |
|---------|---------|-----------------|
| Architecture | Transformer-based | CNN-based |
| Detection | Query-based | Anchor-free points |
| NMS required | No ✨ | Yes |
| Inference speed | Real-time | Real-time |
| Memory usage | Higher | Lower |
| Small objects | Good | Excellent |
| Large objects | Excellent | Good |

**Available RT-DETR Models:**

| Model | Parameters | Speed | Best For |
|-------|------------|-------|----------|
| `PekingU/rtdetr_r18vd` | 20M | Fastest | Quick experiments |
| `PekingU/rtdetr_r34vd` | 31M | Fast | Balanced |
| `PekingU/rtdetr_r50vd` | 42M | Medium | Recommended ⭐ |
| `PekingU/rtdetr_r101vd` | 76M | Slower | Maximum accuracy |

**When to use RT-DETR:**
- Need end-to-end differentiable pipeline
- Want to avoid NMS post-processing
- Detecting large objects or scenes
- Have sufficient GPU memory

**When to use FCOS:**
- Need maximum efficiency
- Detecting small objects
- Limited GPU memory
- Need multi-scale detection (P3-P7)

**Requirements:**
```bash
pip install transformers
```

---

## Multiple Loggers

Log to TensorBoard and CSV simultaneously.

```python
from autotimm import AutoTrainer, LoggerConfig


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/tb"}),
            LoggerConfig(backend="csv", params={"save_dir": "logs/csv"}),
        ],
    )


if __name__ == "__main__":
    main()
```

---

## Auto-Tuning

Automatically find optimal learning rate and batch size.

```python
from autotimm import AutoTrainer, TunerConfig


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        tuner_config=TunerConfig(
            auto_lr=True,
            auto_batch_size=True,
            lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0, "num_training": 100},
            batch_size_kwargs={"mode": "power", "init_val": 16},
        ),
    )

    trainer.fit(model, datamodule=data)  # Runs tuning before training


if __name__ == "__main__":
    main()
```

---

## Inference

Make predictions with a trained model.

```python
import torch
from PIL import Image
from torchvision import transforms

from autotimm import ImageClassifier, MetricConfig, MetricManager


def main():
    # Define metrics for loading
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        ),
    ]
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Load model
    model = ImageClassifier.load_from_checkpoint(
        "path/to/checkpoint.ckpt",
        backbone="resnet50",
        num_classes=10,
        metrics=metric_manager,
    )
    model.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Predict
    image = Image.open("test.jpg").convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = model(input_tensor).softmax(dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    print(f"Prediction: {pred}, Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
```

---

## Detailed Evaluation

Log confusion matrix and detailed metrics with MetricManager.

```python
from autotimm import LoggingConfig, MetricConfig, MetricManager


def main():
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
        MetricConfig(
            name="precision",
            backend="torchmetrics",
            metric_class="Precision",
            params={"task": "multiclass", "average": "macro"},
            stages=["test"],
        ),
        MetricConfig(
            name="recall",
            backend="torchmetrics",
            metric_class="Recall",
            params={"task": "multiclass", "average": "macro"},
            stages=["test"],
        ),
    ]

    # Create MetricManager for programmatic access
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Access specific metrics
    accuracy = metric_manager.get_metric_by_name("accuracy", stage="val")
    f1_config = metric_manager.get_config_by_name("f1")

    # Iterate over all configs
    for config in metric_manager:
        print(f"{config.name}: stages={config.stages}")

    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
            log_confusion_matrix=True,
        ),
    )


if __name__ == "__main__":
    main()
```

---

## Multi-GPU Training

Distributed training across multiple GPUs.

```python
from autotimm import AutoTrainer


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision="bf16-mixed",
    )


if __name__ == "__main__":
    main()
```

---

## ViT Fine-Tuning

Two-phase fine-tuning for Vision Transformers with MetricManager.

```python
from autotimm import AutoTrainer, ImageClassifier, MetricConfig, MetricManager


def main():
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        ),
    ]

    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    # Phase 1: Linear probe (frozen backbone)
    model = ImageClassifier(
        backbone="vit_base_patch16_224",
        num_classes=data.num_classes,
        metrics=metric_manager,
        freeze_backbone=True,
        lr=1e-2,
    )

    trainer = AutoTrainer(max_epochs=5)
    trainer.fit(model, datamodule=data)

    # Phase 2: Full fine-tune
    for param in model.backbone.parameters():
        param.requires_grad = True

    model._lr = 1e-4
    trainer = AutoTrainer(max_epochs=20, gradient_clip_val=1.0)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

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

---

## MLflow Tracking

Track experiments with MLflow.

```python
from autotimm import AutoTrainer, LoggerConfig


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[
            LoggerConfig(
                backend="mlflow",
                params={
                    "experiment_name": "cifar10-experiments",
                    "tracking_uri": "http://localhost:5000",
                },
            ),
        ],
    )


if __name__ == "__main__":
    main()
```

---

## Albumentations

Use strong augmentations with albumentations.

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

---

## Custom Albumentations

Define custom albumentations pipeline.

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

    data = ImageDataModule(
        data_dir="./dataset",
        transform_backend="albumentations",
        train_transforms=custom_train,
    )


if __name__ == "__main__":
    main()
```

---

## Backbone Discovery

Explore available timm backbones.

```python
import autotimm


def main():
    # List all backbones
    all_models = autotimm.list_backbones()
    print(f"Total models: {len(all_models)}")

    # Search by pattern
    resnet = autotimm.list_backbones("*resnet*")
    efficientnet = autotimm.list_backbones("*efficientnet*", pretrained_only=True)
    vit = autotimm.list_backbones("*vit*")

    # Inspect a backbone
    backbone = autotimm.create_backbone("resnet50")
    print(f"Output features: {backbone.num_features}")
    print(f"Parameters: {autotimm.count_parameters(backbone):,}")


if __name__ == "__main__":
    main()
```

---

## Running Examples

Clone the repository and run examples:

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[all]"

# Run an example
python examples/classify_cifar10.py
```
