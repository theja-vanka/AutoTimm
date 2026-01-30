# Object Detection Examples

This page demonstrates object detection tasks using AutoTimm.

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

## Running Object Detection Examples

Clone the repository and run examples:

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[all]"

# Run COCO detection example
python examples/object_detection_coco.py

# Run transformer-based detection
python examples/object_detection_transformers.py

# Run RT-DETR detection
python examples/object_detection_rtdetr.py
```
