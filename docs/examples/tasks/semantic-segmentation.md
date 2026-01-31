# Semantic Segmentation Examples

Complete examples for training semantic segmentation models with AutoTimm.

## Basic Example: Cityscapes

Train DeepLabV3+ on Cityscapes dataset for urban scene segmentation.

```python
from autotimm import (
    AutoTrainer,
    SemanticSegmentor,
    SegmentationDataModule,
    MetricConfig,
    LoggerConfig,
    LoggingConfig,
)


def main():
    # Data - Cityscapes with 19 classes
    data = SegmentationDataModule(
        data_dir="./cityscapes",
        format="cityscapes",
        image_size=512,
        batch_size=8,
        num_workers=4,
        augmentation_preset="default",
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mIoU",
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
        MetricConfig(
            name="pixel_acc",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={
                "task": "multiclass",
                "num_classes": 19,
                "ignore_index": 255,
            },
            stages=["val", "test"],
        ),
    ]

    # Model - DeepLabV3+ with ResNet-50
    model = SemanticSegmentor(
        backbone="resnet50",
        num_classes=19,
        head_type="deeplabv3plus",
        loss_type="combined",  # CE + Dice
        ce_weight=1.0,
        dice_weight=1.0,
        ignore_index=255,
        metrics=metrics,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="cosine",
    )

    # Trainer
    trainer = AutoTrainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs/cityscapes"})],
        checkpoint_monitor="val/mIoU",
        checkpoint_mode="max",
    )

    # Train
    trainer.fit(model, datamodule=data)

    # Test
    results = trainer.test(model, datamodule=data)
    print(f"Test mIoU: {results[0]['test/mIoU']:.4f}")


if __name__ == "__main__":
    main()
```

## Pascal VOC Example

Train on Pascal VOC 2012 with 21 classes (20 objects + background).

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig, AutoTrainer, LoggerConfig


def main():
    # Data
    data = SegmentationDataModule(
        data_dir="./VOC2012",
        format="voc",
        image_size=512,
        batch_size=16,
        num_workers=4,
        augmentation_preset="strong",
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": 21,
                "average": "macro",
                "ignore_index": 255,
            },
            stages=["val", "test"],
            prog_bar=True,
        ),
    ]

    # Model - FCN baseline
    model = SemanticSegmentor(
        backbone="resnet50",
        num_classes=21,
        head_type="fcn",  # Simpler architecture
        loss_type="combined",
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
        scheduler="cosine",
    )

    # Trainer
    trainer = AutoTrainer(
        max_epochs=100,
        logger=[LoggerConfig(backend="tensorboard")],
    )

    # Train
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

## Custom Dataset Example

Train on a custom dataset with PNG masks.

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig, AutoTrainer


def main():
    # Custom dataset with 5 classes (0-4) + ignore (255)
    data = SegmentationDataModule(
        data_dir="./custom_dataset",
        format="png",  # Uses images/ and masks/ folders
        image_size=512,
        batch_size=8,
        augmentation_preset="default",
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": 5,
                "average": "macro",
                "ignore_index": 255,
            },
            stages=["val"],
            prog_bar=True,
        ),
    ]

    # Model
    model = SemanticSegmentor(
        backbone="resnet18",  # Lighter backbone for small dataset
        num_classes=5,
        head_type="deeplabv3plus",
        loss_type="dice",  # Dice only for class imbalance
        metrics=metrics,
    )

    # Trainer
    trainer = AutoTrainer(max_epochs=50)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

## Custom Transforms Example

Use albumentations for advanced augmentation.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig, AutoTrainer


def get_train_transforms():
    return A.Compose([
        A.RandomScale(scale_limit=0.5, p=1.0),
        A.RandomCrop(height=512, width=512, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def main():
    # Data with custom transforms
    data = SegmentationDataModule(
        data_dir="./data",
        format="png",
        custom_train_transforms=get_train_transforms(),
        custom_val_transforms=get_val_transforms(),
        batch_size=8,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={"task": "multiclass", "num_classes": 10, "average": "macro"},
            stages=["val"],
            prog_bar=True,
        ),
    ]

    # Model
    model = SemanticSegmentor(
        backbone="efficientnet_b3",
        num_classes=10,
        head_type="deeplabv3plus",
        loss_type="combined",
        metrics=metrics,
    )

    # Trainer
    trainer = AutoTrainer(max_epochs=100)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

## Inference Example

Load a trained model and run inference.

```python
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np
from autotimm import SemanticSegmentor


def visualize_segmentation(image, prediction, num_classes):
    """Visualize segmentation results."""
    # Create color map
    colors = plt.cm.get_cmap('tab20', num_classes)

    # Create colored mask
    colored_mask = np.zeros((*prediction.shape, 3))
    for class_id in range(num_classes):
        mask = prediction == class_id
        colored_mask[mask] = colors(class_id)[:3]

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation")
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(colored_mask, alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("segmentation_result.png")
    plt.show()


def main():
    # Load model
    model = SemanticSegmentor.load_from_checkpoint("best_model.ckpt")
    model.eval()

    # Load and preprocess image
    image = Image.open("test_image.jpg")
    original_size = image.size

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    image_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        prediction = model.predict(image_tensor)

    # prediction shape: [1, H, W]
    prediction = prediction[0].cpu().numpy()

    # Resize to original size
    prediction_pil = Image.fromarray(prediction.astype(np.uint8))
    prediction_resized = prediction_pil.resize(original_size, Image.NEAREST)
    prediction = np.array(prediction_resized)

    # Visualize
    visualize_segmentation(image, prediction, num_classes=19)


if __name__ == "__main__":
    main()
```

## Using Swin Transformer

Use Vision Transformer backbone for better accuracy.

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig, AutoTrainer


def main():
    # Data
    data = SegmentationDataModule(
        data_dir="./cityscapes",
        format="cityscapes",
        image_size=512,
        batch_size=4,  # Smaller batch for transformer
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="mIoU",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={
                "task": "multiclass",
                "num_classes": 19,
                "average": "macro",
                "ignore_index": 255,
            },
            stages=["val"],
            prog_bar=True,
        ),
    ]

    # Model - Swin Transformer
    model = SemanticSegmentor(
        backbone="swin_tiny_patch4_window7_224",
        num_classes=19,
        head_type="deeplabv3plus",
        loss_type="combined",
        metrics=metrics,
        lr=1e-4,
    )

    # Trainer with mixed precision
    trainer = AutoTrainer(
        max_epochs=200,
        precision="16-mixed",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

## Comparing Losses

Compare different loss functions.

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, MetricConfig, AutoTrainer, LoggerConfig


def train_with_loss(loss_type, run_name):
    """Train model with specific loss type."""
    data = SegmentationDataModule(
        data_dir="./data",
        format="png",
        image_size=512,
        batch_size=8,
    )

    metrics = [
        MetricConfig(
            name="iou",
            backend="torchmetrics",
            metric_class="JaccardIndex",
            params={"task": "multiclass", "num_classes": 10, "average": "macro"},
            stages=["val"],
            prog_bar=True,
        ),
    ]

    model = SemanticSegmentor(
        backbone="resnet50",
        num_classes=10,
        head_type="deeplabv3plus",
        loss_type=loss_type,  # "ce", "dice", "focal", or "combined"
        metrics=metrics,
    )

    trainer = AutoTrainer(
        max_epochs=50,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": f"logs/{run_name}"})],
    )

    trainer.fit(model, datamodule=data)
    
    # Only run test if test set exists
    try:
        results = trainer.test(model, datamodule=data)
        return results[0]['test/iou']
    except:
        # Return validation IoU if test set doesn't exist
        return trainer.callback_metrics.get('val/iou', 0.0).item()


def main():
    # Compare losses
    results = {}

    results['ce'] = train_with_loss("ce", "ce_loss")
    results['dice'] = train_with_loss("dice", "dice_loss")
    results['focal'] = train_with_loss("focal", "focal_loss")
    results['combined'] = train_with_loss("combined", "combined_loss")

    print("\nResults:")
    for loss_type, iou in results.items():
        print(f"{loss_type}: {iou:.4f}")


if __name__ == "__main__":
    main()
```

## Using Import Aliases

Cleaner imports with submodule aliases:

```python
from autotimm.task import SemanticSegmentor
from autotimm.loss import DiceLoss, CombinedSegmentationLoss
from autotimm.head import DeepLabV3PlusHead
from autotimm.metric import MetricConfig


def main():
    # Can also directly instantiate losses
    dice_loss = DiceLoss(num_classes=19, ignore_index=255)

    # Model using alias imports
    model = SemanticSegmentor(
        backbone="resnet50",
        num_classes=19,
        head_type="deeplabv3plus",
        loss_type="combined",
        metrics=[
            MetricConfig(
                name="iou",
                backend="torchmetrics",
                metric_class="JaccardIndex",
                params={"task": "multiclass", "num_classes": 19, "average": "macro"},
                stages=["val"],
                prog_bar=True,
            ),
        ],
    )


if __name__ == "__main__":
    main()
```

## See Also

- [Semantic Segmentation Guide](../user-guide/models/semantic-segmentation.md)
- [Data Loading Guide](../user-guide/data-loading/segmentation-data.md)
- [API Reference](../api/segmentation.md)
