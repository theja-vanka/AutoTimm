"""Train on a custom folder dataset using albumentations + OpenCV.

When transform_backend="albumentations" is used with a folder dataset,
images are loaded with OpenCV (cv2.imread) instead of PIL, and passed
as RGB numpy arrays directly to the albumentations pipeline.

This example demonstrates:
- Custom albumentations pipeline definition
- Using albumentations with folder datasets
- MetricManager for metric configuration
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Define a custom albumentations pipeline
    custom_train = A.Compose(
        [
            A.RandomResizedCrop(size=(224, 224)),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-20, 20), scale=(0.8, 1.2), p=0.5),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(std_range=(0.02, 0.05)),
                ],
                p=0.3,
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    data = ImageDataModule(
        data_dir="/path/to/your/dataset",
        image_size=224,
        batch_size=32,
        num_workers=4,
        transform_backend="albumentations",
        train_transforms=custom_train,  # overrides the default albumentations preset
    )

    data.setup("fit")
    print(data.summary())

    # Configure metrics
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
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    print(f"\nConfigured {len(metric_manager)} metrics:")
    for config in metric_manager:
        print(f"  - {config.name}: stages={config.stages}")

    model = ImageClassifier(
        backbone="convnext_tiny",
        num_classes=data.num_classes,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        lr=3e-4,
        scheduler="cosine",
        label_smoothing=0.1,
    )

    trainer = AutoTrainer(
        max_epochs=30,
        precision="bf16-mixed",
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "lightning_logs"}),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
