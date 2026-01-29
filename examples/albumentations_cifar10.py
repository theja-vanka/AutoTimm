"""Train on CIFAR-10 using albumentations transforms (OpenCV backend).

This example demonstrates:
- Using albumentations for strong augmentations
- Built-in dataset support with albumentations
- MetricManager for metric configuration
"""

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
    # Use albumentations with the "strong" augmentation preset.
    # Built-in datasets (CIFAR10, etc.) automatically convert PIL → numpy
    # so albumentations pipelines work seamlessly.
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
        num_workers=4,
        transform_backend="albumentations",
        augmentation_preset="strong",
    )

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
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    print(f"Configured {len(metric_manager)} metrics:")
    for config in metric_manager:
        print(f"  - {config.name}: stages={config.stages}")

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=False,
        ),
        lr=1e-3,
        scheduler="cosine",
    )

    trainer = AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "lightning_logs"}),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
