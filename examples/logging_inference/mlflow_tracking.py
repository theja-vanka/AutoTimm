"""Train with MLflow experiment tracking.

This example demonstrates:
- MLflow integration for experiment tracking
- Training on CIFAR-100 with ResNet-50
- Top-k accuracy metrics
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
    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR100",
        image_size=224,
        batch_size=64,
        num_workers=4,
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
        MetricConfig(
            name="top5_accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "top_k": 5},
            stages=["val", "test"],
        ),
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=100)

    print(f"Configured {len(metric_manager)} metrics:")
    for config in metric_manager:
        print(f"  - {config.name}: stages={config.stages}")

    model = ImageClassifier(
        backbone="resnet50",
        num_classes=100,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        lr=1e-3,
        scheduler="cosine",
        head_dropout=0.2,
        label_smoothing=0.1,
        mixup_alpha=0.2,
    )

    # MLflow logs to ./mlruns by default.
    # Start the MLflow UI with: mlflow ui --port 5000
    trainer = AutoTrainer(
        max_epochs=20,
        precision="bf16-mixed",
        logger=[
            LoggerConfig(
                backend="mlflow",
                params={
                    "experiment_name": "cifar100-resnet50",
                    "tracking_uri": "file:./mlruns",
                },
            ),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
