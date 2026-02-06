"""Demonstrate using multiple logging backends simultaneously.

This example demonstrates:
- Configuring multiple loggers (TensorBoard + CSV)
- Using LoggerManager for centralized logger configuration
- Using MetricManager for metric access
- Logging to multiple destinations in parallel

Usage:
    python examples/multiple_loggers.py
"""

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggerManager,
    LoggingConfig,
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
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        lr=1e-3,
        scheduler="cosine",
    )

    # Option 1: Use LoggerManager for multiple loggers
    logger_manager = LoggerManager(
        configs=[
            LoggerConfig(
                backend="tensorboard",
                params={"save_dir": "logs/tensorboard", "name": "cifar10_run"},
            ),
            LoggerConfig(
                backend="csv",
                params={"save_dir": "logs/csv", "name": "cifar10_run"},
            ),
        ]
    )

    trainer = AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        logger=logger_manager,
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    # Option 2: Pass list of LoggerConfig directly (equivalent)
    # trainer = AutoTrainer(
    #     max_epochs=10,
    #     accelerator="auto",
    #     logger=[
    #         LoggerConfig(backend="tensorboard", params={"save_dir": "logs/tensorboard"}),
    #         LoggerConfig(backend="csv", params={"save_dir": "logs/csv"}),
    #     ],
    #     checkpoint_monitor="val/accuracy",
    #     checkpoint_mode="max",
    # )

    # Option 3: Add W&B alongside TensorBoard (if wandb installed)
    # trainer = AutoTrainer(
    #     max_epochs=10,
    #     logger=[
    #         LoggerConfig(backend="tensorboard", params={"save_dir": "logs/tb"}),
    #         LoggerConfig(backend="wandb", params={"project": "cifar10", "name": "run_1"}),
    #     ],
    # )

    print("Training with multiple loggers...")
    print("  - TensorBoard: logs/tensorboard/")
    print("  - CSV: logs/csv/")

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)

    print("\nTraining complete!")
    print("View TensorBoard logs: tensorboard --logdir logs/tensorboard")
    print("View CSV logs: cat logs/csv/cifar10_run/metrics.csv")


if __name__ == "__main__":
    main()
