"""Train a ResNet-18 on CIFAR-10 with TensorBoard logging.

This example demonstrates:
- Explicit metric configuration with MetricManager
- Enhanced logging (learning rate, gradient norms)
- Optional automatic LR and batch size finding
"""

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
    MetricManager,
    TunerConfig,
)


def main():
    # 1. Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,  # Will be overridden if auto_batch_size=True
        num_workers=4,
    )

    # 2. Configure metrics explicitly
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

    # Create MetricManager for programmatic access
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # You can access metrics by name
    print(f"Configured metrics: {[c.name for c in metric_manager]}")

    # 3. Model with explicit configuration
    # Note: lr is set here but can be overridden by the LR finder
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
        ),
        lr=1e-3,  # Initial LR (will be tuned if auto_lr=True)
        scheduler="cosine",
    )

    # 4. Train with automatic hyperparameter tuning
    # Option A: Use automatic LR and batch size finding (recommended for new projects)
    trainer = AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "lightning_logs"}),
        ],
        tuner_config=TunerConfig(
            auto_lr=True,  # Find optimal learning rate before training
            auto_batch_size=False,  # Set to True to also find optimal batch size
            lr_find_kwargs={
                "min_lr": 1e-6,
                "max_lr": 1.0,
                "num_training": 100,
            },
        ),
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    # Option B: Use manual hyperparameters (comment out tuner_config above)
    # trainer = AutoTrainer(
    #     max_epochs=10,
    #     accelerator="auto",
    #     logger=[
    #         LoggerConfig(backend="tensorboard", params={"save_dir": "lightning_logs"}),
    #     ],
    #     checkpoint_monitor="val/accuracy",
    #     checkpoint_mode="max",
    # )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
