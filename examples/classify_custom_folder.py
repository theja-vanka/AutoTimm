"""Train on a custom ImageFolder dataset with W&B logging."""

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
)

# 1. Data -- expects data_dir/train/<class>/*.jpg and data_dir/val/<class>/*.jpg
data = ImageDataModule(
    data_dir="/path/to/your/dataset",
    image_size=384,
    batch_size=16,
    num_workers=4,
)

# 2. Set up the data to discover num_classes
data.setup("fit")

# 3. Configure metrics
metrics = [
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
        stages=["val"],
    ),
    MetricConfig(
        name="recall",
        backend="torchmetrics",
        metric_class="Recall",
        params={"task": "multiclass", "average": "macro"},
        stages=["val"],
    ),
]

# 4. Model
model = ImageClassifier(
    backbone="efficientnet_b3",
    num_classes=data.num_classes,
    metrics=metrics,
    logging_config=LoggingConfig(
        log_learning_rate=True,
        log_gradient_norm=True,
    ),
    lr=3e-4,
    scheduler="cosine",
    label_smoothing=0.1,
    mixup_alpha=0.2,
)

# 5. Train with W&B logging and mixed precision
trainer = AutoTrainer(
    max_epochs=20,
    precision="bf16-mixed",
    logger=[
        LoggerConfig(backend="wandb", params={"project": "my-project"}),
    ],
    checkpoint_monitor="val/accuracy",
    checkpoint_mode="max",
)

trainer.fit(model, datamodule=data)
