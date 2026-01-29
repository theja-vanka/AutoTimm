"""Handle class-imbalanced datasets with weighted random sampling."""

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
)

# balanced_sampling=True uses WeightedRandomSampler to oversample
# minority classes, so every class is seen roughly equally per epoch.
data = ImageDataModule(
    data_dir="/path/to/imbalanced/dataset",
    image_size=224,
    batch_size=32,
    num_workers=4,
    balanced_sampling=True,
    persistent_workers=True,
)

data.setup("fit")

# Print class distribution to verify imbalance
print(data.summary())

# Configure metrics - include per-class metrics for imbalanced datasets
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
        name="f1_macro",
        backend="torchmetrics",
        metric_class="F1Score",
        params={"task": "multiclass", "average": "macro"},
        stages=["val", "test"],
    ),
    MetricConfig(
        name="f1_weighted",
        backend="torchmetrics",
        metric_class="F1Score",
        params={"task": "multiclass", "average": "weighted"},
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

model = ImageClassifier(
    backbone="efficientnet_b0",
    num_classes=data.num_classes,
    metrics=metrics,
    logging_config=LoggingConfig(
        log_learning_rate=True,
        log_gradient_norm=False,
        log_confusion_matrix=True,  # Useful for imbalanced datasets
    ),
    lr=1e-3,
    scheduler="cosine",
    label_smoothing=0.1,
)

trainer = AutoTrainer(
    max_epochs=30,
    logger=[
        LoggerConfig(backend="tensorboard", params={"save_dir": "lightning_logs"}),
    ],
    checkpoint_monitor="val/f1_macro",  # Use F1 for imbalanced datasets
    checkpoint_mode="max",
)

trainer.fit(model, datamodule=data)
