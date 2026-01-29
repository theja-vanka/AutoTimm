"""Demonstrate using timm metrics alongside torchmetrics.

This example shows how to:
- Configure multiple metrics from different backends
- Use timm's accuracy function (top-1 and top-5)
- Combine torchmetrics and timm metrics in one training run
"""

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
)

# Data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR100",  # 100 classes - good for top-5 accuracy
    image_size=224,
    batch_size=64,
    num_workers=4,
)

# Configure metrics from multiple backends
metrics = [
    # Torchmetrics - standard accuracy
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
    # Torchmetrics - top-5 accuracy
    MetricConfig(
        name="top5_accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass", "top_k": 5},
        stages=["val", "test"],
    ),
    # Torchmetrics - F1 Score
    MetricConfig(
        name="f1",
        backend="torchmetrics",
        metric_class="F1Score",
        params={"task": "multiclass", "average": "macro"},
        stages=["val", "test"],
    ),
    # Timm - accuracy (uses logits directly, returns top-k tuple)
    MetricConfig(
        name="timm_acc",
        backend="timm",
        metric_class="accuracy",
        params={"topk": (1, 5)},  # timm returns tuple of top-k accuracies
        stages=["val"],
    ),
]

# Model with comprehensive metrics
model = ImageClassifier(
    backbone="resnet34",
    num_classes=100,
    metrics=metrics,
    logging_config=LoggingConfig(
        log_learning_rate=True,
        log_gradient_norm=False,
    ),
    lr=1e-3,
    scheduler="cosine",
    label_smoothing=0.1,
)

# Trainer with TensorBoard logging
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
