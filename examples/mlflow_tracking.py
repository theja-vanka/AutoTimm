"""Train with MLflow experiment tracking."""

from autotimm import ImageClassifier, ImageDataModule, create_trainer

data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR100",
    image_size=224,
    batch_size=64,
    num_workers=4,
)

model = ImageClassifier(
    backbone="resnet50",
    num_classes=100,
    lr=1e-3,
    scheduler="cosine",
    head_dropout=0.2,
    label_smoothing=0.1,
    mixup_alpha=0.2,
)

# MLflow logs to ./mlruns by default.
# Start the MLflow UI with: mlflow ui --port 5000
trainer = create_trainer(
    max_epochs=20,
    precision="bf16-mixed",
    logger="mlflow",
    logger_kwargs={
        "experiment_name": "cifar100-resnet50",
        "tracking_uri": "file:./mlruns",
    },
    accumulate_grad_batches=2,
    gradient_clip_val=1.0,
)

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
