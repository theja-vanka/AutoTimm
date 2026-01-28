"""Train on CIFAR-10 using albumentations transforms (OpenCV backend)."""

from autotimm import ImageClassifier, ImageDataModule, create_trainer

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

model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    lr=1e-3,
    scheduler="cosine",
)

trainer = create_trainer(
    max_epochs=10,
    accelerator="auto",
    logger="tensorboard",
)

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
