"""Train a ResNet-18 on CIFAR-10 with TensorBoard logging."""

from autotimm import ImageClassifier, ImageDataModule, create_trainer

# 1. Data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=64,
    num_workers=4,
)

# 2. Model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    lr=1e-3,
    scheduler="cosine",
)

# 3. Train
trainer = create_trainer(
    max_epochs=10,
    accelerator="auto",
    logger="tensorboard",
)

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
