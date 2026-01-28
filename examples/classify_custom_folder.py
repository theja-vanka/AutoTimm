"""Train on a custom ImageFolder dataset with W&B logging."""

from autotimm import ImageClassifier, ImageDataModule, create_trainer

# 1. Data -- expects data_dir/train/<class>/*.jpg and data_dir/val/<class>/*.jpg
data = ImageDataModule(
    data_dir="/path/to/your/dataset",
    image_size=384,
    batch_size=16,
    num_workers=4,
)

# 2. Set up the data to discover num_classes
data.setup("fit")

# 3. Model
model = ImageClassifier(
    backbone="efficientnet_b3",
    num_classes=data.num_classes,
    lr=3e-4,
    scheduler="cosine",
    label_smoothing=0.1,
    mixup_alpha=0.2,
)

# 4. Train with W&B logging and mixed precision
trainer = create_trainer(
    max_epochs=20,
    precision="bf16-mixed",
    logger="wandb",
    logger_kwargs={"project": "my-project"},
)

trainer.fit(model, datamodule=data)
