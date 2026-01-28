"""Handle class-imbalanced datasets with weighted random sampling."""

from autotimm import ImageClassifier, ImageDataModule, create_trainer

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

model = ImageClassifier(
    backbone="efficientnet_b0",
    num_classes=data.num_classes,
    lr=1e-3,
    scheduler="cosine",
    label_smoothing=0.1,
)

trainer = create_trainer(
    max_epochs=30,
    logger="tensorboard",
)

trainer.fit(model, datamodule=data)
