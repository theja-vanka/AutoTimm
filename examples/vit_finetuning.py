"""Fine-tune a Vision Transformer with frozen backbone (linear probing)."""

from autotimm import BackboneConfig, ImageClassifier, ImageDataModule, create_trainer

data = ImageDataModule(
    data_dir="/path/to/your/dataset",
    image_size=224,
    batch_size=32,
    num_workers=4,
    augmentation_preset="randaugment",
)

data.setup("fit")

# Configure the backbone with stochastic depth
backbone_cfg = BackboneConfig(
    model_name="vit_base_patch16_224",
    pretrained=True,
    drop_path_rate=0.1,
)

# Phase 1: Linear probing -- freeze backbone, train only the head
model = ImageClassifier(
    backbone=backbone_cfg,
    num_classes=data.num_classes,
    lr=1e-2,
    scheduler="cosine",
    freeze_backbone=True,
)

trainer = create_trainer(
    max_epochs=5,
    precision="bf16-mixed",
    logger="wandb",
    logger_kwargs={"project": "vit-finetune", "name": "linear-probe"},
)

trainer.fit(model, datamodule=data)

# Phase 2: Full fine-tuning -- unfreeze backbone, lower learning rate
for param in model.backbone.parameters():
    param.requires_grad = True

model._lr = 1e-4
trainer = create_trainer(
    max_epochs=20,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    logger="wandb",
    logger_kwargs={"project": "vit-finetune", "name": "full-finetune"},
)

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
