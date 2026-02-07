"""Fine-tune a Vision Transformer with frozen backbone (linear probing).

This example demonstrates:
- Two-phase ViT fine-tuning (linear probe then full fine-tune)
- Using BackboneConfig for stochastic depth
- MetricManager for metric configuration

Usage:
    python examples/vit_finetuning.py
"""

from autotimm import (
    AutoTrainer,
    BackboneConfig,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Data
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

    # Configure metrics
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

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    print(f"Number of classes: {metric_manager.num_classes}")
    print(f"Configured metrics: {[c.name for c in metric_manager]}")

    # Phase 1: Linear probing -- freeze backbone, train only the head
    print("\n" + "=" * 60)
    print("Phase 1: Linear Probing (frozen backbone)")
    print("=" * 60)

    model = ImageClassifier(
        backbone=backbone_cfg,
        num_classes=data.num_classes,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=False,
        ),
        lr=1e-2,
        scheduler="cosine",
        freeze_backbone=True,
    )

    trainer = AutoTrainer(
        max_epochs=5,
        precision="bf16-mixed",
        logger=[
            LoggerConfig(
                backend="wandb",
                params={"project": "vit-finetune", "name": "linear-probe"},
            ),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    trainer.fit(model, datamodule=data)

    # Phase 2: Full fine-tuning -- unfreeze backbone, lower learning rate
    print("\n" + "=" * 60)
    print("Phase 2: Full Fine-Tuning (unfrozen backbone)")
    print("=" * 60)

    for param in model.backbone.parameters():
        param.requires_grad = True

    model._lr = 1e-4
    trainer = AutoTrainer(
        max_epochs=20,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        logger=[
            LoggerConfig(
                backend="wandb",
                params={"project": "vit-finetune", "name": "full-finetune"},
            ),
        ],
        checkpoint_monitor="val/accuracy",
        checkpoint_mode="max",
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
