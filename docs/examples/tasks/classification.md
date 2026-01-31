# Classification Examples

This page demonstrates image classification tasks using AutoTimm.

## CIFAR-10 Classification

Basic training with ResNet-18 on CIFAR-10 using MetricManager.

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
    )

    # Metrics
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    # Create MetricManager
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Model
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=10,
        metrics=metric_manager,
        lr=1e-3,
    )

    # Train
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
        checkpoint_monitor="val/accuracy",
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## Custom Folder Dataset

Training on your own images organized in folders.

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Your data structure:
    # dataset/
    #   train/
    #     class_a/...
    #     class_b/...
    #   val/
    #     class_a/...
    #     class_b/...

    data = ImageDataModule(
        data_dir="./dataset",
        image_size=384,
        batch_size=16,
    )
    data.setup("fit")

    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    model = ImageClassifier(
        backbone="efficientnet_b3",
        num_classes=data.num_classes,
        metrics=metric_manager,
    )

    trainer = AutoTrainer(
        max_epochs=20,
        logger=[LoggerConfig(backend="wandb", params={"project": "my-project"})],
        checkpoint_monitor="val/accuracy",
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## ViT Fine-Tuning

Two-phase fine-tuning for Vision Transformers with MetricManager.

```python
from autotimm import AutoTrainer, ImageClassifier, MetricConfig, MetricManager


def main():
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        ),
    ]

    metric_manager = MetricManager(configs=metric_configs, num_classes=data.num_classes)

    # Phase 1: Linear probe (frozen backbone)
    model = ImageClassifier(
        backbone="vit_base_patch16_224",
        num_classes=data.num_classes,
        metrics=metric_manager,
        freeze_backbone=True,
        lr=1e-2,
    )

    trainer = AutoTrainer(max_epochs=5)
    trainer.fit(model, datamodule=data)

    # Phase 2: Full fine-tune
    for param in model.backbone.parameters():
        param.requires_grad = True

    model._lr = 1e-4
    trainer = AutoTrainer(max_epochs=20, gradient_clip_val=1.0)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## Running Classification Examples

Clone the repository and run examples:

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[all]"

# Run CIFAR-10 example
python examples/classify_cifar10.py

# Run custom folder example
python examples/classify_custom_folder.py

# Run ViT fine-tuning example
python examples/vit_finetuning.py
```
