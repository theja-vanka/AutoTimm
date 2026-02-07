"""Example: Image Classification with Hugging Face Hub Models.

This example shows how to use different HF Hub models for image classification,
including standard CNNs, Vision Transformers, and modern architectures.

Usage:
    python examples/hf_hub_classification.py
"""

from __future__ import annotations

import autotimm
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
    list_hf_hub_backbones,
)


def discover_classification_backbones():
    """Discover suitable backbones for classification on HF Hub."""
    print("=" * 80)
    print("Discovering Classification Backbones on HF Hub")
    print("=" * 80)

    # Popular CNN architectures
    print("\n1. ResNet models:")
    resnets = list_hf_hub_backbones(model_name="resnet", limit=5)
    for model in resnets:
        print(f"   {model}")

    # Vision Transformers
    print("\n2. Vision Transformer (ViT) models:")
    vits = list_hf_hub_backbones(model_name="vit", limit=5)
    for model in vits:
        print(f"   {model}")

    # EfficientNet models
    print("\n3. EfficientNet models:")
    efficientnets = list_hf_hub_backbones(model_name="efficientnet", limit=5)
    for model in efficientnets:
        print(f"   {model}")

    # ConvNeXt models
    print("\n4. ConvNeXt models:")
    convnexts = list_hf_hub_backbones(model_name="convnext", limit=5)
    for model in convnexts:
        print(f"   {model}")

    # MobileNet models (efficient for edge deployment)
    print("\n5. MobileNet models (for edge devices):")
    mobilenets = list_hf_hub_backbones(model_name="mobilenet", limit=5)
    for model in mobilenets:
        print(f"   {model}")


def compare_model_sizes():
    """Compare different HF Hub models by size and features."""
    print("\n" + "=" * 80)
    print("Comparing Model Sizes")
    print("=" * 80)

    models_to_compare = [
        "hf-hub:timm/mobilenetv3_small_100.lamb_in1k",  # Small, efficient
        "hf-hub:timm/resnet18.a1_in1k",  # Classic small CNN
        "hf-hub:timm/resnet50.a1_in1k",  # Classic medium CNN
        "hf-hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k",  # Small ViT
        "hf-hub:timm/convnext_tiny.fb_in22k",  # Modern small CNN
    ]

    print(f"\n{'Model':<55} {'Parameters':>15} {'Features':>10}")
    print("-" * 80)

    for model_name in models_to_compare:
        try:
            backbone = autotimm.create_backbone(model_name)
            params = autotimm.count_parameters(backbone, trainable_only=False)
            features = backbone.num_features

            # Extract short name for display
            short_name = model_name.replace("hf-hub:timm/", "")
            print(f"{short_name:<55} {params:>15,} {features:>10}")
        except Exception as e:
            print(f"{model_name:<55} Error: {e}")


def train_with_resnet_hf():
    """Train CIFAR-10 classifier with ResNet from HF Hub."""
    print("\n" + "=" * 80)
    print("Training with ResNet50 from HF Hub")
    print("=" * 80)

    # Data setup
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="top5_acc",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10, "top_k": 5},
            stages=["val", "test"],
            prog_bar=False,
        ),
    ]

    # Model using HF Hub ResNet50
    print("\nCreating ImageClassifier with HF Hub ResNet50...")
    model = ImageClassifier(
        backbone="hf-hub:timm/resnet50.a1_in1k",
        num_classes=10,
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
    )

    print(f"Total parameters: {autotimm.count_parameters(model):,}")

    # Trainer
    trainer = AutoTrainer(
        max_epochs=5,
        accelerator="auto",
        precision="16-mixed",
        logger_configs=[],
    )

    # Train
    print("\nTraining...")
    trainer.fit(model, datamodule=data)

    # Test
    print("\nTesting...")
    trainer.test(model, datamodule=data)


def train_with_vision_transformer():
    """Train CIFAR-10 classifier with Vision Transformer from HF Hub."""
    print("\n" + "=" * 80)
    print("Training with Vision Transformer from HF Hub")
    print("=" * 80)

    # Data setup
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=32,  # Smaller batch for ViT
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub Vision Transformer
    print("\nCreating ImageClassifier with HF Hub ViT...")
    model = ImageClassifier(
        backbone="hf-hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
        num_classes=10,
        metrics=metrics,
        lr=1e-4,  # Lower LR for transformers
        optimizer="adamw",
        weight_decay=0.05,  # Higher weight decay for transformers
    )

    print(f"Total parameters: {autotimm.count_parameters(model):,}")

    # Trainer with gradient clipping for transformers
    trainer = AutoTrainer(
        max_epochs=5,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        logger_configs=[],
    )

    # Train
    print("\nTraining...")
    trainer.fit(model, datamodule=data)

    # Test
    print("\nTesting...")
    trainer.test(model, datamodule=data)


def train_with_mobilenet_for_edge():
    """Train lightweight model for edge deployment."""
    print("\n" + "=" * 80)
    print("Training MobileNetV3 for Edge Deployment")
    print("=" * 80)

    # Data setup
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=128,  # Larger batch for small model
        num_workers=4,
    )

    # Metrics
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "num_classes": 10},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]

    # Model using HF Hub MobileNetV3
    print("\nCreating ImageClassifier with HF Hub MobileNetV3...")
    model = ImageClassifier(
        backbone="hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
        num_classes=10,
        metrics=metrics,
        lr=1e-3,
        optimizer="adamw",
    )

    print(f"Total parameters: {autotimm.count_parameters(model):,}")
    print("(Perfect for edge devices and mobile deployment!)")

    # Trainer
    trainer = AutoTrainer(
        max_epochs=10,
        accelerator="auto",
        precision="16-mixed",
        logger_configs=[],
    )

    # Train
    print("\nTraining...")
    trainer.fit(model, datamodule=data)

    # Test
    print("\nTesting...")
    trainer.test(model, datamodule=data)


def main():
    """Run all classification examples."""
    print("\n" + "=" * 80)
    print("HF Hub Classification Examples")
    print("=" * 80)

    # Example 1: Discover models
    discover_classification_backbones()

    # Example 2: Compare models
    compare_model_sizes()

    # Example 3: Train with different architectures (optional - uncomment to run)
    # print("\nNote: Training examples are commented out by default.")
    # print("Uncomment the desired training function in main() to run.\n")

    # Uncomment ONE of these to run training:
    # train_with_resnet_hf()
    # train_with_vision_transformer()
    # train_with_mobilenet_for_edge()

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. HF Hub provides access to hundreds of classification models")
    print("  2. Use smaller models (MobileNet, EfficientNet) for edge deployment")
    print("  3. Use ViT models for state-of-the-art accuracy")
    print("  4. ResNet/ConvNeXt offer a good balance")
    print("  5. All models work with the same AutoTimm API")


if __name__ == "__main__":
    main()
