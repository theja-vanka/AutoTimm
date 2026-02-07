"""Example: Advanced Usage of Hugging Face Hub Models.

This example demonstrates advanced patterns for using HF Hub models including:
- Transfer learning from different pretraining datasets
- Comparing model families and architectures
- Custom model configurations
- Model ensembling
- Production deployment considerations

Usage:
    python examples/hf_hub_advanced.py
"""

from __future__ import annotations

import torch

import autotimm


def explore_pretraining_datasets():
    """Explore models pretrained on different datasets."""
    print("=" * 80)
    print("Exploring Different Pretraining Datasets")
    print("=" * 80)

    # Models with different pretraining
    pretraining_examples = [
        ("ImageNet-1k", "hf-hub:timm/resnet50.a1_in1k"),
        ("ImageNet-21k", "hf-hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k"),
        ("ImageNet-22k", "hf-hub:timm/convnext_base.fb_in22k_ft_in1k"),
        ("Semi-supervised", "hf-hub:timm/resnet50.fb_swsl_ig1b_ft_in1k"),
    ]

    print("\nDifferent pretraining approaches:\n")
    for name, model_id in pretraining_examples:
        print(f"{name}:")
        print(f"  Model: {model_id}")
        short_name = model_id.replace("hf-hub:timm/", "")

        # Extract pretraining info from model name
        if "in1k" in short_name:
            print("  Dataset: ImageNet-1k (1.28M images, 1k classes)")
        elif "in21k" in short_name:
            print("  Dataset: ImageNet-21k (14M images, 21k classes)")
        elif "in22k" in short_name:
            print("  Dataset: ImageNet-22k (14M images, 22k classes)")
        elif "swsl" in short_name:
            print("  Dataset: Instagram (940M images, semi-supervised)")

        print(
            f"  Benefit: {'Larger dataset → better features' if 'in21k' in short_name or 'in22k' in short_name or 'swsl' in short_name else 'Standard baseline'}"
        )
        print()

    print("Key Insights:")
    print("  • Models pretrained on larger datasets (21k/22k) often transfer better")
    print("  • Semi-supervised models have seen billions of images")
    print("  • Choose based on target task similarity to pretraining data")


def compare_architecture_families():
    """Compare different architecture families available on HF Hub."""
    print("\n" + "=" * 80)
    print("Comparing Architecture Families")
    print("=" * 80)

    families = [
        {
            "name": "ResNet",
            "model": "hf-hub:timm/resnet50.a1_in1k",
            "year": "2015",
            "features": "Skip connections, batch norm",
            "strengths": "Stable, proven, fast training",
            "weaknesses": "Older design, limited capacity",
        },
        {
            "name": "EfficientNet",
            "model": "hf-hub:timm/efficientnet_b2.ra_in1k",
            "year": "2019",
            "features": "Compound scaling, SE blocks",
            "strengths": "Efficient, great accuracy/params ratio",
            "weaknesses": "Slower than ResNet",
        },
        {
            "name": "Vision Transformer",
            "model": "hf-hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
            "year": "2020",
            "features": "Self-attention, patches",
            "strengths": "Scalable, strong with large data",
            "weaknesses": "Needs more data/compute",
        },
        {
            "name": "ConvNeXt",
            "model": "hf-hub:timm/convnext_tiny.fb_in22k",
            "year": "2022",
            "features": "Modern CNN design, LayerNorm",
            "strengths": "Best of CNNs and Transformers",
            "weaknesses": "Relatively new",
        },
    ]

    print("\nArchitecture Comparison:\n")
    for family in families:
        print(f"{family['name']} ({family['year']}):")
        print(f"  Model: {family['model']}")
        print(f"  Key Features: {family['features']}")
        print(f"  Strengths: {family['strengths']}")
        print(f"  Weaknesses: {family['weaknesses']}")
        print()


def benchmark_inference_speed():
    """Benchmark inference speed of different HF Hub models."""
    print("\n" + "=" * 80)
    print("Inference Speed Benchmark")
    print("=" * 80)

    models_to_test = [
        ("MobileNetV3", "hf-hub:timm/mobilenetv3_small_100.lamb_in1k"),
        ("ResNet18", "hf-hub:timm/resnet18.a1_in1k"),
        ("ResNet50", "hf-hub:timm/resnet50.a1_in1k"),
        ("EfficientNet-B0", "hf-hub:timm/efficientnet_b0.ra_in1k"),
        ("ConvNeXt-Tiny", "hf-hub:timm/convnext_tiny.fb_in22k"),
    ]

    print("\nMeasuring inference time on CPU (batch_size=1, image_size=224)...\n")
    print(f"{'Model':<25} {'Parameters':>15} {'Avg Time (ms)':>15}")
    print("-" * 60)

    import time

    for name, model_id in models_to_test:
        try:
            # Create model
            backbone = autotimm.create_backbone(model_id)
            backbone.eval()
            params = autotimm.count_parameters(backbone, trainable_only=False)

            # Warmup
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.inference_mode():
                for _ in range(5):
                    _ = backbone(dummy_input)

            # Benchmark
            num_runs = 20
            start = time.time()
            with torch.inference_mode():
                for _ in range(num_runs):
                    _ = backbone(dummy_input)
            avg_time = (time.time() - start) / num_runs * 1000  # ms

            print(f"{name:<25} {params:>15,} {avg_time:>15.2f}")
        except Exception as e:
            print(f"{name:<25} Error: {e}")

    print(
        "\nNote: Results vary by hardware. Run on target device for accurate metrics."
    )


def demonstrate_transfer_learning_strategies():
    """Demonstrate different transfer learning strategies."""
    print("\n" + "=" * 80)
    print("Transfer Learning Strategies")
    print("=" * 80)

    strategies = [
        {
            "name": "Fine-tune All Layers",
            "description": "Train all parameters",
            "code": """
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
    lr=1e-3,
)
# All layers trainable by default
            """,
            "when": "Large dataset (>10k images), dissimilar to ImageNet",
        },
        {
            "name": "Fine-tune Top Layers",
            "description": "Freeze early layers, train later ones",
            "code": """
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
    lr=1e-4,
)
# Freeze backbone bottom layers
for name, param in model.backbone.named_parameters():
    if 'layer1' in name or 'layer2' in name:
        param.requires_grad = False
            """,
            "when": "Medium dataset (1k-10k images), similar to ImageNet",
        },
        {
            "name": "Feature Extraction",
            "description": "Freeze all backbone, train only head",
            "code": """
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
    lr=1e-2,
)
# Freeze entire backbone
for param in model.backbone.parameters():
    param.requires_grad = False
            """,
            "when": "Small dataset (<1k images), very similar to ImageNet",
        },
        {
            "name": "Two-Stage Training",
            "description": "Feature extraction first, then fine-tuning",
            "code": """
# Stage 1: Train head only (10 epochs)
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    num_classes=10,
    lr=1e-2,
)
for param in model.backbone.parameters():
    param.requires_grad = False
trainer.fit(model, datamodule=data, max_epochs=10)

# Stage 2: Fine-tune all layers (20 epochs)
for param in model.backbone.parameters():
    param.requires_grad = True
model.lr = 1e-4  # Lower learning rate
trainer.fit(model, datamodule=data, max_epochs=20)
            """,
            "when": "Any size dataset, best accuracy needed",
        },
    ]

    print("\nCommon transfer learning approaches:\n")
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy['name']}")
        print(f"   Strategy: {strategy['description']}")
        print(f"   Best for: {strategy['when']}")
        print(f"   Example code:{strategy['code']}")
        print()


def demonstrate_model_versioning():
    """Show how to use specific model versions from HF Hub."""
    print("\n" + "=" * 80)
    print("Model Versioning with HF Hub")
    print("=" * 80)

    print("\nHF Hub provides versioned models with detailed metadata:\n")

    examples = [
        {
            "model": "hf-hub:timm/resnet50.a1_in1k",
            "description": "ResNet50 trained with recipe 'a1' on ImageNet-1k",
            "benefit": "Reproducible, documented training procedure",
        },
        {
            "model": "hf-hub:timm/resnet50.a2_in1k",
            "description": "ResNet50 trained with recipe 'a2' (different hyperparams)",
            "benefit": "Compare training recipes, choose best for your task",
        },
        {
            "model": "hf-hub:timm/resnet50.fb_swsl_ig1b_ft_in1k",
            "description": "Facebook's semi-supervised model, fine-tuned on ImageNet",
            "benefit": "Better initialization from 1B Instagram images",
        },
    ]

    for ex in examples:
        print(f"Model: {ex['model']}")
        print(f"  What: {ex['description']}")
        print(f"  Why: {ex['benefit']}")
        print()

    print("Benefits of versioning:")
    print("  • Reproducibility: Exact model and weights")
    print("  • Experimentation: Compare different training recipes")
    print("  • Documentation: Model cards with training details")
    print("  • Updates: Access improved versions without breaking code")


def demonstrate_custom_configurations():
    """Show advanced model configurations with HF Hub models."""
    print("\n" + "=" * 80)
    print("Custom Model Configurations")
    print("=" * 80)

    print("\nAdvanced configuration examples:\n")

    # Example 1: Custom dropout
    print("1. Custom dropout rates:")
    print("""
from autotimm.backbone import BackboneConfig

config = BackboneConfig(
    model_name="hf-hub:timm/resnet50.a1_in1k",
    pretrained=True,
    drop_rate=0.3,        # Higher dropout for regularization
    drop_path_rate=0.1,   # Stochastic depth
)
backbone = autotimm.create_backbone(config)
    """)

    # Example 2: Different number of classes
    print("\n2. Fine-grained classification (100 classes):")
    print("""
model = ImageClassifier(
    backbone="hf-hub:timm/convnext_tiny.fb_in22k",
    num_classes=100,  # Fine-grained dataset
    lr=5e-4,
    optimizer="adamw",
    weight_decay=0.05,
)
    """)

    # Example 3: Multi-task learning
    print("\n3. Custom head for multi-task learning:")
    print("""
# Create backbone
backbone = autotimm.create_backbone("hf-hub:timm/resnet50.a1_in1k")

# Custom multi-task head (implement yourself)
class MultiTaskHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.classifier = nn.Linear(in_features, 10)
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        return {
            'class': self.classifier(x),
            'value': self.regressor(x),
        }
    """)


def create_model_selection_guide():
    """Comprehensive guide for selecting HF Hub models."""
    print("\n" + "=" * 80)
    print("Model Selection Decision Tree")
    print("=" * 80)

    print("""
Choose your model based on these criteria:

1. DATASET SIZE:
   Small (<1k images)     → hf-hub:timm/mobilenetv3_small_100.lamb_in1k
   Medium (1k-10k)        → hf-hub:timm/resnet50.a1_in1k
   Large (10k-100k)       → hf-hub:timm/convnext_small.fb_in22k_ft_in1k
   Very Large (>100k)     → hf-hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k

2. DEPLOYMENT TARGET:
   Mobile/Edge            → hf-hub:timm/mobilenetv3_small_100.lamb_in1k
   Web Browser            → hf-hub:timm/efficientnet_b0.ra_in1k
   Server/Cloud           → hf-hub:timm/resnet50.a1_in1k
   Research/Offline       → hf-hub:timm/convnext_base.fb_in22k_ft_in1k

3. ACCURACY REQUIREMENT:
   Good enough            → hf-hub:timm/resnet18.a1_in1k
   Production quality     → hf-hub:timm/resnet50.a1_in1k
   State-of-the-art       → hf-hub:timm/convnext_large.fb_in22k_ft_in1k
   Research benchmark     → hf-hub:timm/vit_large_patch16_224.augreg_in21k_ft_in1k

4. INFERENCE SPEED:
   Real-time (>30 FPS)    → hf-hub:timm/mobilenetv3_large_100.ra_in1k
   Fast (10-30 FPS)       → hf-hub:timm/efficientnet_b2.ra_in1k
   Moderate (5-10 FPS)    → hf-hub:timm/resnet50.a1_in1k
   Slow (<5 FPS)          → hf-hub:timm/convnext_xlarge.fb_in22k_ft_in1k

5. DOMAIN:
   Natural images         → hf-hub:timm/resnet50.a1_in1k
   Medical imaging        → hf-hub:timm/resnet50.fb_swsl_ig1b_ft_in1k (better features)
   Fine-grained           → hf-hub:timm/convnext_small.fb_in22k_ft_in1k (22k pretraining)
   Document/Text          → hf-hub:timm/vit_small_patch16_224.augreg_in21k_ft_in1k
    """)


def main():
    """Run all advanced examples."""
    print("\n" + "=" * 80)
    print("Advanced HF Hub Usage Examples")
    print("=" * 80)

    # Example 1: Pretraining datasets
    explore_pretraining_datasets()

    # Example 2: Architecture families
    compare_architecture_families()

    # Example 3: Inference speed (can be slow)
    print("\nSkipping inference benchmark (uncomment in main() to run)")
    # benchmark_inference_speed()

    # Example 4: Transfer learning
    demonstrate_transfer_learning_strategies()

    # Example 5: Model versioning
    demonstrate_model_versioning()

    # Example 6: Custom configurations
    demonstrate_custom_configurations()

    # Example 7: Selection guide
    create_model_selection_guide()

    print("\n" + "=" * 80)
    print("Advanced Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Choose pretraining dataset based on target task")
    print("  2. Different architectures have different strengths")
    print("  3. Consider deployment constraints (speed, memory)")
    print("  4. Use appropriate transfer learning strategy")
    print("  5. Leverage HF Hub versioning for reproducibility")
    print("  6. Customize models for specific requirements")


if __name__ == "__main__":
    main()
