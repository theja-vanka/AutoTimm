# Backbone Utilities Examples

This page demonstrates how to explore and use timm backbones in AutoTimm.

## Backbone Discovery

Explore available timm backbones and their properties.

```python
import autotimm


def main():
    # List all backbones
    all_models = autotimm.list_backbones()
    print(f"Total models: {len(all_models)}")

    # Search by pattern
    resnet = autotimm.list_backbones("*resnet*")
    print(f"ResNet variants: {len(resnet)}")
    
    efficientnet = autotimm.list_backbones("*efficientnet*", pretrained_only=True)
    print(f"EfficientNet variants (pretrained): {len(efficientnet)}")
    
    vit = autotimm.list_backbones("*vit*")
    print(f"Vision Transformer variants: {len(vit)}")

    # Print some examples
    print("\nSample ResNet models:")
    for model in resnet[:5]:
        print(f"  - {model}")

    # Inspect a backbone
    backbone = autotimm.create_backbone("resnet50")
    print(f"\nResNet50 details:")
    print(f"  Output features: {backbone.num_features}")
    print(f"  Parameters: {autotimm.count_parameters(backbone):,}")


if __name__ == "__main__":
    main()
```

---

## Popular Backbone Families

Common backbone architectures available in timm.

### ResNet Family

```python
import autotimm

# List ResNet models
resnets = autotimm.list_backbones("resnet*")

# Popular ResNet variants
popular_resnets = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
]

for model in popular_resnets:
    backbone = autotimm.create_backbone(model)
    print(f"{model}: {autotimm.count_parameters(backbone):,} params")
```

### EfficientNet Family

```python
import autotimm

# List EfficientNet models
efficientnets = autotimm.list_backbones("efficientnet*")

# Popular EfficientNet variants
popular_efficientnets = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnetv2_rw_s",
    "efficientnetv2_rw_m",
]

for model in popular_efficientnets:
    backbone = autotimm.create_backbone(model)
    print(f"{model}: {autotimm.count_parameters(backbone):,} params")
```

### Vision Transformer Family

```python
import autotimm

# List ViT models
vits = autotimm.list_backbones("vit*")

# Popular ViT variants
popular_vits = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
]

for model in popular_vits:
    backbone = autotimm.create_backbone(model)
    print(f"{model}: {autotimm.count_parameters(backbone):,} params")
```

### Swin Transformer Family

```python
import autotimm

# List Swin models
swins = autotimm.list_backbones("swin*")

# Popular Swin variants
popular_swins = [
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "swin_large_patch4_window7_224",
]

for model in popular_swins:
    backbone = autotimm.create_backbone(model)
    print(f"{model}: {autotimm.count_parameters(backbone):,} params")
```

---

## Backbone Comparison

Compare multiple backbones for your use case.

```python
import autotimm


def compare_backbones(model_names):
    """Compare multiple backbones."""
    print(f"{'Model':<40} {'Params':>12} {'Features':>10}")
    print("-" * 65)
    
    for name in model_names:
        backbone = autotimm.create_backbone(name)
        params = autotimm.count_parameters(backbone)
        features = backbone.num_features
        print(f"{name:<40} {params:>12,} {features:>10}")


def main():
    # Compare different ResNet variants
    print("ResNet Comparison:")
    compare_backbones([
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
    ])
    
    print("\nEfficientNet Comparison:")
    compare_backbones([
        "efficientnet_b0",
        "efficientnet_b2",
        "efficientnet_b4",
    ])
    
    print("\nTransformer Comparison:")
    compare_backbones([
        "vit_tiny_patch16_224",
        "vit_base_patch16_224",
        "swin_tiny_patch4_window7_224",
        "swin_base_patch4_window7_224",
    ])


if __name__ == "__main__":
    main()
```

---

## Backbone Features

Understanding backbone output features.

```python
import autotimm
import torch


def inspect_backbone(model_name):
    """Inspect backbone architecture and output."""
    backbone = autotimm.create_backbone(model_name, pretrained=True)
    
    # Get backbone info
    print(f"Model: {model_name}")
    print(f"Parameters: {autotimm.count_parameters(backbone):,}")
    print(f"Output features: {backbone.num_features}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = backbone(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print()


def main():
    # Inspect various backbones
    models = [
        "resnet50",
        "efficientnet_b3",
        "vit_base_patch16_224",
        "swin_tiny_patch4_window7_224",
        "convnext_tiny",
    ]
    
    for model in models:
        inspect_backbone(model)


if __name__ == "__main__":
    main()
```

---

## Searching for Specific Backbones

Find backbones matching specific criteria.

```python
import autotimm


def main():
    # Search for mobile-optimized models
    mobile_models = autotimm.list_backbones("*mobile*")
    print(f"Mobile models: {len(mobile_models)}")
    print("Examples:", mobile_models[:3])
    
    # Search for ConvNeXt models
    convnext_models = autotimm.list_backbones("*convnext*")
    print(f"\nConvNeXt models: {len(convnext_models)}")
    print("Examples:", convnext_models[:3])
    
    # Search for DeiT models
    deit_models = autotimm.list_backbones("*deit*")
    print(f"\nDeiT models: {len(deit_models)}")
    print("Examples:", deit_models[:3])
    
    # Search for RegNet models
    regnet_models = autotimm.list_backbones("*regnet*")
    print(f"\nRegNet models: {len(regnet_models)}")
    print("Examples:", regnet_models[:3])
    
    # Only pretrained models
    pretrained_efficientnets = autotimm.list_backbones(
        "*efficientnet*", 
        pretrained_only=True
    )
    print(f"\nPretrained EfficientNets: {len(pretrained_efficientnets)}")


if __name__ == "__main__":
    main()
```

---

## Backbone Selection Guide

**For Classification:**

| Use Case | Recommended Backbone | Reason |
|----------|---------------------|--------|
| Quick prototyping | `resnet18`, `resnet34` | Fast, small, reliable |
| Balanced performance | `resnet50`, `efficientnet_b3` | Good accuracy/speed tradeoff |
| Maximum accuracy | `vit_large_patch16_224`, `swin_large_patch4_window7_224` | State-of-the-art performance |
| Mobile/Edge deployment | `efficientnet_b0`, `mobilenetv3_small_100` | Efficient inference |
| Fine-tuning | `vit_base_patch16_224` | Excellent transfer learning |

**For Object Detection:**

| Use Case | Recommended Backbone | Reason |
|----------|---------------------|--------|
| General purpose | `resnet50`, `resnet101` | Proven performance |
| Speed critical | `efficientnet_b0`, `resnet34` | Fast inference |
| Small objects | `swin_tiny_patch4_window7_224` | Hierarchical features |
| Maximum accuracy | `swin_base_patch4_window7_224` | Best mAP |

---

## Running Backbone Discovery Examples

Clone the repository and run examples:

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[all]"

# Run backbone discovery example
python examples/backbone_discovery.py
```
