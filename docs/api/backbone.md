# Backbone

Backbone creation and discovery via timm.

## BackboneConfig

Configuration dataclass for timm backbones.

### API Reference

::: autotimm.BackboneConfig
    options:
      show_source: true

### Usage Examples

#### Basic Configuration

```python
from autotimm import BackboneConfig

cfg = BackboneConfig(
    model_name="resnet50",
    pretrained=True,
)
```

#### With Regularization

```python
cfg = BackboneConfig(
    model_name="vit_base_patch16_224",
    pretrained=True,
    drop_rate=0.1,           # Dropout
    drop_path_rate=0.1,      # Stochastic depth
)
```

#### With Extra kwargs

```python
cfg = BackboneConfig(
    model_name="efficientnet_b3",
    pretrained=True,
    extra_kwargs={
        "global_pool": "avg",
        "in_chans": 3,
    },
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"resnet50"` | timm model name |
| `pretrained` | `bool` | `True` | Load pretrained weights |
| `num_classes` | `int` | `0` | 0 for feature extractor |
| `drop_rate` | `float` | `0.0` | Dropout rate |
| `drop_path_rate` | `float` | `0.0` | Stochastic depth rate |
| `extra_kwargs` | `dict` | `{}` | Extra timm.create_model kwargs |

---

## create_backbone

Create a timm backbone from a config or model name.

### API Reference

::: autotimm.create_backbone
    options:
      show_source: true

### Usage Examples

#### From String

```python
import autotimm

backbone = autotimm.create_backbone("resnet50")
print(f"Output features: {backbone.num_features}")
```

#### From Config

```python
from autotimm import BackboneConfig, create_backbone

cfg = BackboneConfig(
    model_name="vit_base_patch16_224",
    pretrained=True,
    drop_path_rate=0.1,
)
backbone = create_backbone(cfg)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg` | `BackboneConfig \| str` | Config or model name |

### Returns

| Type | Description |
|------|-------------|
| `nn.Module` | Headless timm model |

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Model name not found in timm |

---

## list_backbones

List available timm backbones with optional filtering.

### API Reference

::: autotimm.list_backbones
    options:
      show_source: true

### Usage Examples

#### List All Backbones

```python
import autotimm

all_models = autotimm.list_backbones()
print(f"Total models: {len(all_models)}")  # 1000+
```

#### Search by Pattern

```python
# ResNet variants
resnet_models = autotimm.list_backbones("*resnet*")
print(resnet_models[:10])

# EfficientNet variants
efficientnet_models = autotimm.list_backbones("*efficientnet*")

# Vision Transformers
vit_models = autotimm.list_backbones("*vit*")

# ConvNeXt
convnext_models = autotimm.list_backbones("*convnext*")
```

#### Pretrained Only

```python
pretrained = autotimm.list_backbones("*resnet*", pretrained_only=True)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `str` | `""` | Glob pattern (e.g., `"*resnet*"`) |
| `pretrained_only` | `bool` | `False` | Only pretrained models |

### Returns

| Type | Description |
|------|-------------|
| `list[str]` | List of model names |

---

## Popular Backbones

### CNNs

| Family | Models | Notes |
|--------|--------|-------|
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` | Classic |
| ResNeXt | `resnext50_32x4d`, `resnext101_32x8d` | Grouped convolutions |
| EfficientNet | `efficientnet_b0` to `efficientnet_b7` | Compound scaling |
| EfficientNetV2 | `efficientnetv2_s`, `efficientnetv2_m`, `efficientnetv2_l` | Faster training |
| ConvNeXt | `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large` | Modern CNN |
| RegNet | `regnetx_002`, `regnety_040` | Efficient design |
| MobileNet | `mobilenetv3_small_100`, `mobilenetv3_large_100` | Mobile-friendly |

### Transformers

| Family | Models | Notes |
|--------|--------|-------|
| ViT | `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`, `vit_large_patch16_224` | Vision Transformer |
| DeiT | `deit_tiny_patch16_224`, `deit_small_patch16_224`, `deit_base_patch16_224` | Data-efficient ViT |
| Swin | `swin_tiny_patch4_window7_224`, `swin_small_patch4_window7_224`, `swin_base_patch4_window7_224` | Hierarchical ViT |
| BEiT | `beit_base_patch16_224`, `beit_large_patch16_224` | BERT-style pretraining |

### Hybrids

| Family | Models | Notes |
|--------|--------|-------|
| CoAtNet | `coatnet_0_rw_224`, `coatnet_1_rw_224` | CNN + Attention |
| MaxViT | `maxvit_tiny_rw_224`, `maxvit_small_rw_224` | Multi-axis attention |

---

## Backbone Info

```python
import autotimm

# Create backbone
backbone = autotimm.create_backbone("resnet50")

# Get output features
print(f"Output features: {backbone.num_features}")  # 2048

# Count parameters
total = autotimm.count_parameters(backbone, trainable_only=False)
print(f"Total parameters: {total:,}")  # 23,508,032
```
