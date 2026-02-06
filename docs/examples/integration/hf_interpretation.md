# Model Interpretation with HuggingFace Hub Models

Comprehensive guide to interpreting and explaining predictions from HuggingFace Hub models.

## Overview

This example demonstrates various interpretation techniques for understanding what HuggingFace Hub models learn and how they make decisions. Includes GradCAM, attention visualization, integrated gradients, and quantitative evaluation metrics.

## What This Example Covers

- **GradCAM visualization** - Highlighting important regions for CNNs
- **Attention visualization** - Understanding ViT self-attention patterns
- **Integrated Gradients** - Pixel-level attribution
- **Quantitative metrics** - Insertion, deletion, sensitivity
- **Architecture comparison** - Interpretation across model families
- **Interactive visualizations** - Plotly-based exploration tools

## Key Features

### Supported Interpretation Methods

1. **GradCAM** - Best for CNNs (ResNet, EfficientNet, ConvNeXt)
2. **GradCAM++** - Enhanced version with better localization
3. **Attention Visualization** - For Vision Transformers
4. **Integrated Gradients** - Attribution method for any architecture
5. **Interactive Viz** - Plotly-based interactive heatmaps

### Supported Architectures

- **ResNet/ResNeXt** - Excellent GradCAM support
- **EfficientNet** - Works with all gradient-based methods
- **ConvNeXt** - Modern CNN with clear feature hierarchies
- **Vision Transformers** - Native attention visualization
- **DeiT** - Enhanced ViT with distillation tokens

## Example Code

### Quick Start

```python
from autotimm import ImageClassifier
from autotimm.interpretation import GradCAM, quick_explain
from PIL import Image

# Load model from HuggingFace Hub
model = ImageClassifier(
    backbone="hf-hub:timm/resnet18.a1_in1k",
    num_classes=10,
)

# Load image
image = Image.open("example.jpg")

# One-line explanation
result = quick_explain(model, image, save_path="explanation.png")
```

### GradCAM with Different Architectures

```python
from autotimm.interpretation import GradCAM

# ResNet
model = ImageClassifier(backbone="hf-hub:timm/resnet50.a1_in1k", num_classes=10)
explainer = GradCAM(model, target_layer="backbone.layer4")
heatmap = explainer(image, target_class=0)

# ConvNeXt
model = ImageClassifier(backbone="hf-hub:timm/convnext_tiny.fb_in1k", num_classes=10)
explainer = GradCAM(model, target_layer="backbone.stages.3")
heatmap = explainer(image, target_class=0)

# EfficientNet
model = ImageClassifier(backbone="hf-hub:timm/efficientnet_b0.ra_in1k", num_classes=10)
explainer = GradCAM(model, target_layer="backbone.conv_head")
heatmap = explainer(image, target_class=0)
```

### Attention Visualization for ViTs

```python
from autotimm.interpretation import AttentionVisualizer

# Load Vision Transformer
model = ImageClassifier(
    backbone="hf-hub:timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    num_classes=10,
)

# Visualize attention
visualizer = AttentionVisualizer(model)
attention_maps = visualizer.get_attention_maps(image)
rollout = visualizer.attention_rollout(image)
```

### Quantitative Evaluation

```python
from autotimm.interpretation.metrics import ExplanationMetrics

metrics = ExplanationMetrics(model)

# Insertion/Deletion metrics
insertion_score = metrics.insertion(image, heatmap, target_class=0)
deletion_score = metrics.deletion(image, heatmap, target_class=0)

# Sensitivity
sensitivity = metrics.sensitivity(image, heatmap, target_class=0)

print(f"Insertion AUC: {insertion_score:.4f} (higher is better)")
print(f"Deletion AUC: {deletion_score:.4f} (lower is better)")
print(f"Sensitivity: {sensitivity:.4f} (higher is better)")
```

## Run the Example

```bash
python examples/huggingface/hf_interpretation.py
```

## Output

The example generates:

- **Heatmap visualizations** for different architectures
- **Attention maps** for Vision Transformers
- **Comparison plots** across interpretation methods
- **Quantitative metrics** for explanation quality
- **Interactive HTML** files for exploration

## Use Cases

- **Model debugging** - Identify what the model is looking at
- **Error analysis** - Understand misclassifications
- **Trust building** - Explain predictions to stakeholders
- **Architecture comparison** - Compare interpretability across models
- **Feature analysis** - Discover learned features

## Best Practices

1. **Choose the right method**:
   - CNNs → GradCAM or GradCAM++
   - ViTs → Attention visualization
   - Any model → Integrated Gradients

2. **Target layer selection**:
   - Use later layers for better localization
   - ResNet: `layer4`, ConvNeXt: `stages.3`, EfficientNet: `conv_head`

3. **Quantitative evaluation**:
   - Don't rely on visual inspection alone
   - Use insertion/deletion metrics
   - Compare multiple methods

4. **Interactive visualization**:
   - Use Plotly for exploration
   - Adjust opacity and colormap
   - Zoom into important regions

## Related Examples

- [HuggingFace Hub Models](huggingface-hub.md) - Loading HF Hub models
- [Transfer Learning](hf_transfer_learning.md) - Fine-tuning strategies
- [Custom Data](../utilities/hf_custom_data.md) - Working with custom datasets

## See Also

- [Interpretation Guide](../../user-guide/interpretation/methods.md)
- [GradCAM Documentation](../../api/interpretation.md#gradcam)
- [Attention Visualization](../../api/interpretation.md#attention)
