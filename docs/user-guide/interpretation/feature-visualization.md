# Feature Visualization

The `FeatureVisualizer` class provides tools for analyzing and visualizing feature maps from any layer in your neural network.

## Overview

Feature visualization helps you understand:

- **What features the model learns** at different depths
- **How features respond** to specific inputs
- **Layer-by-layer progression** of learned representations
- **Feature sparsity** and activation patterns
- **Receptive fields** of individual neurons

## Class: `FeatureVisualizer`

```python
from autotimm.interpretation import FeatureVisualizer

viz = FeatureVisualizer(
    model: nn.Module,
    use_cuda: bool = True
)
```

**Parameters:**

- `model` (nn.Module): Model to visualize
- `use_cuda` (bool): Whether to use CUDA if available

**Example:**

```python
from autotimm import ImageClassifier
from autotimm.interpretation import FeatureVisualizer

model = ImageClassifier(backbone="resnet50", num_classes=10)
viz = FeatureVisualizer(model)
```

---

## Methods

### `plot_feature_maps()`

Visualize feature maps from a specific layer.

```python
fig = viz.plot_feature_maps(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    layer_name: str,
    num_features: int = 16,
    sort_by: str = "activation",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure
```

**Parameters:**

- `image`: Input image
- `layer_name` (str): Name of layer to visualize (e.g., "backbone.layer3")
- `num_features` (int): Number of feature maps to display (default: 16)
- `sort_by` (str): How to select features:
  - `"activation"` (default): Highest mean activation
  - `"variance"`: Highest variance
  - `"random"`: Random selection
- `save_path` (Optional[str]): Path to save figure
- `figsize` (Optional[Tuple[int, int]]): Figure size (width, height)

**Returns:**
- `plt.Figure`: Matplotlib figure with feature maps

**Example:**

```python
from PIL import Image

image = Image.open("dog.jpg")

# Plot 16 most activated features
fig = viz.plot_feature_maps(
    image,
    layer_name="backbone.layer3",
    num_features=16,
    sort_by="activation",
    save_path="features.png"
)
```

**Output:**

Each subplot shows:
- Feature map visualization
- Channel index
- Mean activation value

---

### `get_features()`

Extract raw features from a specific layer.

```python
features = viz.get_features(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    layer_name: str
) -> torch.Tensor
```

**Parameters:**

- `image`: Input image
- `layer_name` (str): Name of layer

**Returns:**
- `torch.Tensor`: Feature tensor of shape (B, C, H, W)

**Example:**

```python
features = viz.get_features(image, layer_name="backbone.layer4")
print(f"Shape: {features.shape}")  # e.g., (1, 512, 7, 7)
print(f"Mean activation: {features.mean():.3f}")
```

---

### `get_feature_statistics()`

Compute comprehensive statistics for a layer's features.

```python
stats = viz.get_feature_statistics(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    layer_name: str
) -> Dict[str, float]
```

**Parameters:**

- `image`: Input image
- `layer_name` (str): Name of layer

**Returns:**
- `Dict[str, float]`: Dictionary containing:
  - `mean`: Mean activation across all features
  - `std`: Standard deviation
  - `sparsity`: Fraction of zero activations (0.0-1.0)
  - `max`: Maximum activation
  - `min`: Minimum activation
  - `active_channels`: Number of channels with mean > 0.01
  - `num_channels`: Total number of channels
  - `spatial_size`: Tuple of (height, width)

**Example:**

```python
stats = viz.get_feature_statistics(image, layer_name="backbone.layer4")

print(f"Layer Statistics:")
print(f"  Mean activation: {stats['mean']:.3f}")
print(f"  Std deviation: {stats['std']:.3f}")
print(f"  Sparsity: {stats['sparsity']:.2%}")
print(f"  Active channels: {stats['active_channels']}/{stats['num_channels']}")
print(f"  Spatial size: {stats['spatial_size']}")
```

**Output:**
```
Layer Statistics:
  Mean activation: 0.234
  Std deviation: 0.156
  Sparsity: 34.52%
  Active channels: 487/512
  Spatial size: (7, 7)
```

---

### `compare_layers()`

Compare feature statistics across multiple layers.

```python
all_stats = viz.compare_layers(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    layer_names: List[str],
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]
```

**Parameters:**

- `image`: Input image
- `layer_names` (List[str]): List of layer names to compare
- `save_path` (Optional[str]): Path to save comparison plot

**Returns:**
- `Dict[str, Dict[str, float]]`: Dictionary mapping layer names to their statistics

**Example:**

```python
layers = ["backbone.layer1", "backbone.layer2", "backbone.layer3", "backbone.layer4"]
all_stats = viz.compare_layers(
    image,
    layers,
    save_path="layer_comparison.png"
)

# Analyze progression
for layer, stats in all_stats.items():
    print(f"\n{layer}:")
    print(f"  Channels: {stats['num_channels']}")
    print(f"  Spatial: {stats['spatial_size']}")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
```

**Output Visualization:**

The saved plot shows 4 subplots:
1. Mean activation per layer
2. Standard deviation per layer
3. Sparsity per layer
4. Active channels per layer

---

### `get_top_activating_features()`

Find channels with highest mean activation.

```python
top_features = viz.get_top_activating_features(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    layer_name: str,
    top_k: int = 10
) -> List[Tuple[int, float]]
```

**Parameters:**

- `image`: Input image
- `layer_name` (str): Name of layer
- `top_k` (int): Number of top channels to return

**Returns:**
- `List[Tuple[int, float]]`: List of (channel_index, mean_activation) tuples

**Example:**

```python
top_features = viz.get_top_activating_features(
    image,
    layer_name="backbone.layer4",
    top_k=5
)

print("Top 5 Activating Channels:")
for channel, activation in top_features:
    print(f"  Channel {channel}: {activation:.3f}")
```

**Output:**
```
Top 5 Activating Channels:
  Channel 342: 0.892
  Channel 156: 0.847
  Channel 423: 0.821
  Channel 89: 0.798
  Channel 267: 0.765
```

---

### `visualize_receptive_field()`

Approximate the receptive field using occlusion analysis.

```python
sensitivity = viz.visualize_receptive_field(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    layer_name: str,
    channel: int,
    position: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> np.ndarray
```

**Parameters:**

- `image`: Input image
- `layer_name` (str): Name of layer
- `channel` (int): Channel index to analyze
- `position` (Optional[Tuple[int, int]]): Position in feature map (h, w). If None, uses center
- `save_path` (Optional[str]): Path to save visualization

**Returns:**
- `np.ndarray`: Sensitivity map showing receptive field

**Example:**

```python
# Get top activating channel first
top_features = viz.get_top_activating_features(image, "backbone.layer3", top_k=1)
channel = top_features[0][0]

# Visualize its receptive field
sensitivity = viz.visualize_receptive_field(
    image,
    layer_name="backbone.layer3",
    channel=channel,
    save_path="receptive_field.png"
)

print(f"Receptive field computed for channel {channel}")
```

**Note:** This is computationally intensive as it performs occlusion analysis. For faster analysis, use smaller images.

**Output:**

Saves a figure with two subplots:
1. Original image
2. Receptive field heatmap (hot = more influential)

---

## Common Use Cases

### 1. Understanding Model Depth Progression

Visualize how features evolve from shallow to deep layers:

```python
layers = ["backbone.layer1", "backbone.layer2", "backbone.layer3", "backbone.layer4"]

for layer in layers:
    viz.plot_feature_maps(
        image,
        layer_name=layer,
        num_features=16,
        save_path=f"{layer}_features.png"
    )
```

**Observations:**
- Early layers (layer1): Edge detection, colors, simple patterns
- Middle layers (layer2-3): Textures, parts of objects
- Deep layers (layer4): Complex patterns, object parts

### 2. Analyzing Model Sparsity

Check how sparse your model's activations are:

```python
layers = ["backbone.layer1", "backbone.layer2", "backbone.layer3", "backbone.layer4"]
all_stats = viz.compare_layers(image, layers)

for layer, stats in all_stats.items():
    print(f"{layer}: Sparsity = {stats['sparsity']:.2%}")
```

**High sparsity (>50%)**: Many neurons are inactive, potentially indicating:
- Good feature specialization
- Possible overfitting
- Need for regularization

**Low sparsity (<20%)**: Most neurons active, potentially indicating:
- Dense representations
- Possible redundancy
- May benefit from pruning

### 3. Finding Important Channels

Identify which channels are most responsive to your input:

```python
# Get top channels
top_channels = viz.get_top_activating_features(
    image,
    layer_name="backbone.layer4",
    top_k=10
)

# Visualize only these channels
features = viz.get_features(image, "backbone.layer4")
top_indices = [ch for ch, _ in top_channels]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for idx, channel in enumerate(top_indices):
    ax = axes[idx // 5, idx % 5]
    feature_map = features[0, channel].cpu().numpy()
    ax.imshow(feature_map, cmap='viridis')
    ax.set_title(f"Ch {channel}")
    ax.axis('off')
plt.savefig("top_channels.png")
```

### 4. Debugging Feature Learning

Check if your model is learning meaningful features:

```python
# Compare statistics before and after training
stats_before = viz.get_feature_statistics(image, "backbone.layer4")
# ... train model ...
stats_after = viz.get_feature_statistics(image, "backbone.layer4")

print("Before training:")
print(f"  Mean: {stats_before['mean']:.3f}, Sparsity: {stats_before['sparsity']:.2%}")
print("After training:")
print(f"  Mean: {stats_after['mean']:.3f}, Sparsity: {stats_after['sparsity']:.2%}")
```

### 5. Model Comparison

Compare feature learning across different architectures:

```python
models = {
    "ResNet18": ImageClassifier(backbone="resnet18", num_classes=10),
    "ResNet50": ImageClassifier(backbone="resnet50", num_classes=10),
    "EfficientNet": ImageClassifier(backbone="efficientnet_b0", num_classes=10),
}

for name, model in models.items():
    viz = FeatureVisualizer(model)
    stats = viz.get_feature_statistics(image, layer_name="backbone.layer4")
    print(f"{name}: Mean={stats['mean']:.3f}, Sparsity={stats['sparsity']:.2%}")
```

---

## Advanced Techniques

### Custom Feature Selection

Implement custom sorting logic:

```python
features = viz.get_features(image, "backbone.layer3")

# Select based on L1 norm
channel_l1 = features.abs().mean(dim=(2, 3)).squeeze()
top_k = 16
_, top_indices = torch.topk(channel_l1, k=top_k)

# Visualize selected features
selected_features = features[0, top_indices]
# ... plot selected features ...
```

### Temporal Analysis

Track feature evolution during training:

```python
# During training loop
if epoch % 5 == 0:
    stats = viz.get_feature_statistics(sample_image, "backbone.layer4")
    history['mean'].append(stats['mean'])
    history['sparsity'].append(stats['sparsity'])

# Plot evolution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['mean'])
plt.title('Mean Activation Over Training')
plt.subplot(1, 2, 2)
plt.plot(history['sparsity'])
plt.title('Sparsity Over Training')
plt.savefig('feature_evolution.png')
```

### Multi-Image Analysis

Analyze features across multiple images:

```python
images = [load_image(f"test_{i}.jpg") for i in range(10)]

all_activations = []
for image in images:
    top = viz.get_top_activating_features(image, "backbone.layer4", top_k=5)
    all_activations.extend([ch for ch, _ in top])

# Find most commonly activated channels
from collections import Counter
common_channels = Counter(all_activations).most_common(10)
print("Most commonly activated channels across images:")
for channel, count in common_channels:
    print(f"  Channel {channel}: {count} times")
```

---

## Performance Tips

### 1. Use Smaller Images for Exploration

```python
# Resize for faster analysis
from torchvision import transforms
resize = transforms.Resize((224, 224))
small_image = resize(large_image)

viz.plot_feature_maps(small_image, "backbone.layer3")
```

### 2. Batch Processing

For multiple images, process in batches:

```python
# Extract features once
features_list = []
for image in images:
    features = viz.get_features(image, "backbone.layer4")
    features_list.append(features)

# Analyze batch statistics
all_features = torch.cat(features_list, dim=0)
batch_mean = all_features.mean().item()
batch_sparsity = (all_features == 0).float().mean().item()
```

### 3. Cache Features

If analyzing the same image multiple times:

```python
# Extract once
features_l3 = viz.get_features(image, "backbone.layer3")
features_l4 = viz.get_features(image, "backbone.layer4")

# Analyze cached features
# (implement custom analysis on cached tensors)
```

---

## Troubleshooting

### ValueError: Layer not found

**Problem:** Layer name is incorrect

**Solution:** Check available layer names:
```python
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(name)
```

### Memory Error

**Problem:** Image too large or too many features

**Solution:**
- Resize input image
- Reduce `num_features` parameter
- Process features in chunks

### Blank Feature Maps

**Problem:** Features are all zero or very small

**Solution:**
- Check that model is trained
- Verify model is in eval mode
- Try a different layer (earlier layers are usually more active)

---

## See Also

- [Interpretation Methods](methods.md) - GradCAM, Integrated Gradients, etc.
- [Callbacks](callbacks.md) - Monitor features during training
- [Main Guide](index.md) - Overview and quick start
