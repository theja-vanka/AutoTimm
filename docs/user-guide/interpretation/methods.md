# Interpretation Methods

Complete reference for all interpretation methods in AutoTimm.

## GradCAM

Gradient-weighted Class Activation Mapping uses gradients to identify important regions.

### Class: `GradCAM`

```python
from autotimm.interpretation import GradCAM

explainer = GradCAM(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
    use_cuda: bool = True
)
```

**Parameters:**

- `model` (nn.Module): The model to interpret
- `target_layer` (Optional[Union[str, nn.Module]]): Layer to use for interpretation
  - If `None`, automatically detects the last convolutional layer
  - Can be a string (e.g., "backbone.layer4") or module reference
- `use_cuda` (bool): Whether to use CUDA if available

**Methods:**

#### `explain(image, target_class=None, normalize=True)`

Generate GradCAM heatmap for an image.

**Parameters:**
- `image` (Union[Image.Image, np.ndarray, torch.Tensor]): Input image
- `target_class` (Optional[int]): Class to explain (None = use predicted class)
- `normalize` (bool): Whether to normalize heatmap to [0, 1]

**Returns:**
- `np.ndarray`: Heatmap of shape (H, W) with values in [0, 1]

**Example:**

```python
explainer = GradCAM(model, target_layer="backbone.layer4")
heatmap = explainer.explain(image, target_class=5)
```

#### `explain_batch(images, target_classes=None, batch_size=32)`

Explain multiple images efficiently.

**Parameters:**
- `images` (List): List of images
- `target_classes` (Optional[List[int]]): Target classes for each image
- `batch_size` (int): Batch size for processing

**Returns:**
- `List[np.ndarray]`: List of heatmaps

**Example:**

```python
heatmaps = explainer.explain_batch(images, batch_size=16)
```

#### `set_target_layer(layer_name)`

Change the target layer dynamically.

**Parameters:**
- `layer_name` (Union[str, nn.Module]): New target layer

**Example:**

```python
explainer.set_target_layer("backbone.layer3")
```

---

## GradCAM++

Improved version of GradCAM with better localization for multiple objects.

### Class: `GradCAMPlusPlus`

```python
from autotimm.interpretation import GradCAMPlusPlus

explainer = GradCAMPlusPlus(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
    use_cuda: bool = True
)
```

**Parameters:** Same as GradCAM

**Methods:** Same as GradCAM

**Key Differences from GradCAM:**

- Uses second-order gradients for weight calculation
- Better at localizing multiple occurrences of objects
- Improved performance on overlapping objects

**Example:**

```python
explainer = GradCAMPlusPlus(model)
heatmap = explainer.explain(image)

# Compare with GradCAM
from autotimm.interpretation import compare_methods
results = compare_methods(model, image, methods=["gradcam", "gradcam++"])
```

---

## Integrated Gradients

Path-based attribution method with theoretical guarantees.

### Class: `IntegratedGradients`

```python
from autotimm.interpretation import IntegratedGradients

explainer = IntegratedGradients(
    model: nn.Module,
    baseline: str = 'black',
    steps: int = 50,
    use_cuda: bool = True
)
```

**Parameters:**

- `model` (nn.Module): The model to interpret
- `baseline` (str): Baseline for path integration
  - `'black'`: All zeros (default)
  - `'white'`: All ones
  - `'blur'`: Gaussian blurred version of input
  - `'random'`: Random noise
- `steps` (int): Number of integration steps (more = more accurate but slower)
- `use_cuda` (bool): Whether to use CUDA

**Methods:**

#### `explain(image, target_class=None)`

Generate attribution map.

**Parameters:**
- `image` (Union[Image.Image, np.ndarray, torch.Tensor]): Input image
- `target_class` (Optional[int]): Target class

**Returns:**
- `np.ndarray`: Attribution map of shape (H, W) with values in [0, 1]

**Example:**

```python
# Black baseline (default)
explainer = IntegratedGradients(model, baseline='black', steps=50)
attribution = explainer.explain(image, target_class=3)

# Blur baseline (often better for natural images)
explainer_blur = IntegratedGradients(model, baseline='blur', steps=50)
attribution_blur = explainer_blur.explain(image)
```

#### `visualize_polarity(image, target_class=None, save_path=None)`

Visualize positive and negative attributions separately.

**Parameters:**
- `image`: Input image
- `target_class`: Target class
- `save_path` (Optional[str]): Path to save visualization

**Returns:**
- Matplotlib figure

**Example:**

```python
fig = explainer.visualize_polarity(
    image,
    target_class=5,
    save_path="polarity.png"
)
```

#### `check_completeness(image, target_class=None)`

Verify the completeness axiom (attributions sum to prediction difference).

**Parameters:**
- `image`: Input image
- `target_class`: Target class

**Returns:**
- `Dict`: Contains 'completeness_score' and 'is_satisfied'

**Example:**

```python
completeness = explainer.check_completeness(image, target_class=5)
print(f"Completeness score: {completeness['completeness_score']:.4f}")
print(f"Axiom satisfied: {completeness['is_satisfied']}")
```

---

## SmoothGrad

Noise reduction wrapper for any interpretation method.

### Class: `SmoothGrad`

```python
from autotimm.interpretation import SmoothGrad, GradCAM

base_explainer = GradCAM(model)
explainer = SmoothGrad(
    base_explainer: BaseInterpreter,
    noise_level: float = 0.15,
    num_samples: int = 50
)
```

**Parameters:**

- `base_explainer` (BaseInterpreter): Any interpretation method to wrap
- `noise_level` (float): Standard deviation of Gaussian noise (as fraction of input range)
- `num_samples` (int): Number of noisy samples to average

**Methods:**

#### `explain(image, target_class=None)`

Generate smoothed attribution map.

**Parameters:**
- `image`: Input image
- `target_class`: Target class

**Returns:**
- `np.ndarray`: Smoothed attribution map

**Example:**

```python
# Smooth GradCAM
base = GradCAM(model)
smooth = SmoothGrad(base, noise_level=0.15, num_samples=50)
heatmap = smooth.explain(image)

# Smooth Integrated Gradients
base_ig = IntegratedGradients(model)
smooth_ig = SmoothGrad(base_ig, noise_level=0.10, num_samples=30)
attribution = smooth_ig.explain(image)
```

**Trade-offs:**

- Higher `num_samples`: Smoother results but slower
- Higher `noise_level`: More smoothing but may lose details
- Recommended: `noise_level=0.15`, `num_samples=50`

---

## Attention Visualization

For Vision Transformers (ViTs).

### Class: `AttentionRollout`

Recursive aggregation of attention across layers.

```python
from autotimm.interpretation import AttentionRollout

explainer = AttentionRollout(
    model: nn.Module,
    head_fusion: str = 'mean',
    discard_ratio: float = 0.9,
    use_cuda: bool = True
)
```

**Parameters:**

- `model` (nn.Module): Vision Transformer model
- `head_fusion` (str): How to fuse multi-head attention
  - `'mean'`: Average across heads (default)
  - `'max'`: Maximum across heads
  - `'min'`: Minimum across heads
- `discard_ratio` (float): Fraction of lowest attentions to discard (0.0-1.0)
- `use_cuda` (bool): Whether to use CUDA

**Methods:**

#### `explain(image, discard_cls_token=True)`

Generate attention rollout map.

**Parameters:**
- `image`: Input image
- `discard_cls_token` (bool): Whether to discard CLS token attention

**Returns:**
- `np.ndarray`: Attention map of shape (H, W)

**Example:**

```python
from transformers import ViTForImageClassification

vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
explainer = AttentionRollout(vit_model, head_fusion='mean')
attention_map = explainer.explain(image)
```

### Class: `AttentionFlow`

Visualize attention flow from a specific source patch.

```python
from autotimm.interpretation import AttentionFlow

explainer = AttentionFlow(
    model: nn.Module,
    target_patch: int = 0,  # CLS token
    use_cuda: bool = True
)
```

**Parameters:**

- `model`: Vision Transformer model
- `target_patch` (int): Source patch index to track
- `use_cuda`: Whether to use CUDA

**Methods:**

#### `explain(image)`

Generate attention flow map from target patch.

**Returns:**
- `np.ndarray`: Flow map showing where attention flows from target patch

**Example:**

```python
# Track attention from CLS token
flow_cls = AttentionFlow(vit_model, target_patch=0)
flow_map = flow_cls.explain(image)

# Track attention from center patch
num_patches = 196  # 14x14 for 224x224 image with 16x16 patches
center_patch = num_patches // 2
flow_center = AttentionFlow(vit_model, target_patch=center_patch)
flow_map_center = flow_center.explain(image)
```

---

## Method Comparison

### Performance Characteristics

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| GradCAM | ⚡⚡⚡ Fast | Good | General CNN visualization |
| GradCAM++ | ⚡⚡ Medium | Better | Multiple objects |
| Integrated Gradients | ⚡ Slow | Best | Pixel-level attributions |
| SmoothGrad | ⚡ Slow | Cleaner | Reducing noise |
| AttentionRollout | ⚡⚡ Medium | Good | ViTs, global attention |
| AttentionFlow | ⚡⚡ Medium | Good | ViTs, local attention |

### When to Use Each Method

**GradCAM**: Your default choice for CNNs. Fast, reliable, and works well in most cases.

```python
explainer = GradCAM(model)
```

**GradCAM++**: When you have multiple objects or need better localization.

```python
explainer = GradCAMPlusPlus(model)
```

**Integrated Gradients**: When you need theoretical guarantees or pixel-level attributions.

```python
explainer = IntegratedGradients(model, baseline='blur', steps=50)
```

**SmoothGrad**: When you want to reduce noise in visualizations.

```python
base = GradCAM(model)
explainer = SmoothGrad(base, noise_level=0.15, num_samples=50)
```

**Attention Methods**: For Vision Transformers to understand attention patterns.

```python
explainer = AttentionRollout(vit_model, head_fusion='mean')
```

---

## Common Parameters

### Target Class

Most methods accept `target_class` parameter:

- `None` (default): Use the predicted class
- `int`: Specific class to explain

```python
# Explain predicted class
heatmap = explainer.explain(image)

# Explain specific class
heatmap = explainer.explain(image, target_class=5)
```

### Target Layer

For gradient-based methods, specify which layer to use:

- `None` (default): Auto-detect last convolutional layer
- `str`: Layer name (e.g., "backbone.layer4")
- `nn.Module`: Direct module reference

```python
# Auto-detect
explainer = GradCAM(model)

# By name
explainer = GradCAM(model, target_layer="backbone.layer3")

# By reference
explainer = GradCAM(model, target_layer=model.backbone.layer4)
```

### CUDA Acceleration

Enable GPU acceleration for faster computation:

```python
explainer = GradCAM(model, use_cuda=True)
```

---

## See Also

- [High-Level API](index.md#high-level-api) - Simple functions for common use cases
- [Task Adapters](task-adapters.md) - Task-specific interpretation
- [Feature Visualization](feature-visualization.md) - Analyze learned features
