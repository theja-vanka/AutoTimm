# Task-Specific Interpretation

AutoTimm provides specialized interpretation functions for object detection and semantic segmentation tasks.

## Overview

While classification interpretation is straightforward, detection and segmentation tasks require specialized handling:

- **Object Detection**: Explain individual detections with bbox-aware visualization
- **Semantic Segmentation**: Explain class predictions with uncertainty quantification

---

## Object Detection Interpretation

### Function: `explain_detection()`

```python
from autotimm.interpretation import explain_detection

results = explain_detection(
    model: nn.Module,
    image: Union[str, Image.Image, np.ndarray, torch.Tensor],
    method: str = 'gradcam',
    detection_threshold: float = 0.5,
    target_layer: Optional[Union[str, nn.Module]] = None,
    bbox: Optional[List[float]] = None,
    colormap: str = 'viridis',
    alpha: float = 0.4,
    save_path: Optional[str] = None,
    max_detections: int = 10
) -> Dict
```

**Parameters:**

- `model` (nn.Module): Object detection model (ObjectDetector, YOLOXDetector, etc.)
- `image`: Input image
- `method` (str): Interpretation method ('gradcam', 'gradcam++', 'integrated_gradients')
- `detection_threshold` (float): Minimum confidence score (default: 0.5)
- `target_layer` (Optional): Layer for interpretation (None = auto-detect)
- `bbox` (Optional[List[float]]): Specific bbox to explain [x1, y1, x2, y2] (None = all)
- `colormap` (str): Colormap for heatmaps
- `alpha` (float): Overlay transparency
- `save_path` (Optional[str]): Path to save visualization
- `max_detections` (int): Maximum detections to explain (default: 10)

**Returns:**
- `Dict`: Contains:
  - `'detections'`: List of detection dictionaries
  - `'heatmaps'`: Heatmap for each detection
  - `'visualization'`: Combined visualization (if save_path provided)

### Basic Usage

```python
from autotimm import ObjectDetector
from autotimm.interpretation import explain_detection
from PIL import Image

# Load model
model = ObjectDetector.load_from_checkpoint("detector.ckpt")

# Load image
image = Image.open("street.jpg")

# Explain all detections
results = explain_detection(
    model,
    image,
    method='gradcam',
    detection_threshold=0.5,
    save_path='detection_explanation.png'
)

# Print detections
for i, det in enumerate(results['detections']):
    print(f"Detection {i}:")
    print(f"  Class: {det['class_id']} ({det['class_name']})")
    print(f"  Confidence: {det['score']:.3f}")
    print(f"  BBox: {det['bbox']}")
```

### Explain Specific Detection

Explain only a specific bounding box:

```python
# Define bbox of interest [x1, y1, x2, y2]
bbox = [100, 150, 300, 400]

results = explain_detection(
    model,
    image,
    bbox=bbox,
    method='gradcam',
    save_path='single_detection.png'
)
```

### High-Confidence Detections Only

Filter to only explain high-confidence detections:

```python
results = explain_detection(
    model,
    image,
    detection_threshold=0.8,  # Only explain confident detections
    max_detections=5,         # Limit to top 5
    save_path='confident_detections.png'
)
```

### With GradCAM++

Use GradCAM++ for better localization of multiple objects:

```python
results = explain_detection(
    model,
    image,
    method='gradcam++',
    detection_threshold=0.5,
    save_path='detection_gradcampp.png'
)
```

### Understanding the Output

The returned dictionary contains:

```python
results = {
    'detections': [
        {
            'class_id': 2,
            'class_name': 'car',
            'score': 0.89,
            'bbox': [120.5, 180.2, 340.8, 420.6],  # [x1, y1, x2, y2]
        },
        # ... more detections
    ],
    'heatmaps': [
        np.ndarray,  # Heatmap for first detection
        # ... more heatmaps
    ],
    'visualization': np.ndarray,  # Combined visualization (if save_path)
}
```

### Visualization Format

The saved visualization includes:
- Original image with bounding boxes
- Class labels and confidence scores
- Heatmap overlays for each detection (masked to bbox)
- Color-coded bboxes per detection

---

## Semantic Segmentation Interpretation

### Function: `explain_segmentation()`

```python
from autotimm.interpretation import explain_segmentation

results = explain_segmentation(
    model: nn.Module,
    image: Union[str, Image.Image, np.ndarray, torch.Tensor],
    target_class: Optional[int] = None,
    method: str = 'gradcam',
    target_layer: Optional[Union[str, nn.Module]] = None,
    show_uncertainty: bool = False,
    uncertainty_method: str = 'entropy',
    colormap: str = 'viridis',
    alpha: float = 0.4,
    save_path: Optional[str] = None
) -> Dict
```

**Parameters:**

- `model` (nn.Module): Semantic segmentation model (SemanticSegmentor, etc.)
- `image`: Input image
- `target_class` (Optional[int]): Specific class to explain (None = all classes)
- `method` (str): Interpretation method
- `target_layer` (Optional): Layer for interpretation
- `show_uncertainty` (bool): Whether to visualize prediction uncertainty
- `uncertainty_method` (str): Uncertainty quantification method. Options: `'entropy'` (Shannon entropy of class probabilities), `'margin'` (difference between top-2 probabilities)
- `colormap` (str): Colormap for heatmaps
- `alpha` (float): Overlay transparency
- `save_path` (Optional[str]): Path to save visualization

**Returns:**
- `Dict`: Contains:
  - `'prediction'`: Predicted segmentation mask
  - `'heatmap'`: Attribution heatmap (if target_class specified)
  - `'uncertainty'`: Uncertainty map (if show_uncertainty=True)
  - `'visualization'`: Combined visualization

### Basic Usage

```python
from autotimm import SemanticSegmentor
from autotimm.interpretation import explain_segmentation
from PIL import Image

# Load model
model = SemanticSegmentor.load_from_checkpoint("segmentor.ckpt")

# Load image
image = Image.open("cityscape.jpg")

# Explain segmentation
results = explain_segmentation(
    model,
    image,
    save_path='segmentation_explanation.png'
)

# Check predictions
mask = results['prediction']
unique_classes = np.unique(mask)
print(f"Detected classes: {unique_classes}")
```

### Explain Specific Class

Focus on a particular class (e.g., "road" or "car"):

```python
# Explain class 0 (e.g., "road")
results = explain_segmentation(
    model,
    image,
    target_class=0,
    method='gradcam',
    save_path='road_explanation.png'
)

# The heatmap shows what influenced the "road" predictions
heatmap = results['heatmap']
```

### With Uncertainty Quantification

Visualize model uncertainty to identify ambiguous regions:

```python
results = explain_segmentation(
    model,
    image,
    show_uncertainty=True,
    uncertainty_method='entropy',
    save_path='segmentation_with_uncertainty.png'
)

# High uncertainty regions (entropy-based)
uncertainty = results['uncertainty']
high_uncertainty = uncertainty > 0.8  # Threshold
print(f"High uncertainty pixels: {high_uncertainty.sum()}")
```

### Entropy vs. Margin Uncertainty

**Entropy**: Measures overall prediction confidence across all classes
- High entropy = model uncertain across many classes
- Low entropy = model confident in prediction

**Margin**: Measures gap between top-2 predictions
- Low margin = close call between two classes
- High margin = clear winner

```python
# Entropy uncertainty (overall confidence)
results_entropy = explain_segmentation(
    model,
    image,
    show_uncertainty=True,
    uncertainty_method='entropy',
    save_path='uncertainty_entropy.png'
)

# Margin uncertainty (binary confusion)
results_margin = explain_segmentation(
    model,
    image,
    show_uncertainty=True,
    uncertainty_method='margin',
    save_path='uncertainty_margin.png'
)
```

### Multi-Class Analysis

Analyze multiple classes of interest:

```python
import matplotlib.pyplot as plt

classes_of_interest = [0, 1, 2]  # e.g., road, sidewalk, building
class_names = ['road', 'sidewalk', 'building']

fig, axes = plt.subplots(1, len(classes_of_interest), figsize=(15, 5))

for idx, (class_id, class_name) in enumerate(zip(classes_of_interest, class_names)):
    results = explain_segmentation(
        model,
        image,
        target_class=class_id,
        method='gradcam',
    )

    axes[idx].imshow(results['heatmap'], cmap='hot')
    axes[idx].set_title(f'{class_name} (class {class_id})')
    axes[idx].axis('off')

plt.savefig('multi_class_analysis.png')
```

### Understanding the Output

```python
results = {
    'prediction': np.ndarray,      # Shape (H, W), dtype int (class IDs)
    'heatmap': np.ndarray,         # Shape (H, W), dtype float [0, 1] (if target_class)
    'uncertainty': np.ndarray,     # Shape (H, W), dtype float [0, 1] (if show_uncertainty)
    'visualization': np.ndarray,   # Shape (H, W, 3), dtype uint8 (if save_path)
}
```

### Visualization Format

The saved visualization includes:
- **Top left**: Original image
- **Top right**: Predicted segmentation mask (colored by class)
- **Bottom left**: Interpretation heatmap (if target_class specified)
- **Bottom right**: Uncertainty map (if show_uncertainty=True)

---

## Advanced Use Cases

### 1. False Positive Analysis (Detection)

Investigate why model detected a false positive:

```python
# Get all detections
results = explain_detection(model, image, detection_threshold=0.3)

# Analyze low-confidence detections (potential false positives)
for det in results['detections']:
    if det['score'] < 0.5:
        print(f"Low-confidence detection:")
        print(f"  Class: {det['class_name']}")
        print(f"  Score: {det['score']:.3f}")
        # Check heatmap to see what triggered detection
```

### 2. Boundary Refinement Analysis (Segmentation)

Identify where segmentation boundaries are uncertain:

```python
results = explain_segmentation(
    model,
    image,
    show_uncertainty=True,
    uncertainty_method='margin',
)

uncertainty = results['uncertainty']

# Find uncertain boundaries
from scipy import ndimage
edges = ndimage.sobel(results['prediction'])
uncertain_edges = (edges > 0) & (uncertainty > 0.7)

print(f"Uncertain boundary pixels: {uncertain_edges.sum()}")
```

### 3. Class Confusion Analysis (Segmentation)

Understand which classes the model confuses:

```python
# Get predictions and uncertainty
results = explain_segmentation(
    model,
    image,
    show_uncertainty=True,
    uncertainty_method='entropy',
)

prediction = results['prediction']
uncertainty = results['uncertainty']

# Find high-uncertainty regions for each class
for class_id in range(num_classes):
    class_mask = (prediction == class_id)
    class_uncertainty = uncertainty[class_mask]

    if len(class_uncertainty) > 0:
        print(f"Class {class_id}:")
        print(f"  Mean uncertainty: {class_uncertainty.mean():.3f}")
        print(f"  Highly uncertain pixels: {(class_uncertainty > 0.8).sum()}")
```

### 4. Multi-Scale Detection Analysis

Analyze detections at different scales:

```python
from torchvision import transforms

scales = [0.5, 1.0, 1.5]

for scale in scales:
    # Resize image
    resize = transforms.Resize((int(image.height * scale), int(image.width * scale)))
    scaled_image = resize(image)

    # Explain detections
    results = explain_detection(
        model,
        scaled_image,
        detection_threshold=0.5,
        save_path=f'detection_scale_{scale}.png'
    )

    print(f"Scale {scale}: {len(results['detections'])} detections")
```

### 5. Temporal Consistency (Video)

Check interpretation consistency across video frames:

```python
import cv2

video = cv2.VideoCapture("video.mp4")
frame_heatmaps = []

for frame_idx in range(0, 100, 10):  # Sample every 10 frames
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    results = explain_detection(
        model,
        frame_pil,
        detection_threshold=0.5,
    )

    if results['heatmaps']:
        frame_heatmaps.append(results['heatmaps'][0])  # First detection

# Analyze temporal stability
heatmap_diffs = []
for i in range(len(frame_heatmaps) - 1):
    diff = np.abs(frame_heatmaps[i+1] - frame_heatmaps[i]).mean()
    heatmap_diffs.append(diff)

print(f"Mean heatmap change: {np.mean(heatmap_diffs):.3f}")
```

---

## Performance Considerations

### Object Detection

**Factors affecting speed:**
- Number of detections (more = slower)
- Image resolution (larger = slower)
- Interpretation method (GradCAM fastest, Integrated Gradients slowest)

**Optimization tips:**
```python
# Process only top detections
results = explain_detection(
    model,
    image,
    max_detections=5,           # Limit explanations
    detection_threshold=0.7,     # Higher threshold
)

# Use faster method
results = explain_detection(
    model,
    image,
    method='gradcam',            # Faster than integrated_gradients
)
```

### Semantic Segmentation

**Factors affecting speed:**
- Image resolution (major factor)
- Target class (None = faster, specific class = slower)
- Uncertainty calculation (adds overhead)

**Optimization tips:**
```python
# Disable uncertainty for speed
results = explain_segmentation(
    model,
    image,
    show_uncertainty=False,      # Faster
)

# Resize image for faster processing
from torchvision import transforms
resize = transforms.Resize((512, 512))
small_image = resize(image)

results = explain_segmentation(model, small_image)
```

---

## Troubleshooting

### Detection: No Explanations Generated

**Problem:** Empty detections list

**Solutions:**
- Lower `detection_threshold`
- Verify model is trained
- Check image preprocessing
- Try different images

```python
# Debug: Check raw detections
results = explain_detection(
    model,
    image,
    detection_threshold=0.1,  # Very low threshold
)
print(f"Number of detections: {len(results['detections'])}")
```

### Detection: Poor Localization

**Problem:** Heatmaps don't align with objects

**Solutions:**
- Try GradCAM++ instead of GradCAM
- Specify a different target layer
- Check bbox coordinates

```python
results = explain_detection(
    model,
    image,
    method='gradcam++',          # Better for multiple objects
    target_layer='backbone.layer3',  # Try different layer
)
```

### Segmentation: Blank Heatmaps

**Problem:** Heatmap is all zeros

**Solutions:**
- Verify target_class exists in prediction
- Check that model is trained
- Try different target layer

```python
# Check if class exists in prediction
results = explain_segmentation(model, image)
prediction = results['prediction']
unique_classes = np.unique(prediction)
print(f"Present classes: {unique_classes}")

# Explain only present classes
for class_id in unique_classes:
    results = explain_segmentation(
        model,
        image,
        target_class=int(class_id),
        save_path=f'class_{class_id}_explanation.png'
    )
```

### High Memory Usage

**Problem:** Out of memory errors

**Solutions:**
- Reduce image resolution
- Process fewer detections
- Disable uncertainty calculation

```python
# Resize image
from torchvision import transforms
resize = transforms.Resize((512, 512))
small_image = resize(image)

# Explain with reduced memory
results = explain_detection(
    model,
    small_image,
    max_detections=3,            # Fewer detections
)
```

---

## Comparison with Classification

| Aspect | Classification | Detection | Segmentation |
|--------|---------------|-----------|--------------|
| **Output** | Single class | Multiple bboxes | Per-pixel class |
| **Target** | Whole image | Per-detection | Per-class or all |
| **Heatmap** | Single | Per-bbox | Single or per-class |
| **Uncertainty** | N/A | Per-detection confidence | Per-pixel entropy/margin |
| **Speed** | Fast | Medium (× num_detections) | Slower (depends on resolution) |

---

## See Also

- [Interpretation Methods](methods.md) - Available interpretation methods
- [Feature Visualization](feature-visualization.md) - Analyze learned features
- [Main Guide](index.md) - Overview and quick start
