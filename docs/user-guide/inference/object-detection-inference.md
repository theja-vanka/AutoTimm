# Object Detection Inference

This guide covers how to use trained object detection models for inference and prediction.

## Loading a Detection Model

```python
from autotimm import ObjectDetector, MetricConfig, TransformConfig

# Define metrics
metrics = [
    MetricConfig(
        name="mAP",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy"},
        stages=["val"],
    ),
]

# Load model with TransformConfig for preprocessing
model = ObjectDetector.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
    transform_config=TransformConfig(),  # Enable preprocess()
)
model.eval()
```

---

## Single Image Detection (Recommended)

Use the built-in `preprocess()` method for correct model-specific normalization:

```python
import torch
from PIL import Image

# Load image
image = Image.open("image.jpg").convert("RGB")

# Preprocess using model's native normalization
input_tensor = model.preprocess(image)  # Returns (1, 3, 640, 640)

# Detect objects
with torch.no_grad():
    detections = model.predict_step(input_tensor, batch_idx=0)
```

---

## Single Image Detection (Manual)

If you need manual control over transforms:

```python
import torch
from PIL import Image
from torchvision import transforms

# Prepare transform
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform image
image = Image.open("image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # (1, 3, 640, 640)

# Detect objects
model.eval()
with torch.no_grad():
    detections = model.predict_step(input_tensor, batch_idx=0)

# detections is a dict with:
# - "boxes": Tensor of shape (N, 4) in [x1, y1, x2, y2] format
# - "scores": Tensor of shape (N,) with confidence scores
# - "labels": Tensor of shape (N,) with class indices

boxes = detections["boxes"]
scores = detections["scores"]
labels = detections["labels"]

print(f"Found {len(boxes)} objects:")
for box, score, label in zip(boxes, scores, labels):
    print(f"  Class {label.item()}: {score.item():.2%} confidence at {box.tolist()}")
```

**Tip:** Use `model.get_data_config()` to get the correct normalization values for manual transforms:

```python
config = model.get_data_config()
print(f"Mean: {config['mean']}")
print(f"Std: {config['std']}")
print(f"Input size: {config['input_size']}")
```

---

## Visualize Detections

```python
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def visualize_detections(
    image_path, 
    boxes, 
    labels, 
    scores, 
    class_names=None, 
    threshold=0.5,
    figsize=(12, 8)
):
    """Visualize object detection results."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    # Filter by confidence threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1

        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        class_name = class_names[label] if class_names else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        ax.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(facecolor='red', alpha=0.5),
            fontsize=10, color='white'
        )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig('detections.jpg', dpi=150, bbox_inches='tight')
    plt.show()


# Usage
visualize_detections(
    "image.jpg",
    boxes=detections["boxes"],
    labels=detections["labels"],
    scores=detections["scores"],
    class_names=["person", "bicycle", "car", ...],  # COCO classes
    threshold=0.3,
)
```

---

## Batch Detection

```python
from torch.utils.data import DataLoader
from autotimm import DetectionDataModule

# Prepare data
data = DetectionDataModule(
    data_dir="./test_images",
    image_size=640,
    batch_size=8,
)
data.setup("test")

# Batch inference
model.eval()
all_detections = []

with torch.no_grad():
    for batch in data.test_dataloader():
        images = batch["image"]
        batch_detections = model.predict_step(images, batch_idx=0)
        all_detections.append(batch_detections)

# Process results
for i, dets in enumerate(all_detections):
    print(f"Batch {i}: Found {len(dets['boxes'])} objects")
```

---

## Complete Detection Pipeline

Production-ready detection pipeline with TransformConfig:

```python
import torch
from PIL import Image
from autotimm import ObjectDetector, MetricConfig, TransformConfig


class DetectionPipeline:
    """End-to-end object detection pipeline."""

    def __init__(
        self,
        checkpoint_path,
        backbone,
        num_classes,
        class_names=None,
        score_threshold=0.3,
        image_size=640,
    ):
        # Load model with TransformConfig for preprocessing
        metrics = [
            MetricConfig(
                name="mAP",
                backend="torchmetrics",
                metric_class="MeanAveragePrecision",
                params={"box_format": "xyxy"},
                stages=["val"],
            ),
        ]

        self.model = ObjectDetector.load_from_checkpoint(
            checkpoint_path,
            backbone=backbone,
            num_classes=num_classes,
            metrics=metrics,
            transform_config=TransformConfig(image_size=image_size),  # Enable preprocess()
        )
        self.model.eval()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.class_names = class_names
        self.score_threshold = score_threshold
        self.image_size = image_size

    def predict(self, image_path):
        """Detect objects in a single image."""
        # Load and preprocess using model's native normalization
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        input_tensor = self.model.preprocess(image).to(self.device)

        # Detect
        with torch.no_grad():
            detections = self.model.predict_step(input_tensor, batch_idx=0)

        # Filter by threshold
        keep = detections["scores"] >= self.score_threshold
        boxes = detections["boxes"][keep]
        scores = detections["scores"][keep]
        labels = detections["labels"][keep]

        # Scale boxes back to original image size
        scale_x = original_size[0] / self.image_size
        scale_y = original_size[1] / self.image_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Format results
        results = []
        for box, score, label in zip(boxes, scores, labels):
            class_name = self.class_names[label] if self.class_names else label.item()
            results.append({
                "class": class_name,
                "class_index": label.item(),
                "confidence": score.item(),
                "bbox": box.cpu().tolist(),  # [x1, y1, x2, y2]
            })

        return results

    def predict_batch(self, image_paths):
        """Detect objects in multiple images."""
        return [self.predict(path) for path in image_paths]


# Usage
pipeline = DetectionPipeline(
    checkpoint_path="best-detector.ckpt",
    backbone="resnet50",
    num_classes=80,
    class_names=["person", "bicycle", "car", ...],  # COCO classes
    score_threshold=0.3,
    image_size=640,
)

# Detect objects
results = pipeline.predict("test_image.jpg")
for det in results:
    print(f"{det['class']}: {det['confidence']:.2%} at {det['bbox']}")
```

---

## COCO Class Names

For COCO datasets, use these 80 class names:

```python
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]

# Use with pipeline
pipeline = DetectionPipeline(
    checkpoint_path="best-detector.ckpt",
    backbone="resnet50",
    num_classes=80,
    class_names=COCO_CLASSES,
)
```

---

## Detection Tuning

### Adjust Score Threshold

Control the confidence threshold for detections:

```python
# Low threshold - more detections, more false positives
pipeline = DetectionPipeline(score_threshold=0.1)

# Medium threshold - balanced (recommended)
pipeline = DetectionPipeline(score_threshold=0.3)

# High threshold - fewer detections, more precision
pipeline = DetectionPipeline(score_threshold=0.7)
```

### NMS Threshold

Adjust Non-Maximum Suppression to control duplicate detection:

```python
# During model creation
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    nms_thresh=0.5,  # Default
)

# Lower = stricter NMS (fewer duplicates)
model = ObjectDetector(nms_thresh=0.3)

# Higher = more lenient NMS (may have duplicates)
model = ObjectDetector(nms_thresh=0.7)
```

### Image Size

Balance between speed and accuracy:

```python
# Fast inference (smaller objects may be missed)
pipeline = DetectionPipeline(image_size=512)

# Balanced (recommended)
pipeline = DetectionPipeline(image_size=640)

# Better small object detection (slower)
pipeline = DetectionPipeline(image_size=800)

# Maximum accuracy (much slower)
pipeline = DetectionPipeline(image_size=1024)
```

---

## Performance Tips

### 1. Batch Processing

Process multiple images in one forward pass:

```python
def predict_batch_efficient(model, image_paths, transform, device, batch_size=8):
    """Efficient batch detection."""
    model.eval()
    all_results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Load and transform batch
        images = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))

        # Stack into batch
        batch_tensor = torch.stack(images).to(device)

        # Detect
        with torch.no_grad():
            detections = model.predict_step(batch_tensor, batch_idx=0)

        all_results.append(detections)

    return all_results
```

### 2. GPU Inference

Always use GPU for detection when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Also move input to GPU
input_tensor = input_tensor.to(device)
```

### 3. Optimal Settings

Recommended settings for different scenarios:

**Speed-optimized:**
```python
pipeline = DetectionPipeline(
    image_size=512,
    score_threshold=0.5,
    backbone="resnet34",  # Smaller backbone
)
```

**Balanced:**
```python
pipeline = DetectionPipeline(
    image_size=640,
    score_threshold=0.3,
    backbone="resnet50",
)
```

**Accuracy-optimized:**
```python
pipeline = DetectionPipeline(
    image_size=800,
    score_threshold=0.1,  # More detections
    backbone="resnet101",  # Larger backbone
)
```

---

## Saving Detection Results

### Save to JSON

```python
import json

results = pipeline.predict("image.jpg")

# Save to JSON
with open("detections.json", "w") as f:
    json.dump(results, f, indent=2)

# Load from JSON
with open("detections.json", "r") as f:
    loaded_results = json.load(f)
```

### Save Annotated Image

```python
from PIL import Image, ImageDraw, ImageFont

def save_annotated_image(image_path, detections, output_path, class_names=None):
    """Save image with drawn bounding boxes."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det["bbox"]
        class_name = det["class"]
        confidence = det["confidence"]

        # Draw rectangle
        draw.rectangle(bbox, outline="red", width=3)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((bbox[0], bbox[1] - 20), label, fill="white", font=font)

    image.save(output_path)

# Usage
results = pipeline.predict("input.jpg")
save_annotated_image("input.jpg", results, "output.jpg", class_names=COCO_CLASSES)
```

---

## Common Issues

### No Detections

**Problem:** Model returns empty results

**Solutions:**
```python
# 1. Lower the score threshold
pipeline = DetectionPipeline(score_threshold=0.1)

# 2. Check image preprocessing
# Ensure image is RGB
image = Image.open("img.jpg").convert("RGB")

# 3. Verify model is loaded correctly
print(f"Model num_classes: {model.num_classes}")

# 4. Check if objects are in the training classes
```

### Too Many Duplicate Detections

**Problem:** Same object detected multiple times

**Solutions:**
```python
# 1. Lower NMS threshold (stricter)
model = ObjectDetector(nms_thresh=0.3)

# 2. Increase score threshold
pipeline = DetectionPipeline(score_threshold=0.5)
```

### Missing Small Objects

**Problem:** Small objects not detected

**Solutions:**
```python
# 1. Use larger image size
pipeline = DetectionPipeline(image_size=800)

# 2. Lower score threshold
pipeline = DetectionPipeline(score_threshold=0.1)

# 3. Use model trained on higher resolution
```

### Slow Inference

**Problem:** Detection is too slow

**Solutions:**
```python
# 1. Use smaller image size
pipeline = DetectionPipeline(image_size=512)

# 2. Use smaller backbone
model = ObjectDetector(backbone="resnet34")

# 3. Increase score threshold (fewer post-processing)
pipeline = DetectionPipeline(score_threshold=0.5)

# 4. Process in batches on GPU
```

---

## See Also

- [Classification Inference](classification-inference.md) - For classification models
- [Model Export](model-export.md) - Export to TorchScript/ONNX
- [Object Detection Examples](../../examples/tasks/object-detection.md) - More examples
