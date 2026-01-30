# Inference

This guide covers how to use trained AutoTimm models for inference and prediction.

## Basic Inference

### Load a Trained Model

```python
import torch
from autotimm import ImageClassifier, MetricConfig

# Define metrics (required for loading)
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
    ),
]

# Load from checkpoint
model = ImageClassifier.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)
model.eval()
```

### Single Image Prediction

```python
from PIL import Image
from torchvision import transforms

# Prepare transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform image
image = Image.open("image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities.max().item()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

## Batch Prediction

### Using DataLoader

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Prepare dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("path/to/images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Predict all
model.eval()
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, _ in dataloader:
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_predictions.extend(preds.tolist())
        all_probabilities.extend(probs.tolist())
```

### Using Trainer.predict

```python
from autotimm import AutoTrainer, ImageDataModule

data = ImageDataModule(
    data_dir="./test_images",
    image_size=224,
    batch_size=32,
)
data.setup("test")

trainer = AutoTrainer()
predictions = trainer.predict(model, dataloaders=data.test_dataloader())

# predictions is a list of batched probability tensors
all_probs = torch.cat(predictions, dim=0)
all_preds = all_probs.argmax(dim=1)
```

## GPU Inference

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

with torch.no_grad():
    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
```

## Top-K Predictions

```python
def get_topk_predictions(model, image_tensor, k=5, class_names=None):
    """Get top-k predictions with class names and probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        topk_probs, topk_indices = probs.topk(k, dim=1)

    results = []
    for i in range(k):
        idx = topk_indices[0, i].item()
        prob = topk_probs[0, i].item()
        name = class_names[idx] if class_names else str(idx)
        results.append({"class": name, "probability": prob})

    return results

# Usage
results = get_topk_predictions(model, input_tensor, k=5, class_names=data.class_names)
for r in results:
    print(f"{r['class']}: {r['probability']:.2%}")
```

## Export for Production

### TorchScript

```python
# Trace the model
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save("model_scripted.pt")

# Load and use
loaded_model = torch.jit.load("model_scripted.pt")
output = loaded_model(input_tensor)
```

### ONNX Export

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=11,
)

# Verify
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### ONNX Inference

```python
import onnxruntime as ort
import numpy as np

# Load
session = ort.InferenceSession("model.onnx")

# Prepare input
image_np = input_tensor.numpy()

# Run inference
outputs = session.run(None, {"image": image_np})
logits = outputs[0]
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
```

## Inference Pipeline

Complete inference pipeline example:

```python
import torch
from PIL import Image
from torchvision import transforms
from autotimm import ImageClassifier, MetricConfig


class InferencePipeline:
    """End-to-end inference pipeline."""

    def __init__(self, checkpoint_path, backbone, num_classes, class_names=None):
        # Define minimal metrics for loading
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["val"],
            ),
        ]

        self.model = ImageClassifier.load_from_checkpoint(
            checkpoint_path,
            backbone=backbone,
            num_classes=num_classes,
            metrics=metrics,
        )
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.class_names = class_names

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, image_path):
        """Predict class for a single image."""
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs.max().item()

        pred_class = self.class_names[pred_idx] if self.class_names else pred_idx

        return {
            "class": pred_class,
            "class_index": pred_idx,
            "confidence": confidence,
            "probabilities": probs[0].cpu().tolist(),
        }

    def predict_batch(self, image_paths):
        """Predict classes for multiple images."""
        return [self.predict(path) for path in image_paths]


# Usage
pipeline = InferencePipeline(
    checkpoint_path="best-model.ckpt",
    backbone="resnet50",
    num_classes=10,
    class_names=["cat", "dog", "bird", ...],
)

result = pipeline.predict("test_image.jpg")
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")
```

## Object Detection Inference

### Load Detection Model

```python
from autotimm import ObjectDetector, MetricConfig

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

# Load model
model = ObjectDetector.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
)
model.eval()
```

### Single Image Detection

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

### Visualize Detections

```python
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def visualize_detections(image_path, boxes, labels, scores, class_names=None, threshold=0.5):
    """Visualize object detection results."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(12, 8))
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

### Batch Detection

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
```

### Detection Pipeline

Complete detection pipeline with post-processing:

```python
import torch
from PIL import Image
from torchvision import transforms
from autotimm import ObjectDetector, MetricConfig


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
        # Load model
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
        )
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.class_names = class_names
        self.score_threshold = score_threshold
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, image_path):
        """Detect objects in a single image."""
        # Load and transform
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

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

### COCO Class Names

For COCO datasets, use these class names:

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

pipeline = DetectionPipeline(
    checkpoint_path="best-detector.ckpt",
    backbone="resnet50",
    num_classes=80,
    class_names=COCO_CLASSES,
)
```

### Detection Performance Tips

1. **NMS Threshold**: Adjust `nms_thresh` in the model to control duplicate detection suppression
   ```python
   model = ObjectDetector(
       backbone="resnet50",
       num_classes=80,
       nms_thresh=0.5,  # Default, lower = stricter NMS
   )
   ```

2. **Score Threshold**: Higher threshold = fewer but more confident detections
   ```python
   model = ObjectDetector(
       backbone="resnet50",
       num_classes=80,
       score_thresh=0.05,  # Default for training
   )
   # For inference, filter manually with higher threshold (0.3-0.5)
   ```

3. **Image Size**: Larger images detect smaller objects better but slower
   - 512×512: Faster inference
   - 640×640: Balanced (recommended)
   - 800×800 or 1024×1024: Better small object detection

4. **Batch Processing**: Process multiple images in one forward pass
   ```python
   images = torch.stack([transform(img) for img in image_list])
   with torch.no_grad():
       detections = model.predict_step(images, batch_idx=0)
   ```

## Performance Tips

1. **Batch predictions** - Process multiple images at once for GPU efficiency
2. **Use fp16** - Reduce memory and speed up inference:
   ```python
   model = model.half()
   input_tensor = input_tensor.half()
   ```
3. **Disable gradient tracking** - Always use `torch.no_grad()`
4. **Use TorchScript** - For production deployment
5. **Pre-load model** - Keep model in memory between predictions
