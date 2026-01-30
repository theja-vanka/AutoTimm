# Inference

This guide covers how to use trained AutoTimm models for inference and prediction in production.

## Quick Links

- **[Classification Inference](classification-inference.md)** - Image classification model inference
- **[Object Detection Inference](object-detection-inference.md)** - Object detection model inference  
- **[Model Export](model-export.md)** - Export to TorchScript, ONNX, and quantization

---

---

## Quick Start

### Classification Inference

```python
from autotimm import ImageClassifier, MetricConfig
import torch
from PIL import Image
from torchvision import transforms

# Load model
metrics = [MetricConfig(name="accuracy", backend="torchmetrics",
                        metric_class="Accuracy", params={"task": "multiclass"},
                        stages=["val"])]
model = ImageClassifier.load_from_checkpoint(
    "checkpoint.ckpt",
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open("test.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities.max().item()

print(f"Predicted: {predicted_class} ({confidence:.2%})")
```

👉 **[Full Classification Inference Guide](classification-inference.md)**

### Object Detection Inference

```python
from autotimm import ObjectDetector, MetricConfig
import torch
from PIL import Image
from torchvision import transforms

# Load model
metrics = [MetricConfig(name="mAP", backend="torchmetrics",
                        metric_class="MeanAveragePrecision",
                        params={"box_format": "xyxy"}, stages=["val"])]
model = ObjectDetector.load_from_checkpoint(
    "detector.ckpt",
    backbone="resnet50",
    num_classes=80,
    metrics=metrics,
)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open("test.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Detect objects
with torch.no_grad():
    detections = model.predict_step(input_tensor, batch_idx=0)

boxes = detections["boxes"]
scores = detections["scores"]
labels = detections["labels"]

print(f"Found {len(boxes)} objects")
for box, score, label in zip(boxes, scores, labels):
    print(f"  Class {label.item()}: {score.item():.2%} at {box.tolist()}")
```

👉 **[Full Object Detection Inference Guide](object-detection-inference.md)**

### Model Export

```python
import torch
import torch.onnx

# Export to TorchScript
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_scripted.pt")

# Export to ONNX
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=11,
)
```

👉 **[Full Model Export Guide](model-export.md)**

---

## Inference Options Comparison

### By Task

| Task | Guide | Key Features |
|------|-------|--------------|
| **Image Classification** | [Classification Inference](classification-inference.md) | Single/batch prediction, top-k results, probability scores |
| **Object Detection** | [Object Detection Inference](object-detection-inference.md) | Bounding boxes, class labels, visualization, NMS tuning |

### By Deployment Target

| Target | Method | Guide | Benefits |
|--------|--------|-------|----------|
| **Python/PyTorch** | Checkpoint loading | [Classification](classification-inference.md) / [Detection](object-detection-inference.md) | Full flexibility, easy debugging |
| **C++ Application** | TorchScript | [Model Export](model-export.md) | No Python dependency |
| **Cross-platform** | ONNX | [Model Export](model-export.md) | Hardware acceleration, mobile |
| **Mobile/Edge** | ONNX + Quantization | [Model Export](model-export.md) | Smaller size, faster inference |

---

## Inference Workflows

### Development Workflow

1. **Load checkpoint** using PyTorch
2. **Test predictions** on sample images  
3. **Evaluate performance** on test set
4. **Tune thresholds** for your use case

### Production Workflow

1. **Train and validate** model
2. **Export to production format** (TorchScript/ONNX)
3. **Optimize** (quantization, pruning)
4. **Deploy** to target environment
5. **Monitor** inference performance

---

## Performance Optimization

### General Tips

| Optimization | Speed Gain | Accuracy Impact | When to Use |
|--------------|------------|-----------------|-------------|
| **Batch processing** | 2-5x | None | Always for multiple images |
| **GPU inference** | 10-50x | None | When GPU available |
| **FP16 precision** | 2-3x | Minimal | GPU with tensor cores |
| **TorchScript** | 10-20% | None | Production deployment |
| **ONNX Runtime** | 20-40% | None | Cross-platform |
| **Quantization** | 2-4x | 1-2% | CPU/Edge deployment |

### Quick Performance Checklist

✅ Use `model.eval()` before inference  
✅ Wrap predictions in `torch.no_grad()`  
✅ Process multiple images in batches  
✅ Use GPU when available  
✅ Consider export formats for production  
✅ Profile and optimize bottlenecks  

---

## Common Inference Patterns

### Pattern 1: Real-time Single Image

Best for: Interactive applications, web APIs

```python
# Keep model loaded in memory
model = load_model()
model.eval()

# Fast inference for each request
def predict(image_path):
    with torch.no_grad():
        return model(preprocess(image_path))
```

### Pattern 2: Batch Processing

Best for: Processing large datasets, video analysis

```python
# Process in batches for efficiency
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    with torch.no_grad():
        predictions = model(batch)
    # Process predictions
```

### Pattern 3: Streaming

Best for: Video streams, continuous processing

```python
# Process frames as they arrive
for frame in video_stream:
    preprocessed = preprocess(frame)
    with torch.no_grad():
        result = model(preprocessed)
    # Display or save result
```

---

## Detailed Guides

### Classification Inference

Learn about:
- Loading classification models
- Single and batch prediction
- Top-K predictions
- Complete inference pipelines
- Performance optimization
- Common issues and solutions

**[Read the Classification Inference Guide →](classification-inference.md)**

### Object Detection Inference

Learn about:
- Loading detection models
- Single and batch detection
- Visualizing detections
- Complete detection pipelines
- COCO class names
- Detection tuning (NMS, thresholds)
- Saving results

**[Read the Object Detection Inference Guide →](object-detection-inference.md)**

### Model Export

Learn about:
- TorchScript export and usage
- ONNX export and inference
- Quantization techniques
- Export for object detection
- Optimization comparison
- Deployment examples
- Common issues

**[Read the Model Export Guide →](model-export.md)**

---

## See Also

- [Training Guide](training.md) - How to train models
- [Image Classification Data](image-classification-data.md) - Classification data loading
- [Object Detection Data](object-detection-data.md) - Detection data loading
- [Examples](../examples/) - Runnable code examples
