# Inference

This guide covers how to use trained AutoTimm models for inference and prediction.

## Inference Pipeline

```mermaid
graph TD
    A[Trained Model] --> A1[Locate Checkpoint]
    A1 --> A2[Verify Path]
    A2 --> B[Load Checkpoint]
    B --> B1[Load State Dict]
    B1 --> B2[Restore Weights]
    B2 --> B3[Load Hyperparams]
    B3 --> C[Set Eval Mode]
    
    C --> C1[Disable Dropout]
    C1 --> C2[Fix BatchNorm]
    C2 --> C3[No Gradient Tracking]
    C3 --> D[Preprocess Image]
    
    D --> D1[Load Image]
    D1 --> D2[Apply Transforms]
    D2 --> D3[Resize]
    D3 --> D4[Normalize]
    D4 --> D5[To Tensor]
    D5 --> D6[Add Batch Dim]
    D6 --> E{Inference}

    E -->|Classification| F1[Logits]
    F1 --> F1a[Forward Pass]
    F1a --> F1b[Extract Logits]
    
    E -->|Multi-Label| F4[Logits]
    F4 --> F4a[Forward Pass]
    F4a --> F4b[Extract Logits]
    
    E -->|Detection| F2[Boxes + Classes]
    F2 --> F2a[Forward Pass]
    F2a --> F2b[Decode Predictions]
    
    E -->|Segmentation| F3[Pixel Masks]
    F3 --> F3a[Forward Pass]
    F3a --> F3b[Extract Masks]

    F1b --> G1[Softmax]
    G1 --> G1a[Class Probabilities]
    G1a --> G1b[Top-K Classes]
    
    F4b --> G4[Sigmoid + Threshold]
    G4 --> G4a[Class Probabilities]
    G4a --> G4b[Apply Threshold]
    G4b --> G4c[Multi-label Output]
    
    F2b --> G2[NMS]
    G2 --> G2a[Filter by Score]
    G2a --> G2b[IoU Suppression]
    G2b --> G2c[Final Detections]
    
    F3b --> G3[Argmax]
    G3 --> G3a[Class per Pixel]
    G3a --> G3b[Create Mask]

    G1b --> H[Predictions]
    G4c --> H
    G2c --> H
    G3b --> H

    H --> I[Post-process]
    I --> I1[Format Results]
    I1 --> I2[Add Metadata]
    I2 --> I3[Create Visualizations]
    I3 --> J[Results]
    
    J --> J1[Save Predictions]
    J1 --> J2[Generate Report]
    J2 --> J3[Output Files]

    style A fill:#2196F3,stroke:#1976D2,color:#fff
    style B fill:#42A5F5,stroke:#1976D2,color:#fff
    style C fill:#2196F3,stroke:#1976D2,color:#fff
    style D fill:#42A5F5,stroke:#1976D2,color:#fff
    style E fill:#2196F3,stroke:#1976D2,color:#fff
    style H fill:#42A5F5,stroke:#1976D2,color:#fff
    style I fill:#2196F3,stroke:#1976D2,color:#fff
    style J fill:#42A5F5,stroke:#1976D2,color:#fff
```

## Quick Links

- **[Classification Inference](classification-inference.md)** - Image classification model inference
- **[Object Detection Inference](object-detection-inference.md)** - Object detection model inference
- **[Semantic Segmentation Inference](semantic-segmentation-inference.md)** - Semantic segmentation model inference
- **[Model Export](model-export.md)** - Export to TorchScript, ONNX, and quantization

---

## Quick Start

### Classification

```python
from autotimm import ImageClassifier, MetricConfig, TransformConfig
import torch
from PIL import Image

metrics = [MetricConfig(name="accuracy", backend="torchmetrics",
                        metric_class="Accuracy", params={"task": "multiclass"},
                        stages=["val"])]
model = ImageClassifier.load_from_checkpoint(
    "checkpoint.ckpt",
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    transform_config=TransformConfig(),
)
model.eval()

image = Image.open("test.jpg").convert("RGB")
with torch.inference_mode():
    logits = model(model.preprocess(image))
    predicted_class = logits.argmax(dim=1).item()
```

### Multi-Label Classification

```python
from autotimm import ImageClassifier, MetricConfig, TransformConfig
import torch
from PIL import Image

model = ImageClassifier.load_from_checkpoint(
    "multilabel.ckpt",
    backbone="resnet50",
    num_classes=4,
    multi_label=True,
    threshold=0.5,
    transform_config=TransformConfig(),
)
model.eval()

image = Image.open("test.jpg").convert("RGB")
with torch.inference_mode():
    logits = model(model.preprocess(image))
    probs = logits.sigmoid().squeeze(0)        # per-label probabilities
    predicted = (probs > 0.5).nonzero().squeeze(-1).tolist()
```

### Object Detection

```python
from autotimm import ObjectDetector, MetricConfig, TransformConfig

model = ObjectDetector.load_from_checkpoint(
    "detector.ckpt",
    backbone="resnet50",
    num_classes=80,
    transform_config=TransformConfig(image_size=640),
)
model.eval()

with torch.inference_mode():
    detections = model.predict_step(model.preprocess(image), batch_idx=0)
```

### Model Export

```python
# TorchScript
traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
traced.save("model.pt")

# ONNX
torch.onnx.export(model, torch.randn(1, 3, 224, 224), "model.onnx")
```

---

## Performance Tips

| Optimization | Speed Gain | When to Use |
|--------------|------------|-------------|
| Batch processing | 2-5x | Multiple images |
| GPU inference | 10-50x | GPU available |
| FP16 precision | 2-3x | Tensor core GPUs |
| TorchScript | 10-20% | Production |
| ONNX Runtime | 20-40% | Cross-platform |

**Checklist:**

- Use `model.eval()` before inference
- Wrap predictions in `torch.inference_mode()`
- Process multiple images in batches
- Use GPU when available

---

## Detailed Guides

- **[Classification Inference](classification-inference.md)** - Single/batch prediction, top-K, multi-label, pipelines
- **[Object Detection Inference](object-detection-inference.md)** - Bounding boxes, visualization, NMS tuning
- **[Model Export](model-export.md)** - TorchScript, ONNX, quantization
