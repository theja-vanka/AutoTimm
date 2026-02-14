# Model Export

This guide covers how to export trained AutoTimm models to production-ready formats like TorchScript and ONNX.

## TorchScript Export

TorchScript allows you to serialize PyTorch models for deployment in C++ applications or environments without Python.

### Basic TorchScript Export

```python
import torch
from autotimm import ImageClassifier, MetricConfig

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

# Create example input
example_input = torch.randn(1, 3, 224, 224)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save("model_scripted.pt")
```

### Using Traced Model

```python
import torch

# Load traced model
loaded_model = torch.jit.load("model_scripted.pt")
loaded_model.eval()

# Use for inference
with torch.inference_mode():
    input_tensor = torch.randn(1, 3, 224, 224)
    output = loaded_model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
```

### Script Mode (Alternative)

For models with control flow, use script mode instead of trace:

```python
# Script the model (preserves control flow)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

**When to use:**

- **Trace**: Faster, works for most models, but doesn't preserve control flow
- **Script**: Preserves control flow (if/else, loops), but may be slower

---

## ONNX Export

ONNX (Open Neural Network Exchange) is a cross-platform format supported by many inference engines including ONNX Runtime, TensorRT, OpenVINO, and CoreML.

### Basic ONNX Export

```python
from autotimm import ImageClassifier, export_to_onnx
import torch

# Load model
model = ImageClassifier.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# Export to ONNX
example_input = torch.randn(1, 3, 224, 224)
export_to_onnx(model, "model.onnx", example_input)

# Or use the convenience method
model.to_onnx("model.onnx")
```

### Verify and Validate

```python
from autotimm.export import validate_onnx_export

# Validate outputs match original model
is_valid = validate_onnx_export(
    original_model=model,
    onnx_path="model.onnx",
    example_input=example_input,
)
print(f"Export valid: {is_valid}")
```

### Checkpoint to ONNX (One Step)

```python
from autotimm import export_checkpoint_to_onnx, ImageClassifier

path = export_checkpoint_to_onnx(
    checkpoint_path="checkpoint.ckpt",
    save_path="model.onnx",
    model_class=ImageClassifier,
    example_input=torch.randn(1, 3, 224, 224),
)
```

---

## ONNX Inference

### Using ONNX Runtime

```python
from autotimm import load_onnx
import numpy as np

# Load model (validates integrity automatically)
session = load_onnx("model.onnx")

# Or load directly with ONNX Runtime (no AutoTimm dependency)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")

# Run inference
input_name = session.get_inputs()[0].name
image = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {input_name: image})

# Get predictions
logits = outputs[0]
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
predicted_class = np.argmax(probs, axis=1)[0]
confidence = probs[0, predicted_class]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

### ONNX Runtime Optimization

```python
import onnxruntime as ort

# Set execution providers (GPU support)
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Session options for optimization
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load with optimization
session = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=providers
)

# Check which provider is being used
print(f"Using: {session.get_providers()}")
```

---

## Export Object Detection Models

### Detection Model to TorchScript

```python
import torch
from autotimm import ObjectDetector, MetricConfig

# Load detection model
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

# Trace with example input
example_input = torch.randn(1, 3, 640, 640)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("detector_scripted.pt")
```

### Detection Model to ONNX

```python
from autotimm import ObjectDetector, export_to_onnx
import torch

# Export detection model
model = ObjectDetector.load_from_checkpoint("detector.ckpt")
model.eval()

example_input = torch.randn(1, 3, 640, 640)
export_to_onnx(model, "detector.onnx", example_input)

# Or use convenience method
model.to_onnx("detector.onnx", example_input=example_input)
# Detection outputs are automatically flattened: cls_l0..cls_l4, reg_l0..reg_l4, ctr_l0..ctr_l4
```

---

## Quantization

Reduce model size and increase inference speed with quantization.

### Dynamic Quantization

```python
import torch
from autotimm import ImageClassifier, MetricConfig

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

# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), "model_quantized.pth")

# Model is now smaller and faster
# Note: Slight accuracy drop is expected
```

### Static Quantization (More Accurate)

```python
import torch
from torch.quantization import get_default_qconfig, prepare, convert

# Load model
model = ImageClassifier.load_from_checkpoint(...)
model.eval()

# Set quantization config
model.qconfig = get_default_qconfig('fbgemm')

# Prepare for quantization
model_prepared = prepare(model)

# Calibrate with representative data
with torch.inference_mode():
    for data in calibration_dataloader:
        model_prepared(data)

# Convert to quantized model
model_quantized = convert(model_prepared)

# Save
torch.save(model_quantized.state_dict(), "model_quantized.pth")
```

---

## Model Optimization Comparison

| Method | Size Reduction | Speed Increase | Accuracy Impact | Deployment |
|--------|----------------|----------------|-----------------|------------|
| **TorchScript** | None | 10-20% | None | PyTorch C++ |
| **ONNX** | None | 20-30% | None | Cross-platform |
| **Dynamic Quant** | 4x | 2-3x | 1-2% drop | PyTorch |
| **Static Quant** | 4x | 2-4x | 0.5-1% drop | PyTorch |
| **FP16** | 2x | 2-3x | <0.5% drop | GPU only |

---

## Deployment Example

Complete pipeline for deploying a quantized ONNX model:

```python
import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
from autotimm import ImageClassifier, export_to_onnx

# 1. Load and prepare model
model = ImageClassifier.load_from_checkpoint(...)
model.eval()

# 2. Export to ONNX
example_input = torch.randn(1, 3, 224, 224)
export_to_onnx(model, "model.onnx", example_input)

# 3. Optimize ONNX model
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8
)

# 4. Create inference session
session = ort.InferenceSession(
    "model_quantized.onnx",
    providers=['CPUExecutionProvider']
)

# 5. Inference function
def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy()

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    
    return {
        "class": int(np.argmax(probs)),
        "confidence": float(np.max(probs))
    }

# Use
result = predict("test.jpg")
print(f"Class: {result['class']}, Confidence: {result['confidence']:.2%}")
```

---

## Performance Tips

### 1. Choose the Right Format

- **Development/Research**: Use PyTorch checkpoints
- **Production (PyTorch ecosystem)**: Use TorchScript
- **Cross-platform deployment**: Use ONNX
- **Mobile/Edge**: Use ONNX + quantization or TorchScript Mobile

### 2. Optimize for Target Hardware

```python
# For CPU deployment
providers = ['CPUExecutionProvider']

# For GPU deployment
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# For specific hardware (TensorRT on NVIDIA)
providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 3. Batch Inference

Export with dynamic batch size for flexibility:

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    dynamic_axes={
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
)

# Then use with any batch size
batch = np.random.randn(8, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {"image": batch})
```

---

## Common Issues

For model export issues, see the [Troubleshooting - Export & Inference](../../troubleshooting/deployment/export-inference.md) including:

- ONNX export fails
- Model size too large
- Inference speed not improved
- Format compatibility issues

# 2. Enable optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 3. Use quantization
# 4. Process in batches
```

---

## See Also

- [ONNX Export Guide](../deployment/onnx-export.md) - Complete ONNX export guide
- [TorchScript Export Guide](../deployment/torchscript-export.md) - Complete TorchScript export guide
- [Classification Inference](classification-inference.md) - Inference with PyTorch models
- [Object Detection Inference](object-detection-inference.md) - Detection model inference
- [Training Guide](../training/training.md) - How to train models
