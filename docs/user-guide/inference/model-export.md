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
with torch.no_grad():
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

ONNX (Open Neural Network Exchange) is a cross-platform format supported by many inference engines.

### Basic ONNX Export

```python
import torch
import torch.onnx
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

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,                          # Model to export
    dummy_input,                    # Example input
    "model.onnx",                   # Output file
    input_names=["image"],          # Input names
    output_names=["logits"],        # Output names
    dynamic_axes={                  # Dynamic batch size
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=11,               # ONNX opset version
    do_constant_folding=True,       # Optimize constant folding
)

print("ONNX model exported successfully!")
```

### Verify ONNX Model

```python
import onnx

# Load and check model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Print model info
print(f"Graph: {onnx_model.graph}")
```

### Advanced ONNX Export Options

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,              # Export trained parameters
    opset_version=14,                # Latest opset for more ops
    do_constant_folding=True,        # Optimize
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},  # Dynamic batch
        "logits": {0: "batch_size"},
    },
    verbose=False,                   # Print conversion steps
)
```

---

## ONNX Inference

### Using ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Prepare input
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("test.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)
input_numpy = input_tensor.numpy()

# Run inference
outputs = session.run(None, {"image": input_numpy})
logits = outputs[0]

# Get probabilities
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
import torch
import torch.onnx

# Export detection model
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    "detector.onnx",
    input_names=["image"],
    output_names=["boxes", "scores", "labels"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "boxes": {0: "batch_size", 1: "num_detections"},
        "scores": {0: "batch_size", 1: "num_detections"},
        "labels": {0: "batch_size", 1: "num_detections"},
    },
    opset_version=11,
)
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
with torch.no_grad():
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
import torch.onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# 1. Load and prepare model
model = ImageClassifier.load_from_checkpoint(...)
model.eval()

# 2. Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14,
)

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
    
    outputs = session.run(None, {"image": input_tensor})
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

### ONNX Export Fails

**Problem:** Export fails with unsupported operation

**Solutions:**
```python
# 1. Use a lower opset version
torch.onnx.export(..., opset_version=11)

# 2. Simplify the model (remove custom ops)

# 3. Use TorchScript instead
traced_model = torch.jit.trace(model, example_input)
```

### Model Size Too Large

**Problem:** Exported model is too big

**Solutions:**
```python
# 1. Use quantization
quantize_dynamic("model.onnx", "model_quant.onnx")

# 2. Use FP16
# During ONNX export, convert to FP16

# 3. Use a smaller backbone
model = ImageClassifier(backbone="resnet34")  # Instead of resnet50
```

### Inference Speed Not Improved

**Problem:** Exported model isn't faster

**Solutions:**
```python
# 1. Use appropriate execution provider
providers = ['CUDAExecutionProvider']

# 2. Enable optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 3. Use quantization
# 4. Process in batches
```

---

## See Also

- [Classification Inference](classification-inference.md) - Inference with PyTorch models
- [Object Detection Inference](object-detection-inference.md) - Detection model inference
- [Training Guide](../training/training.md) - How to train models
