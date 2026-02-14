# ONNX Export

Export trained AutoTimm models to ONNX format for cross-platform deployment with ONNX Runtime, TensorRT, OpenVINO, CoreML, and more.

## Overview

ONNX (Open Neural Network Exchange) allows you to:

- **Deploy cross-platform** - Run models on any platform with an ONNX runtime
- **Use multiple runtimes** - ONNX Runtime, TensorRT, OpenVINO, CoreML
- **Dynamic batch sizes** - Batch dimension is dynamic by default
- **Hardware acceleration** - GPU, CPU, and specialized accelerators
- **Self-contained files** - Single `.onnx` file for deployment

## Installation

```bash
pip install onnx onnxruntime onnxscript

# For GPU inference
pip install onnxruntime-gpu

# Or install as part of AutoTimm
pip install autotimm[onnx]
```

## Quick Start

### Basic Export

```python
from autotimm import ImageClassifier, export_to_onnx
import torch

# Load trained model
model = ImageClassifier.load_from_checkpoint("model.ckpt")

# Export to ONNX
example_input = torch.randn(1, 3, 224, 224)
export_to_onnx(
    model,
    "model.onnx",
    example_input=example_input
)
```

### Convenience Method

```python
# Even simpler - one line export
model = ImageClassifier(backbone="resnet50", num_classes=10)
model.to_onnx("model.onnx")
```

### Load and Use

```python
import numpy as np
import onnxruntime as ort

# No AutoTimm dependency needed!
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

image = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {input_name: image})
```

## Export Methods

AutoTimm provides multiple ways to export models to ONNX:

### Method 1: export_to_onnx()

Full control over the export process:

```python
from autotimm.export import export_to_onnx

path = export_to_onnx(
    model=model,
    save_path="model.onnx",
    example_input=torch.randn(1, 3, 224, 224),
    opset_version=17,     # ONNX opset version
    simplify=False,       # Optional graph simplification
)
```

**Parameters:**

- `model`: The PyTorch model to export
- `save_path`: Output file path (.onnx extension recommended)
- `example_input`: Example input tensor (required)
- `opset_version`: ONNX opset version (default: 17)
- `dynamic_axes`: Dynamic axes config (default: batch dimension dynamic)
- `simplify`: Whether to simplify the graph with onnx-simplifier

### Method 2: model.to_onnx()

Convenience method on model instances:

```python
# With file save
path = model.to_onnx("model.onnx")

# Without specifying path (uses temp file)
path = model.to_onnx()

# With custom options
path = model.to_onnx(
    "model.onnx",
    example_input=torch.randn(1, 3, 299, 299),
    opset_version=17
)
```

### Method 3: export_checkpoint_to_onnx()

Direct checkpoint export:

```python
from autotimm import export_checkpoint_to_onnx, ImageClassifier

path = export_checkpoint_to_onnx(
    checkpoint_path="model.ckpt",
    save_path="model.onnx",
    model_class=ImageClassifier,
    example_input=torch.randn(1, 3, 224, 224),
)
```

## Dynamic Batch Size

By default, the batch dimension is dynamic, allowing inference with any batch size:

```python
# Export with batch size 1
export_to_onnx(model, "model.onnx", torch.randn(1, 3, 224, 224))

# Inference with any batch size
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Works with batch size 1, 4, 8, etc.
for batch_size in [1, 4, 8]:
    input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {input_name: input_data})
    print(f"Batch {batch_size}: output shape {outputs[0].shape}")
```

### Custom Dynamic Axes

```python
# Dynamic batch and spatial dimensions
dynamic_axes = {
    "input": {0: "batch_size", 2: "height", 3: "width"},
    "output": {0: "batch_size"},
}

export_to_onnx(
    model, "model.onnx", example_input,
    dynamic_axes=dynamic_axes
)
```

## Validation

Verify the exported model matches the original:

```python
from autotimm.export import validate_onnx_export

# Export model
export_to_onnx(model, "model.onnx", example_input)

# Validate outputs match
is_valid = validate_onnx_export(
    original_model=model,
    onnx_path="model.onnx",
    example_input=example_input,
    rtol=1e-5,
    atol=1e-5
)

if is_valid:
    print(":material-check: Export verified successfully")
else:
    print(":material-close: Export validation failed")
```

## Supported Models

All AutoTimm task models support ONNX export:

### Classification

```python
from autotimm import ImageClassifier

model = ImageClassifier(backbone="resnet50", num_classes=1000)
model.to_onnx("classifier.onnx")
```

### Semantic Segmentation

```python
from autotimm import SemanticSegmentor

model = SemanticSegmentor(backbone="resnet50", num_classes=19)
model.to_onnx("segmentor.onnx")
```

### Object Detection

Detection models automatically flatten their list outputs into named tensors for ONNX compatibility:

```python
from autotimm import ObjectDetector

model = ObjectDetector(backbone="resnet50", num_classes=80)
model.to_onnx("detector.onnx", example_input=torch.randn(1, 3, 640, 640))

# Outputs: cls_l0..cls_l4, reg_l0..reg_l4, ctr_l0..ctr_l4 (15 tensors)
```

### Instance Segmentation

```python
from autotimm import InstanceSegmentor

model = InstanceSegmentor(backbone="resnet50", num_classes=80)
model.to_onnx("instance.onnx", example_input=torch.randn(1, 3, 800, 800))

# Outputs: cls_l0..cls_l4, reg_l0..reg_l4, ctr_l0..ctr_l4 (15 tensors)
```

### YOLOX

```python
from autotimm import YOLOXDetector

model = YOLOXDetector(model_name="yolox-s", num_classes=80)
model.to_onnx("yolox.onnx")  # Default input: 640x640

# Outputs: cls_l0..cls_l2, reg_l0..reg_l2 (6 tensors)
```

## Production Deployment

### ONNX Runtime (Python)

```python
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Run inference
image = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {input_name: image})

# Get predictions
logits = outputs[0]
predicted_class = np.argmax(logits, axis=1)[0]
```

### GPU Inference

```python
# Use CUDA provider for GPU acceleration
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Use TensorRT for maximum NVIDIA GPU performance
session = ort.InferenceSession(
    "model.onnx",
    providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

### Session Optimization

```python
# Enable graph optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=["CPUExecutionProvider"]
)
```

### Using load_onnx()

AutoTimm provides a convenience function that validates and loads in one step:

```python
from autotimm import load_onnx

# Validates model integrity, then creates inference session
session = load_onnx("model.onnx")
session = load_onnx("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

## ONNX vs TorchScript

| Feature | ONNX | TorchScript |
|---------|------|-------------|
| **Cross-platform** | :material-check-circle: Wide compatibility | :material-close-circle: PyTorch ecosystem only |
| **Runtime options** | :material-check-circle: ONNX Runtime, TensorRT, OpenVINO, CoreML | :material-close-circle: LibTorch only |
| **Dynamic batch** | :material-check-circle: Built-in support | :material-check-circle: Works but less flexible |
| **C++ deployment** | :material-check-circle: Via ONNX Runtime C++ API | :material-check-circle: Via LibTorch |
| **Mobile** | :material-check-circle: ONNX Runtime Mobile | :material-check-circle: PyTorch Mobile |
| **Python-free** | :material-check-circle: No Python needed | :material-check-circle: No Python needed |
| **GPU optimization** | :material-check-circle: TensorRT, OpenVINO | :material-close-circle: Limited |

**Recommendation:** Use ONNX for cross-platform deployment and hardware-optimized inference. Use TorchScript when staying within the PyTorch ecosystem.

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'onnx'**

```bash
pip install onnx onnxruntime onnxscript
```

**2. RuntimeError: ONNX export failed**

- Try a lower opset version: `opset_version=14`
- Ensure model is in eval mode: `model.eval()`
- Disable `torch.compile`: `compile_model=False`

**3. Outputs don't match original model**

```python
# Validate with looser tolerance
is_valid = validate_onnx_export(model, "model.onnx", example_input, rtol=1e-3, atol=1e-3)
```

**4. Dynamic axes warning**

The warning about `dynamic_axes` being deprecated in favor of `dynamic_shapes` is harmless and can be ignored. AutoTimm handles this internally.

## Limitations

### What Works

:material-check-circle: Standard feedforward models
:material-check-circle: CNN backbones (ResNet, EfficientNet, etc.)
:material-check-circle: Vision Transformers (ViT, Swin, DeiT)
:material-check-circle: Detection models (FCOS, YOLOX)
:material-check-circle: Segmentation models (DeepLabV3+, FCN)
:material-check-circle: Dynamic batch sizes
:material-check-circle: Different input sizes

### What Doesn't Work

:material-close-circle: Nested list/dict outputs (detection outputs are auto-flattened)
:material-close-circle: Training-specific features (optimizers, schedulers)
:material-close-circle: Some custom Python operations without ONNX equivalents
:material-close-circle: Mask head of InstanceSegmentor (detection head only)

## Examples

See complete working examples in the repository:

- `examples/deployment/export_to_onnx.py` - Comprehensive ONNX export examples

## See Also

- [TorchScript Export](torchscript-export.md) - TorchScript export guide
- [Production Deployment](deployment.md) - Complete deployment guide
- [Model Export Guide](../inference/model-export.md) - Overview of all export options
- [API Reference](../../api/export.md) - Complete API documentation
