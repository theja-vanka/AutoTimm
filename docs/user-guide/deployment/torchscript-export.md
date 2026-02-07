# TorchScript Export

Export trained AutoTimm models to TorchScript format for production deployment without Python dependencies.

## Overview

TorchScript allows you to:

- **Deploy without Python** - Run models in C++, mobile, or embedded devices
- **Optimize for inference** - Automatic optimizations for faster inference
- **Single-file deployment** - Self-contained `.pt` files
- **Cross-platform** - Deploy to any platform with LibTorch

## Quick Start

### Basic Export

```python
from autotimm import ImageClassifier, export_to_torchscript
import torch

# Load trained model
model = ImageClassifier.load_from_checkpoint("model.ckpt")

# Export to TorchScript
example_input = torch.randn(1, 3, 224, 224)
export_to_torchscript(
    model,
    "model.pt",
    example_input=example_input
)
```

### Convenience Method

```python
# Even simpler - one line export
model = ImageClassifier(backbone="resnet50", num_classes=10)
model.to_torchscript("model.pt")
```

### Load and Use

```python
import torch

# No AutoTimm dependency needed!
model = torch.jit.load("model.pt")
model.eval()

with torch.no_grad():
    output = model(image)
```

## Export Methods

AutoTimm provides multiple ways to export models:

### Method 1: export_to_torchscript()

Full control over export process:

```python
from autotimm.export import export_to_torchscript

scripted_model = export_to_torchscript(
    model=model,
    save_path="model.pt",
    example_input=torch.randn(1, 3, 224, 224),
    method="trace",  # "trace" or "script"
    optimize=True,   # Apply inference optimizations
)
```

**Parameters:**

- `model`: The PyTorch model to export
- `save_path`: Output file path (.pt extension recommended)
- `example_input`: Example input tensor (required for method="trace")
- `method`: Export method - "trace" (recommended) or "script"
- `optimize`: Whether to apply inference optimizations (default: True)

### Method 2: model.to_torchscript()

Convenience method on model instances:

```python
# With file save
scripted = model.to_torchscript("model.pt")

# Without file save (in-memory)
scripted = model.to_torchscript()

# With custom options
scripted = model.to_torchscript(
    "model.pt",
    example_input=torch.randn(1, 3, 299, 299),
    method="trace"
)
```

### Method 3: export_checkpoint_to_torchscript()

Direct checkpoint export:

```python
from autotimm import export_checkpoint_to_torchscript, ImageClassifier

scripted = export_checkpoint_to_torchscript(
    checkpoint_path="model.ckpt",
    save_path="model.pt",
    model_class=ImageClassifier,
    example_input=torch.randn(1, 3, 224, 224),
)
```

## Trace vs Script

### Trace (Recommended)

**How it works:** Records operations by running example input through the model

**Pros:**
- More reliable for complex models
- Better compatibility
- Captures actual execution path

**Cons:**
- Requires example input
- May not capture dynamic control flow

**Use when:**
- Standard feedforward models
- Models without dynamic behavior
- Production deployment (recommended)

```python
export_to_torchscript(
    model,
    "model.pt",
    example_input=torch.randn(1, 3, 224, 224),
    method="trace"  # Recommended
)
```

### Script (Advanced)

**How it works:** Analyzes Python source code

**Pros:**
- Doesn't require example input
- Captures control flow

**Cons:**
- Less compatible with modern Python features
- May fail on complex type annotations
- Not recommended for AutoTimm models

**Use when:**
- Simple models only
- Dynamic control flow is critical

```python
export_to_torchscript(
    model,
    "model.pt",
    method="script"  # Not recommended
)
```

## Input Shapes

### Fixed Input Size

Export optimized for specific input dimensions:

```python
# Export for 224x224 images
example_input = torch.randn(1, 3, 224, 224)
export_to_torchscript(model, "model_224.pt", example_input)
```

### Different Sizes

Export separate models for different resolutions:

```python
sizes = [(224, 224), (299, 299), (384, 384)]

for h, w in sizes:
    example_input = torch.randn(1, 3, h, w)
    export_to_torchscript(
        model,
        f"model_{h}x{w}.pt",
        example_input
    )
```

### Batch Inference

Export with batch size support:

```python
# Export with batch size 8
example_input = torch.randn(8, 3, 224, 224)
export_to_torchscript(model, "batch_model.pt", example_input)

# Can still use with different batch sizes at inference
model = torch.jit.load("batch_model.pt")
output = model(torch.randn(16, 3, 224, 224))  # Works!
```

## Validation

Verify exported model matches original:

```python
from autotimm.export import validate_torchscript_export

# Export model
scripted = export_to_torchscript(model, "model.pt", example_input)

# Validate outputs match
is_valid = validate_torchscript_export(
    original_model=model,
    scripted_model=scripted,
    example_input=example_input,
    rtol=1e-5,
    atol=1e-8
)

if is_valid:
    print("✓ Export verified successfully")
else:
    print("✗ Export validation failed")
```

## Supported Models

All AutoTimm task models support TorchScript export:

### Classification

```python
from autotimm import ImageClassifier

model = ImageClassifier(backbone="resnet50", num_classes=1000)
model.to_torchscript("classifier.pt")
```

### Object Detection

```python
from autotimm import ObjectDetector

model = ObjectDetector(backbone="resnet50", num_classes=80)
example_input = torch.randn(1, 3, 640, 640)
model.to_torchscript("detector.pt", example_input=example_input)
```

### Semantic Segmentation

```python
from autotimm import SemanticSegmentor

model = SemanticSegmentor(backbone="resnet50", num_classes=19)
model.to_torchscript("segmentor.pt")
```

### Instance Segmentation

```python
from autotimm import InstanceSegmentor

model = InstanceSegmentor(backbone="resnet50", num_classes=80)
example_input = torch.randn(1, 3, 800, 800)
model.to_torchscript("instance.pt", example_input=example_input)
```

### YOLOX

```python
from autotimm import YOLOXDetector

model = YOLOXDetector(model_name="yolox-s", num_classes=80)
example_input = torch.randn(1, 3, 640, 640)
model.to_torchscript("yolox.pt", example_input=example_input)
```

## Production Deployment

### Loading in Python

```python
import torch

# Load model
model = torch.jit.load("model.pt")
model.eval()

# Set device
model = model.to("cuda")  # or "cpu"

# Run inference
with torch.no_grad():
    output = model(image)
```

### C++ Deployment

See [C++ Deployment Guide](cpp-deployment.md) for complete examples.

```cpp
#include <torch/script.h>

// Load model
torch::jit::script::Module module = torch::jit::load("model.pt");

// Run inference
std::vector<torch::jit::IValue> inputs;
inputs.push_back(input_tensor);
auto output = module.forward(inputs).toTensor();
```

### Mobile Deployment

See [Mobile Deployment Guide](mobile-deployment.md) for iOS and Android examples.

## Optimization Tips

### 1. Disable torch.compile

For TorchScript export, disable torch.compile to avoid double optimization:

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    compile_model=False  # Disable for TorchScript export
)
```

### 2. Use eval() Mode

Always set model to evaluation mode:

```python
model.eval()  # Important!
model.to_torchscript("model.pt")
```

### 3. Choose Appropriate Batch Size

Export with expected inference batch size:

```python
# Single image inference
example_input = torch.randn(1, 3, 224, 224)

# Batch inference
example_input = torch.randn(8, 3, 224, 224)
```

### 4. Optimize After Export

TorchScript optimizations are applied by default:

```python
export_to_torchscript(
    model,
    "model.pt",
    example_input=example_input,
    optimize=True  # Default
)
```

### 5. Test Thoroughly

Always validate exported model:

```python
# Compare outputs
original_out = model(test_input)
scripted_out = torch.jit.load("model.pt")(test_input)

assert torch.allclose(original_out, scripted_out, rtol=1e-5)
```

## Troubleshooting

For TorchScript export issues, see the [Troubleshooting - Export & Inference](../../troubleshooting/deployment/export-inference.md) including:

- RuntimeError: Couldn't export module
- Model outputs don't match
- Dynamic control flow not captured
- Module has __getattr__ method

## Limitations

### What Works

✅ Standard feedforward models
✅ CNN backbones (ResNet, EfficientNet, etc.)
✅ Vision Transformers (ViT, Swin, DeiT)
✅ Detection models (FCOS, YOLOX)
✅ Segmentation models (DeepLabV3+, FCN)
✅ Batch inference
✅ Different input sizes

### What Doesn't Work

❌ Python 3.10+ union types with `method="script"`
❌ Complex dynamic control flow
❌ Some custom Python operations
❌ Training-specific features (optimizers, schedulers)

**Recommendation:** Use `method="trace"` for maximum compatibility.

## Examples

See complete working examples in the repository:

- `examples/deployment/export_to_torchscript.py` - Comprehensive export examples
- `examples/deployment/deploy_torchscript_cpp.py` - C++ and mobile deployment code

## See Also

- [Model Export Guide](../inference/model-export.md) - Overview of all export options
- [C++ Deployment](cpp-deployment.md) - Deploy to C++ applications
- [Mobile Deployment](mobile-deployment.md) - Deploy to iOS/Android
- [API Reference](../../api/export.md) - Complete API documentation
