# Export API Reference

Complete API reference for AutoTimm's export module.

## Module: `autotimm.export`

The export module provides functions for converting trained AutoTimm models to various deployment formats.

## Functions

### export_to_torchscript

```python
def export_to_torchscript(
    model: nn.Module,
    save_path: str | Path,
    example_input: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    method: str = "trace",
    strict: bool = True,
    optimize: bool = True,
    wrap_model: bool = True,
) -> torch.jit.ScriptModule
```

Export a PyTorch model to TorchScript format for deployment.

**Parameters:**

- **model** (`nn.Module`) - The PyTorch model to export. Can be any AutoTimm task model (ImageClassifier, ObjectDetector, etc.) or custom PyTorch model.

- **save_path** (`str | Path`) - Path where the exported model will be saved. Use `.pt` extension by convention.

- **example_input** (`torch.Tensor | tuple[torch.Tensor, ...] | None`, optional) - Example input tensor(s) for tracing. Required when `method="trace"`. Should match the expected input shape and type. Default: `None`.

- **method** (`str`, optional) - Export method to use:
  - `"trace"` (recommended) - Records operations by running example input through model
  - `"script"` - Analyzes Python source code (not compatible with modern type annotations)

  Default: `"trace"`

- **strict** (`bool`, optional) - Whether to enforce strict typing during scripting. Only applies when `method="script"`. Default: `True`.

- **optimize** (`bool`, optional) - Whether to apply inference optimizations to the exported model. Recommended for production deployment. Default: `True`.

- **wrap_model** (`bool`, optional) - Internal parameter for Lightning module compatibility. Default: `True`.

**Returns:**

- `torch.jit.ScriptModule` - The exported TorchScript module.

**Raises:**

- `ValueError` - If `method` is invalid or if `example_input` is not provided when `method="trace"`.
- `RuntimeError` - If export fails due to model compatibility issues.

**Example:**

```python
from autotimm import ImageClassifier, export_to_torchscript
import torch

# Load model
model = ImageClassifier.load_from_checkpoint("model.ckpt")
model.eval()

# Export with trace (recommended)
example_input = torch.randn(1, 3, 224, 224)
scripted = export_to_torchscript(
    model=model,
    save_path="model.pt",
    example_input=example_input,
    method="trace",
    optimize=True
)

# Use exported model
with torch.inference_mode():
    output = scripted(example_input)
```

**Notes:**

- Always set model to evaluation mode (`model.eval()`) before exporting
- The `trace` method is more reliable than `script` for most models
- Exported models can be loaded with `torch.jit.load()` without AutoTimm dependency
- PyTorch Lightning modules are automatically handled

---

### load_torchscript

```python
def load_torchscript(
    path: str | Path,
    device: str | torch.device = "cpu"
) -> torch.jit.ScriptModule
```

Load a TorchScript model from file.

**Parameters:**

- **path** (`str | Path`) - Path to the TorchScript `.pt` file.

- **device** (`str | torch.device`, optional) - Device to load the model on. Can be `"cpu"`, `"cuda"`, or a torch.device object. Default: `"cpu"`.

**Returns:**

- `torch.jit.ScriptModule` - The loaded TorchScript module.

**Raises:**

- `FileNotFoundError` - If the specified file doesn't exist.
- `RuntimeError` - If the file is not a valid TorchScript module.

**Example:**

```python
from autotimm import load_torchscript
import torch

# Load on CPU
model = load_torchscript("model.pt", device="cpu")

# Load on GPU
model = load_torchscript("model.pt", device="cuda")

# Run inference
model.eval()
with torch.inference_mode():
    output = model(input_tensor)
```

**Notes:**

- No AutoTimm dependency required for loading
- Alternatively, use `torch.jit.load()` directly
- Always set to eval mode before inference

---

### export_checkpoint_to_torchscript

```python
def export_checkpoint_to_torchscript(
    checkpoint_path: str | Path,
    save_path: str | Path,
    model_class: type,
    example_input: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    method: str = "trace",
    strict: bool = True,
    optimize: bool = True,
) -> torch.jit.ScriptModule
```

Export a Lightning checkpoint directly to TorchScript in one step.

**Parameters:**

- **checkpoint_path** (`str | Path`) - Path to the PyTorch Lightning checkpoint file (`.ckpt`).

- **save_path** (`str | Path`) - Path where the TorchScript model will be saved (`.pt`).

- **model_class** (`type`) - The AutoTimm model class (e.g., `ImageClassifier`, `ObjectDetector`).

- **example_input** (`torch.Tensor | tuple[torch.Tensor, ...] | None`, optional) - Example input tensor for tracing. Required when `method="trace"`. Default: `None`.

- **method** (`str`, optional) - Export method: `"trace"` or `"script"`. Default: `"trace"`.

- **strict** (`bool`, optional) - Enforce strict typing during scripting. Default: `True`.

- **optimize** (`bool`, optional) - Apply inference optimizations. Default: `True`.

**Returns:**

- `torch.jit.ScriptModule` - The exported TorchScript module.

**Raises:**

- `FileNotFoundError` - If checkpoint file doesn't exist.
- `ValueError` - If parameters are invalid.
- `RuntimeError` - If export fails.

**Example:**

```python
from autotimm import ImageClassifier, export_checkpoint_to_torchscript
import torch

# Direct checkpoint to TorchScript conversion
scripted = export_checkpoint_to_torchscript(
    checkpoint_path="best_model.ckpt",
    save_path="deployment_model.pt",
    model_class=ImageClassifier,
    example_input=torch.randn(1, 3, 224, 224),
    method="trace",
    optimize=True
)

print("Model exported successfully!")
```

**Notes:**

- Convenience function that combines loading and exporting
- Useful for deployment pipelines and CI/CD
- Model is automatically set to evaluation mode

---

### validate_torchscript_export

```python
def validate_torchscript_export(
    original_model: nn.Module,
    scripted_model: torch.jit.ScriptModule,
    example_input: torch.Tensor | tuple[torch.Tensor, ...],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool
```

Validate that a TorchScript export produces the same outputs as the original model.

**Parameters:**

- **original_model** (`nn.Module`) - The original PyTorch model.

- **scripted_model** (`torch.jit.ScriptModule`) - The exported TorchScript model.

- **example_input** (`torch.Tensor | tuple[torch.Tensor, ...]`) - Input tensor(s) to test with.

- **rtol** (`float`, optional) - Relative tolerance for numerical comparison. Default: `1e-5`.

- **atol** (`float`, optional) - Absolute tolerance for numerical comparison. Default: `1e-8`.

**Returns:**

- `bool` - `True` if outputs match within tolerance, `False` otherwise.

**Example:**

```python
from autotimm import ImageClassifier, export_to_torchscript
from autotimm.export import validate_torchscript_export
import torch

# Create and export model
model = ImageClassifier(backbone="resnet18", num_classes=10)
model.eval()

example_input = torch.randn(1, 3, 224, 224)
scripted = export_to_torchscript(model, "model.pt", example_input)

# Validate
is_valid = validate_torchscript_export(
    original_model=model,
    scripted_model=scripted,
    example_input=example_input,
    rtol=1e-5,
    atol=1e-8
)

if is_valid:
    print(":material-check: Export validated successfully")
else:
    print(":material-close: Validation failed - outputs don't match")
```

**Notes:**

- Both models are automatically set to evaluation mode
- Compares outputs using `torch.allclose()`
- Use tighter tolerances (`rtol=1e-6`, `atol=1e-9`) for critical applications
- Always validate exports before production deployment

---

## Model Methods

AutoTimm task models include a convenience method for TorchScript export.

### model.to_torchscript

```python
def to_torchscript(
    self,
    save_path: str | None = None,
    example_input: torch.Tensor | None = None,
    method: str = "trace",
    **kwargs: Any,
) -> torch.jit.ScriptModule
```

Export the model to TorchScript format (convenience method).

**Parameters:**

- **save_path** (`str | None`, optional) - Path to save the exported model. If `None`, model is not saved to disk (in-memory only). Default: `None`.

- **example_input** (`torch.Tensor | None`, optional) - Example input tensor for tracing. If `None`, uses default input shape (1, 3, 224, 224). Default: `None`.

- **method** (`str`, optional) - Export method: `"trace"` or `"script"`. Default: `"trace"`.

- ****kwargs** - Additional arguments passed to `export_to_torchscript()`.

**Returns:**

- `torch.jit.ScriptModule` - The exported TorchScript module.

**Example:**

```python
from autotimm import ImageClassifier
import torch

model = ImageClassifier(
    backbone="resnet50",
    num_classes=1000,
    compile_model=False  # Disable torch.compile for export
)
model.eval()

# Export with default input
scripted = model.to_torchscript("model.pt")

# Export with custom input
example_input = torch.randn(1, 3, 299, 299)
scripted = model.to_torchscript("model_299.pt", example_input=example_input)

# In-memory export (no file save)
scripted = model.to_torchscript()
```

**Available For:**

- `ImageClassifier`
- `ObjectDetector` (future)
- `SemanticSegmentor` (future)
- `InstanceSegmentor` (future)
- `YOLOXDetector` (future)

---

## Type Definitions

### Supported Models

All AutoTimm task models support TorchScript export:

- **ImageClassifier** - Image classification models
- **ObjectDetector** - Object detection models (FCOS, YOLOX)
- **SemanticSegmentor** - Semantic segmentation models
- **InstanceSegmentor** - Instance segmentation models (Mask R-CNN)
- **YOLOXDetector** - YOLOX object detection models

### Export Methods

- **trace** (recommended) - Records operations by running example input
  - :material-check-circle: More reliable
  - :material-check-circle: Better compatibility
  - :material-check-circle: Captures actual execution
  - :material-close-circle: Requires example input
  - :material-close-circle: May not capture dynamic control flow

- **script** - Analyzes Python source code
  - :material-check-circle: No example input required
  - :material-check-circle: Captures control flow
  - :material-close-circle: Not compatible with modern Python type annotations
  - :material-close-circle: Less reliable for complex models
  - :material-close-circle: Not recommended for AutoTimm models

---

## Best Practices

### 1. Always Use Evaluation Mode

```python
model.eval()  # Critical!
scripted = model.to_torchscript("model.pt")
```

### 2. Disable torch.compile for Export

```python
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    compile_model=False  # Important for TorchScript
)
```

### 3. Use Trace Method

```python
# Recommended
export_to_torchscript(model, "model.pt", example_input, method="trace")

# Not recommended
export_to_torchscript(model, "model.pt", method="script")  # May fail
```

### 4. Validate Exports

```python
is_valid = validate_torchscript_export(model, scripted, example_input)
assert is_valid, "Export validation failed!"
```

### 5. Match Input Shapes

```python
# Training used 224x224
example_input = torch.randn(1, 3, 224, 224)

# Export with same shape
export_to_torchscript(model, "model.pt", example_input)
```

---

## Error Handling

### Common Errors

**1. RuntimeError: Couldn't export Python operator**

```python
# Solution: Use trace instead of script
export_to_torchscript(model, "model.pt", example_input, method="trace")
```

**2. ValueError: example_input required for tracing**

```python
# Solution: Provide example input
example_input = torch.randn(1, 3, 224, 224)
export_to_torchscript(model, "model.pt", example_input)
```

**3. RuntimeError: ImageClassifier is not attached to a Trainer**

```python
# Solution: This is handled automatically by the export function
# If you encounter this, ensure you're using the latest AutoTimm version
```

**4. Validation fails (outputs don't match)**

```python
# Solution: Ensure model is in eval mode
model.eval()

# Check for dropout/batch_norm issues
with torch.inference_mode():
    export_to_torchscript(model, "model.pt", example_input)
```

---

## Loading and Inference

### Python Inference

```python
import torch

# Load model (no AutoTimm required)
model = torch.jit.load("model.pt")
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Run inference
with torch.inference_mode():
    output = model(input_tensor.to(device))
```

### C++ Inference

```cpp
#include <torch/script.h>

// Load model
torch::jit::script::Module module = torch::jit::load("model.pt");
module.eval();

// Run inference
std::vector<torch::jit::IValue> inputs;
inputs.push_back(input_tensor);

torch::NoGradGuard no_grad;
auto output = module.forward(inputs).toTensor();
```

---

## See Also

- [TorchScript Export Guide](../user-guide/deployment/torchscript-export.md) - Complete usage guide
- [C++ Deployment](../user-guide/deployment/cpp-deployment.md) - Deploy to C++ applications
- [Mobile Deployment](../user-guide/deployment/mobile-deployment.md) - Deploy to iOS/Android
- [Model Export Guide](../user-guide/inference/model-export.md) - Overview of all export options

---

## Version History

### v0.7.2
- Added TorchScript export functionality
- Support for all task models
- Automatic Lightning module handling
- Validation utilities
- Convenience methods on model classes
