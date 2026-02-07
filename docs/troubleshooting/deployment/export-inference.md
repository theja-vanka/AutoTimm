# Export and Inference Issues

Problems with model export and inference.

## ONNX Export Failures

```python
import torch

# 1. Export with dynamic axes for variable input sizes
model = ImageClassifier.load_from_checkpoint("checkpoint.ckpt")
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["output"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=14,  # Use higher opset for better compatibility
)

# 2. If export fails, simplify model
model.to_torchscript(
    file_path="model.pt",
    method="trace",  # Try "script" if trace fails
)
```

## TorchScript Issues

```python
# Some operations don't support TorchScript
# Try tracing instead of scripting
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# Or freeze the model
frozen_model = torch.jit.freeze(traced_model)
frozen_model.save("model_frozen.pt")
```

## Inference Optimization

```python
# 1. Compile model for faster inference (PyTorch 2.0+)
import torch
model = torch.compile(model, mode="reduce-overhead")

# 2. Use half precision for inference
model = model.half()
input_tensor = input_tensor.half()

# 3. Disable gradient computation
with torch.inference_mode():
    with torch.cuda.amp.autocast():  # Automatic mixed precision
        outputs = model(inputs)
```

## Classification Inference Issues

### Out of Memory During Inference

**Solutions:**

```python
# 1. Reduce batch size
batch_size = 16

# 2. Use smaller image size
transform = transforms.Resize(224)

# 3. Clear cache between batches
torch.cuda.empty_cache()

# 4. Use CPU for very large images
device = torch.device("cpu")
```

### Slow Inference

**Solutions:**

```python
# 1. Use GPU
device = torch.device("cuda")

# 2. Increase batch size
batch_size = 64

# 3. Use half precision
model = model.half()

# 4. Keep model in memory (don't reload)
```

## Detection Inference Issues

### No Detections

**Solutions:**

```python
# 1. Lower score threshold
from autotimm.inference import DetectionPipeline
pipeline = DetectionPipeline(score_threshold=0.1)

# 2. Check image is RGB
image = Image.open("img.jpg").convert("RGB")

# 3. Verify model classes
print(f"Model num_classes: {model.num_classes}")
```

### Too Many Duplicate Detections

**Solutions:**

```python
# 1. Lower NMS threshold (stricter)
model = ObjectDetector(nms_thresh=0.3)

# 2. Increase score threshold
pipeline = DetectionPipeline(score_threshold=0.5)
```

### Missing Small Objects

**Solutions:**

```python
# 1. Use larger image size
pipeline = DetectionPipeline(image_size=800)

# 2. Use multi-scale inference
```

## Segmentation Inference Issues

### Mask Size Mismatch

**Explanation:** AutoTimm automatically resizes masks back to original image size using NEAREST interpolation.

**If still having issues:**

```python
# Manually resize
import cv2
mask_resized = cv2.resize(
    mask,
    (original_width, original_height),
    interpolation=cv2.INTER_NEAREST
)
```

## Related Issues

- [Production Deployment](production.md) - Production deployment issues
- [OOM Errors](../performance/oom-errors.md) - Memory problems
- [Slow Training](../performance/slow-training.md) - Performance optimization
