# Model Deployment & Export

Export and optimize HuggingFace Hub models for production deployment across platforms.

## Overview

Comprehensive guide to deploying models from development to production, covering ONNX export, quantization, TorchScript, optimization techniques, and serving infrastructure.

## What This Example Covers

- **ONNX export** - Cross-platform model format
- **Dynamic quantization** - 2-4x smaller, 1.5-2x faster
- **TorchScript** - PyTorch production format
- **Inference optimization** - torch.compile, batching
- **FastAPI serving** - REST API example
- **Deployment checklist** - Production best practices

## Export Formats

### ONNX (Recommended for Production)

```python
import torch
from autotimm import ImageClassifier

model = ImageClassifier(
    backbone="hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
    num_classes=10,
)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=14,
)
```

**Benefits**:

- Cross-platform (C++, C#, Java, JavaScript)
- Hardware acceleration (TensorRT, OpenVINO, CoreML)
- Framework-independent
- Mature ecosystem

### TorchScript (PyTorch Production)

```python
# Method 1: Tracing (recommended)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# Method 2: Scripting (more general)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

**Benefits**:

- No Python dependency
- C++ deployment
- PyTorch Mobile
- Optimized execution

## Optimization Techniques

### Dynamic Quantization

```python
import torch.quantization as quantization

# Apply INT8 quantization
quantized_model = quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8,
)

# Results: 2-4x smaller, 1.5-2x faster on CPU
```

### Inference Optimization

```python
# 1. Disable gradients (2x faster)
model.eval()
with torch.no_grad():
    output = model(image)

# 2. torch.compile (1.5-2x faster, PyTorch 2.0+)
compiled_model = torch.compile(model, mode="reduce-overhead")

# 3. Batch processing (higher throughput)
batch_input = torch.randn(16, 3, 224, 224)
outputs = model(batch_input)
```

## FastAPI Serving

```python
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()

# Load model once at startup
model = ImageClassifier(
    backbone="hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
    num_classes=1000,
)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

    results = [
        {"class": int(idx), "probability": float(prob)}
        for prob, idx in zip(top5_prob[0], top5_idx[0])
    ]

    return {"predictions": results}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

## Run the Example

```bash
python examples/huggingface/hf_deployment.py
```

## Deployment Patterns

### Cloud/Server (GPU)
- **Format**: ONNX + TensorRT or PyTorch
- **Optimization**: FP16, batch processing
- **Serving**: FastAPI + Gunicorn + NGINX
- **Performance**: 50-200 images/sec per GPU

### Cloud/Server (CPU)
- **Format**: ONNX or TorchScript
- **Optimization**: INT8 quantization, batching
- **Serving**: FastAPI + Gunicorn
- **Performance**: 5-20 images/sec per core

### Edge Device (Raspberry Pi)
- **Format**: TFLite or ONNX
- **Optimization**: INT8, small models
- **Models**: MobileNet, EfficientNet-Lite
- **Performance**: 1-5 images/sec

### Mobile (iOS/Android)
- **Format**: CoreML (iOS) or TFLite (Android)
- **Optimization**: INT8/FP16
- **Models**: MobileNetV3, EfficientNet-Lite
- **Performance**: 10-30 images/sec

## Production Checklist

### Model Optimization
- [ ] Export to ONNX for cross-platform
- [ ] Apply quantization for CPU (2-4x speedup)
- [ ] Use TorchScript for PyTorch serving
- [ ] Profile and identify bottlenecks

### Inference Optimization
- [ ] Always use `torch.no_grad()` or `model.eval()`
- [ ] Use `torch.compile()` if PyTorch 2.0+
- [ ] Implement request batching
- [ ] Test on target hardware

### Serving Infrastructure
- [ ] Use FastAPI/Flask for REST API
- [ ] Implement request batching
- [ ] Add caching for common requests
- [ ] Set up load balancing

### Monitoring & Testing
- [ ] Monitor latency (p50, p95, p99)
- [ ] Track accuracy on production data
- [ ] Implement A/B testing
- [ ] Set up alerts for degradation

## Benchmarks

### Model Size Comparison
```
Original (FP32):     45 MB
Quantized (INT8):    12 MB  (3.75x smaller)
Pruned + Quantized:   8 MB  (5.6x smaller)
```

### Inference Speed
```
FP32 (CPU):          50 ms/image
INT8 (CPU):          25 ms/image  (2x faster)
FP16 (GPU):           5 ms/image  (10x faster)
TensorRT (GPU):       2 ms/image  (25x faster)
```

## Best Practices

1. **Choose format by platform**:
   - Multi-platform → ONNX
   - PyTorch only → TorchScript
   - Mobile → TFLite/CoreML

2. **Always quantize for CPU**:
   - 2-4x smaller models
   - 1.5-2x faster inference
   - Minimal accuracy loss (<1%)

3. **Use batching for throughput**:
   - Batch size 8-16 for GPU
   - Batch size 1-4 for CPU
   - Trade latency for throughput

4. **Profile before optimizing**:
   - Identify bottlenecks
   - Don't optimize prematurely
   - Test on production hardware

5. **Monitor in production**:
   - Track latency percentiles
   - Monitor accuracy drift
   - A/B test model updates

## Related Examples

- [HuggingFace Hub Models](huggingface-hub.md)
- [Model Ensemble](hf_ensemble.md)
- [Hyperparameter Tuning](../utilities/hf_hyperparameter_tuning.md)

## See Also

- [Deployment Guide](../../user-guide/deployment.md)
- [ONNX Documentation](https://onnx.ai/)
- [TorchScript Guide](https://pytorch.org/docs/stable/jit.html)
