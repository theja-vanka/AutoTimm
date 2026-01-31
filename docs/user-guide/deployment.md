# Production Deployment

This guide covers deploying AutoTimm models to production environments, including model export, optimization, containerization, and serving.

## Model Export Overview

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| TorchScript | PyTorch ecosystem | Easy, full feature support | Python required |
| ONNX | Cross-platform | Wide compatibility | Some ops unsupported |
| TensorRT | NVIDIA GPUs | Maximum speed | NVIDIA only |

---

## TorchScript Export

TorchScript serializes PyTorch models for deployment without Python dependencies.

### Trace Mode (Recommended)

Trace mode captures the execution graph by running the model with example input.

```python
import torch
from autotimm import ImageClassifier, MetricConfig


def export_torchscript_trace(checkpoint_path, output_path, backbone, num_classes, image_size=224):
    """Export model using TorchScript trace mode."""
    # Load model
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        )
    ]

    model = ImageClassifier.load_from_checkpoint(
        checkpoint_path,
        backbone=backbone,
        num_classes=num_classes,
        metrics=metrics,
    )
    model.eval()

    # Create example input
    example_input = torch.randn(1, 3, image_size, image_size)

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Optimize for inference
    traced_model = torch.jit.optimize_for_inference(traced_model)

    # Save
    traced_model.save(output_path)
    print(f"Saved TorchScript model to {output_path}")

    return traced_model


# Usage
model = export_torchscript_trace(
    checkpoint_path="checkpoints/best.ckpt",
    output_path="model_traced.pt",
    backbone="resnet50",
    num_classes=10,
)
```

### Script Mode

Script mode preserves control flow (if/else, loops). Use when your model has dynamic behavior.

```python
import torch


def export_torchscript_script(model, output_path):
    """Export model using TorchScript script mode."""
    model.eval()

    # Script the model
    scripted_model = torch.jit.script(model)

    # Save
    scripted_model.save(output_path)

    return scripted_model
```

### Loading TorchScript Models

```python
import torch

# Load model (no need for original model definition)
model = torch.jit.load("model_traced.pt")
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

---

## ONNX Export

ONNX provides cross-platform compatibility with runtimes like ONNX Runtime, TensorRT, and OpenVINO.

### Basic ONNX Export

```python
import torch
import torch.onnx
from autotimm import ImageClassifier, MetricConfig


def export_onnx(checkpoint_path, output_path, backbone, num_classes, image_size=224):
    """Export model to ONNX format."""
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        )
    ]

    model = ImageClassifier.load_from_checkpoint(
        checkpoint_path,
        backbone=backbone,
        num_classes=num_classes,
        metrics=metrics,
    )
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"Saved ONNX model to {output_path}")


# Usage
export_onnx(
    checkpoint_path="checkpoints/best.ckpt",
    output_path="model.onnx",
    backbone="resnet50",
    num_classes=10,
)
```

### Verify ONNX Model

```python
import onnx

# Load and check
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# Print model graph
print(onnx.helper.printable_graph(model.graph))
```

### Dynamic Axes Configuration

```python
# For variable batch size
dynamic_axes = {
    "image": {0: "batch_size"},
    "logits": {0: "batch_size"},
}

# For variable image size (use with caution)
dynamic_axes = {
    "image": {0: "batch_size", 2: "height", 3: "width"},
    "logits": {0: "batch_size"},
}
```

---

## Quantization

Reduce model size and improve inference speed with quantization.

### Dynamic Quantization

Quantizes weights to INT8 at runtime. Simple but less optimization.

```python
import torch
from autotimm import ImageClassifier, MetricConfig


def quantize_dynamic(checkpoint_path, backbone, num_classes):
    """Apply dynamic quantization."""
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        )
    ]

    model = ImageClassifier.load_from_checkpoint(
        checkpoint_path,
        backbone=backbone,
        num_classes=num_classes,
        metrics=metrics,
    )
    model.eval()

    # Quantize
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
    )

    return quantized_model


# Usage
quantized = quantize_dynamic(
    checkpoint_path="checkpoints/best.ckpt",
    backbone="resnet50",
    num_classes=10,
)

# Save
torch.save(quantized.state_dict(), "model_quantized.pth")
```

### Static Quantization

Calibrates quantization parameters with representative data. Better accuracy.

```python
import torch
from torch.quantization import get_default_qconfig, prepare, convert


def quantize_static(model, calibration_loader, num_calibration_batches=100):
    """Apply static quantization with calibration."""
    model.eval()

    # Fuse modules (required for some architectures)
    # model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

    # Set quantization config
    model.qconfig = get_default_qconfig("fbgemm")  # For x86 CPUs

    # Prepare model for calibration
    model_prepared = prepare(model)

    # Calibrate with representative data
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            model_prepared(images)

    # Convert to quantized model
    model_quantized = convert(model_prepared)

    return model_quantized
```

### Quantization-Aware Training (QAT)

Train with simulated quantization for best accuracy.

```python
import torch
from torch.quantization import get_default_qat_qconfig, prepare_qat


def prepare_qat_model(model):
    """Prepare model for quantization-aware training."""
    model.train()

    # Set QAT config
    model.qconfig = get_default_qat_qconfig("fbgemm")

    # Prepare for QAT
    model_qat = prepare_qat(model)

    return model_qat


# After training, convert to quantized
def finalize_qat(model_qat):
    """Convert QAT model to quantized model."""
    model_qat.eval()
    model_quantized = torch.quantization.convert(model_qat)
    return model_quantized
```

### ONNX Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx(input_path, output_path):
    """Quantize ONNX model."""
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8,
    )


# Usage
quantize_onnx("model.onnx", "model_quantized.onnx")
```

---

## Docker Containerization

### Dockerfile for CPU Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.onnx .
COPY app.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "app.py"]
```

### Dockerfile for GPU Deployment

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app

# Install PyTorch with CUDA
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model_traced.pt .
COPY app.py .

EXPOSE 8000

CMD ["python3", "app.py"]
```

### Requirements File

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
onnxruntime>=1.15.0  # For ONNX models
fastapi>=0.100.0
uvicorn>=0.23.0
pillow>=9.0.0
numpy>=1.24.0
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  classifier:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models
```

---

## FastAPI Serving

### Classification Server

```python
# app.py
import io
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

app = FastAPI(title="AutoTimm Classifier")

# Load model
MODEL_PATH = "model_traced.pt"
model = torch.jit.load(MODEL_PATH)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

# Preprocessing
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Class names (replace with your classes)
CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3", "class_4"]


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    # Format response
    probs = probabilities[0].cpu().numpy()
    predictions = [
        {"class": CLASS_NAMES[i], "probability": float(probs[i])}
        for i in range(len(CLASS_NAMES))
    ]
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "filename": file.filename,
        "predictions": predictions[:5],
    }


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

        probs = probabilities[0].cpu().numpy()
        top_idx = int(np.argmax(probs))

        results.append(
            {
                "filename": file.filename,
                "class": CLASS_NAMES[top_idx],
                "confidence": float(probs[top_idx]),
            }
        )

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### ONNX Runtime Server

```python
# app_onnx.py
import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI(title="AutoTimm ONNX Classifier")

# Load ONNX model with GPU support
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession("model.onnx", providers=providers)

# Preprocessing
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
    """Preprocess image for ONNX inference."""
    # Resize
    image = image.resize((size, size), Image.BILINEAR)

    # Convert to numpy
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize
    img_array = (img_array - MEAN) / STD

    # Transpose to NCHW
    img_array = img_array.transpose(2, 0, 1)

    # Add batch dimension
    return img_array[np.newaxis, ...]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    input_tensor = preprocess(image)

    # Run inference
    outputs = session.run(None, {"image": input_tensor})
    logits = outputs[0]

    # Softmax
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    top_idx = int(np.argmax(probs))
    confidence = float(probs[0, top_idx])

    return {
        "class_id": top_idx,
        "confidence": confidence,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## TorchServe Setup

TorchServe is an official PyTorch model serving solution.

### Create Model Archive

```python
# handler.py
import io
import torch
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler


class ClassificationHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image)).convert("RGB")
            image = self.transform(image)
            images.append(image)
        return torch.stack(images)

    def inference(self, data):
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, inference_output):
        probs = torch.softmax(inference_output, dim=1)
        return probs.tolist()
```

### Package Model

```bash
# Create model archive
torch-model-archiver \
    --model-name classifier \
    --version 1.0 \
    --serialized-file model_traced.pt \
    --handler handler.py \
    --export-path model_store

# Start TorchServe
torchserve --start \
    --model-store model_store \
    --models classifier=classifier.mar
```

### TorchServe Config

```properties
# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
number_of_netty_threads=4
job_queue_size=10
model_store=model_store
model_snapshot_interval=0
```

---

## Best Practices

### Model Versioning

```python
import os
from datetime import datetime


def save_versioned_model(model, base_path, model_name):
    """Save model with version timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(base_path, f"{model_name}_{timestamp}.pt")

    torch.jit.save(model, version_path)

    # Create/update latest symlink
    latest_path = os.path.join(base_path, f"{model_name}_latest.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(version_path, latest_path)

    return version_path
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health = {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        health["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"

    return health
```

### Monitoring

```python
import time
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
REQUEST_COUNT = Counter("requests_total", "Total requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency")
PREDICTION_CONFIDENCE = Histogram("prediction_confidence", "Prediction confidence")


@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUEST_COUNT.inc()

    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    REQUEST_LATENCY.observe(latency)
    return response


# Start metrics server
start_http_server(9090)
```

### Error Handling

```python
from fastapi import HTTPException


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    try:
        # ... inference code ...
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return result
```

---

## Complete Deployment Example

```python
# deploy.py - Complete deployment script
import os
import torch
from autotimm import ImageClassifier, MetricConfig


def deploy_model(
    checkpoint_path: str,
    backbone: str,
    num_classes: int,
    output_dir: str,
    image_size: int = 224,
    export_onnx: bool = True,
    quantize: bool = True,
):
    """Complete model deployment pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        )
    ]

    model = ImageClassifier.load_from_checkpoint(
        checkpoint_path,
        backbone=backbone,
        num_classes=num_classes,
        metrics=metrics,
    )
    model.eval()

    example_input = torch.randn(1, 3, image_size, image_size)

    # 1. Export TorchScript
    print("Exporting TorchScript...")
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        traced = torch.jit.optimize_for_inference(traced)
    traced.save(os.path.join(output_dir, "model_traced.pt"))

    # 2. Export ONNX
    if export_onnx:
        print("Exporting ONNX...")
        torch.onnx.export(
            model,
            example_input,
            os.path.join(output_dir, "model.onnx"),
            opset_version=14,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )

    # 3. Quantize
    if quantize:
        print("Creating quantized model...")
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        torch.save(quantized.state_dict(), os.path.join(output_dir, "model_quantized.pth"))

    print(f"Deployment artifacts saved to {output_dir}")
    print("Files:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f)) / 1e6
        print(f"  {f}: {size:.1f} MB")


if __name__ == "__main__":
    deploy_model(
        checkpoint_path="checkpoints/best.ckpt",
        backbone="resnet50",
        num_classes=10,
        output_dir="deployment",
    )
```

---

## See Also

- [Model Export](inference/model-export.md) - Detailed export documentation
- [Benchmarks](benchmarks.md) - Model performance comparison
- [Troubleshooting](troubleshooting.md) - Common deployment issues
