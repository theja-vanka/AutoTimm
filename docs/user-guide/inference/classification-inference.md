# Classification Model Inference

This guide covers how to use trained classification models for inference and prediction.

## Loading a Trained Model

```python
import torch
from autotimm import ImageClassifier, MetricConfig

# Define metrics (required for loading)
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
    ),
]

# Load from checkpoint
model = ImageClassifier.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)
model.eval()
```

---

## Single Image Prediction

```python
from PIL import Image
from torchvision import transforms

# Prepare transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform image
image = Image.open("image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities.max().item()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

---

## Batch Prediction

### Using DataLoader

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Prepare dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("path/to/images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Predict all
model.eval()
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, _ in dataloader:
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_predictions.extend(preds.tolist())
        all_probabilities.extend(probs.tolist())
```

### Using Trainer.predict

```python
from autotimm import AutoTrainer, ImageDataModule

data = ImageDataModule(
    data_dir="./test_images",
    image_size=224,
    batch_size=32,
)
data.setup("test")

trainer = AutoTrainer()
predictions = trainer.predict(model, dataloaders=data.test_dataloader())

# predictions is a list of batched probability tensors
all_probs = torch.cat(predictions, dim=0)
all_preds = all_probs.argmax(dim=1)
```

---

## GPU Inference

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

with torch.no_grad():
    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
```

**Performance Tips:**
- Always move model to GPU before inference
- Use `.to(device, non_blocking=True)` for faster transfers
- Process larger batches on GPU for better efficiency

---

## Top-K Predictions

Get the top K most likely classes:

```python
def get_topk_predictions(model, image_tensor, k=5, class_names=None):
    """Get top-k predictions with class names and probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        topk_probs, topk_indices = probs.topk(k, dim=1)

    results = []
    for i in range(k):
        idx = topk_indices[0, i].item()
        prob = topk_probs[0, i].item()
        name = class_names[idx] if class_names else str(idx)
        results.append({"class": name, "probability": prob})

    return results

# Usage
results = get_topk_predictions(
    model, 
    input_tensor, 
    k=5, 
    class_names=["cat", "dog", "bird", "fish", "hamster"]
)

for r in results:
    print(f"{r['class']}: {r['probability']:.2%}")
```

**Output Example:**
```
cat: 95.32%
dog: 3.21%
hamster: 1.15%
bird: 0.28%
fish: 0.04%
```

---

## Complete Inference Pipeline

Production-ready inference pipeline with error handling and caching:

```python
import torch
from PIL import Image
from torchvision import transforms
from autotimm import ImageClassifier, MetricConfig


class InferencePipeline:
    """End-to-end inference pipeline for classification."""

    def __init__(self, checkpoint_path, backbone, num_classes, class_names=None):
        # Define minimal metrics for loading
        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["val"],
            ),
        ]

        # Load model
        self.model = ImageClassifier.load_from_checkpoint(
            checkpoint_path,
            backbone=backbone,
            num_classes=num_classes,
            metrics=metrics,
        )
        self.model.eval()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.class_names = class_names
        self.num_classes = num_classes

        # Prepare transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, image_path, top_k=1):
        """Predict class for a single image."""
        # Load image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)

        # Get top-k results
        topk_probs, topk_indices = probs.topk(top_k, dim=1)

        results = []
        for i in range(top_k):
            idx = topk_indices[0, i].item()
            prob = topk_probs[0, i].item()
            class_name = self.class_names[idx] if self.class_names else idx

            results.append({
                "class": class_name,
                "class_index": idx,
                "confidence": prob,
            })

        # Return single result or list
        if top_k == 1:
            return results[0]
        return results

    def predict_batch(self, image_paths, top_k=1):
        """Predict classes for multiple images."""
        return [self.predict(path, top_k=top_k) for path in image_paths]

    def get_all_probabilities(self, image_path):
        """Get probabilities for all classes."""
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)

        return probs[0].cpu().tolist()


# Usage
pipeline = InferencePipeline(
    checkpoint_path="best-model.ckpt",
    backbone="resnet50",
    num_classes=10,
    class_names=["cat", "dog", "bird", "fish", "hamster", "rabbit", "mouse", "snake", "turtle", "frog"],
)

# Single prediction
result = pipeline.predict("test_image.jpg")
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")

# Top-5 predictions
top5 = pipeline.predict("test_image.jpg", top_k=5)
for i, r in enumerate(top5, 1):
    print(f"{i}. {r['class']}: {r['confidence']:.2%}")

# Batch predictions
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = pipeline.predict_batch(image_paths)
for path, result in zip(image_paths, results):
    print(f"{path}: {result['class']} ({result['confidence']:.2%})")
```

---

## Performance Optimization

### 1. Batch Processing

Process multiple images at once for GPU efficiency:

```python
def predict_batch_efficient(model, image_paths, transform, device, batch_size=32):
    """Efficient batch prediction."""
    model.eval()
    all_predictions = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Load and transform batch
        images = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            images.append(transform(img))

        # Stack into batch
        batch_tensor = torch.stack(images).to(device)

        # Predict
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        all_predictions.extend(preds.cpu().tolist())

    return all_predictions
```

### 2. Half Precision (FP16)

Reduce memory usage and increase speed:

```python
# Convert model to half precision
model = model.half()

# Convert input to half precision
input_tensor = input_tensor.half()

with torch.no_grad():
    logits = model(input_tensor)
```

**Note:** Requires a GPU with tensor cores (V100, A100, RTX series)

### 3. Compiled Models

Use `torch.compile` for optimized inference (PyTorch 2.0+):

```python
import torch

model = ImageClassifier.load_from_checkpoint(...)
model.eval()

# Compile model
model = torch.compile(model, mode="reduce-overhead")

# First run is slower (compilation)
# Subsequent runs are faster
with torch.no_grad():
    output = model(input_tensor)
```

### 4. Disable Gradient Tracking

Always use `torch.no_grad()` for inference:

```python
# Good - saves memory
with torch.no_grad():
    output = model(input_tensor)

# Also good
@torch.no_grad()
def predict(model, input_tensor):
    return model(input_tensor)

# Bad - wastes memory tracking gradients
output = model(input_tensor)  # Don't do this for inference!
```

---

## Common Issues

### Out of Memory

**Problem:** CUDA out of memory error

**Solutions:**
```python
# 1. Reduce batch size
batch_size = 16  # Instead of 32

# 2. Use smaller image size
transform = transforms.Resize(224)  # Instead of 384

# 3. Clear cache between batches
torch.cuda.empty_cache()

# 4. Use CPU for very large images
device = torch.device("cpu")
```

### Slow Inference

**Problem:** Predictions are too slow

**Solutions:**
```python
# 1. Use GPU
device = torch.device("cuda")

# 2. Increase batch size
batch_size = 64

# 3. Use half precision
model = model.half()

# 4. Pre-load model
# Keep model in memory between predictions
# Don't reload for each image
```

### Wrong Predictions

**Problem:** Model predictions don't match training performance

**Solutions:**
```python
# 1. Check transforms match training
# Use exact same normalization as training

# 2. Ensure model is in eval mode
model.eval()

# 3. Check input preprocessing
# Image should be RGB, not BGR
image = Image.open("img.jpg").convert("RGB")

# 4. Verify class mapping
# Ensure class names match training order
```

---

## See Also

- [Object Detection Inference](object-detection-inference.md) - For detection models
- [Model Export](model-export.md) - Export to TorchScript/ONNX
- [Training Guide](training.md) - How to train models
