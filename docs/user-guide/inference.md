# Inference

This guide covers how to use trained AutoTimm models for inference and prediction.

## Basic Inference

### Load a Trained Model

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

### Single Image Prediction

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
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

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

## GPU Inference

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

with torch.no_grad():
    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
```

## Top-K Predictions

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
results = get_topk_predictions(model, input_tensor, k=5, class_names=data.class_names)
for r in results:
    print(f"{r['class']}: {r['probability']:.2%}")
```

## Export for Production

### TorchScript

```python
# Trace the model
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save("model_scripted.pt")

# Load and use
loaded_model = torch.jit.load("model_scripted.pt")
output = loaded_model(input_tensor)
```

### ONNX Export

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=11,
)

# Verify
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### ONNX Inference

```python
import onnxruntime as ort
import numpy as np

# Load
session = ort.InferenceSession("model.onnx")

# Prepare input
image_np = input_tensor.numpy()

# Run inference
outputs = session.run(None, {"image": image_np})
logits = outputs[0]
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
```

## Inference Pipeline

Complete inference pipeline example:

```python
import torch
from PIL import Image
from torchvision import transforms
from autotimm import ImageClassifier, MetricConfig


class InferencePipeline:
    """End-to-end inference pipeline."""

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

        self.model = ImageClassifier.load_from_checkpoint(
            checkpoint_path,
            backbone=backbone,
            num_classes=num_classes,
            metrics=metrics,
        )
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.class_names = class_names

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, image_path):
        """Predict class for a single image."""
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs.max().item()

        pred_class = self.class_names[pred_idx] if self.class_names else pred_idx

        return {
            "class": pred_class,
            "class_index": pred_idx,
            "confidence": confidence,
            "probabilities": probs[0].cpu().tolist(),
        }

    def predict_batch(self, image_paths):
        """Predict classes for multiple images."""
        return [self.predict(path) for path in image_paths]


# Usage
pipeline = InferencePipeline(
    checkpoint_path="best-model.ckpt",
    backbone="resnet50",
    num_classes=10,
    class_names=["cat", "dog", "bird", ...],
)

result = pipeline.predict("test_image.jpg")
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")
```

## Performance Tips

1. **Batch predictions** - Process multiple images at once for GPU efficiency
2. **Use fp16** - Reduce memory and speed up inference:
   ```python
   model = model.half()
   input_tensor = input_tensor.half()
   ```
3. **Disable gradient tracking** - Always use `torch.no_grad()`
4. **Use TorchScript** - For production deployment
5. **Pre-load model** - Keep model in memory between predictions
