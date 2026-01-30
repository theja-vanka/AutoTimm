# Training Utilities Examples

This page demonstrates training utilities including auto-tuning, multi-GPU training, and inference.

## Auto-Tuning

Automatically find optimal learning rate and batch size.

```python
from autotimm import AutoTrainer, TunerConfig


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        tuner_config=TunerConfig(
            auto_lr=True,
            auto_batch_size=True,
            lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0, "num_training": 100},
            batch_size_kwargs={"mode": "power", "init_val": 16},
        ),
    )

    trainer.fit(model, datamodule=data)  # Runs tuning before training


if __name__ == "__main__":
    main()
```

**Auto-Tuning Features:**

- **Learning Rate Finder**: Runs a range test to find the optimal learning rate
- **Batch Size Finder**: Automatically finds the largest batch size that fits in memory
- **Smart Defaults**: Pre-configured settings that work for most cases

**TunerConfig Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `auto_lr` | Enable automatic learning rate finding | `False` |
| `auto_batch_size` | Enable automatic batch size finding | `False` |
| `lr_find_kwargs` | Arguments for LR finder | `{"min_lr": 1e-6, "max_lr": 1.0}` |
| `batch_size_kwargs` | Arguments for batch size finder | `{"mode": "power"}` |

---

## Multi-GPU Training

Distributed training across multiple GPUs.

```python
from autotimm import AutoTrainer


def main():
    # Strategy options: ddp, ddp_spawn, fsdp
    trainer = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices=2,  # Number of GPUs
        strategy="ddp",  # Distributed Data Parallel
        precision="bf16-mixed",  # Mixed precision training
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Multi-GPU Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `ddp` | Distributed Data Parallel | Most use cases (recommended) |
| `ddp_spawn` | DDP with process spawning | Debugging |
| `fsdp` | Fully Sharded Data Parallel | Very large models |

**Precision Options:**

| Precision | Description | Speed | Memory |
|-----------|-------------|-------|--------|
| `32` | Full precision (FP32) | Slowest | Highest |
| `16-mixed` | Mixed precision (FP16) | Faster | Lower |
| `bf16-mixed` | Mixed precision (BF16) | Faster | Lower |

**Tips:**
- Use `ddp` for production training
- Use `bf16-mixed` if your GPU supports it (A100, H100)
- Increase batch size proportionally to number of GPUs
- Set `MASTER_PORT` environment variable to avoid port conflicts

---

## Inference

Make predictions with a trained model.

```python
import torch
from PIL import Image
from torchvision import transforms

from autotimm import ImageClassifier, MetricConfig, MetricManager


def main():
    # Define metrics for loading
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        ),
    ]
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Load model
    model = ImageClassifier.load_from_checkpoint(
        "path/to/checkpoint.ckpt",
        backbone="resnet50",
        num_classes=10,
        metrics=metric_manager,
    )
    model.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Predict single image
    image = Image.open("test.jpg").convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = model(input_tensor).softmax(dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    print(f"Prediction: {pred}, Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
```

---

## Batch Inference

Process multiple images efficiently.

```python
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from autotimm import ImageClassifier, MetricConfig, MetricManager


class ImageFolderDataset(Dataset):
    """Simple dataset for inference."""
    
    def __init__(self, image_dir, transform):
        self.image_paths = list(Path(image_dir).glob("*.jpg"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), str(self.image_paths[idx])


def main():
    # Load model
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        ),
    ]
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)
    
    model = ImageClassifier.load_from_checkpoint(
        "path/to/checkpoint.ckpt",
        backbone="resnet50",
        num_classes=10,
        metrics=metric_manager,
    )
    model.eval()
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageFolderDataset("./test_images", transform)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    
    # Batch prediction
    results = []
    with torch.no_grad():
        for images, paths in dataloader:
            probs = model(images).softmax(dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()
            
            for path, pred, conf in zip(paths, preds, confs):
                results.append({
                    "path": path,
                    "prediction": int(pred),
                    "confidence": float(conf),
                })
    
    # Print results
    for result in results:
        print(f"{result['path']}: {result['prediction']} ({result['confidence']:.2%})")


if __name__ == "__main__":
    main()
```

---

## Running Training Utilities Examples

Clone the repository and run examples:

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[all]"

# Run auto-tuning example
python examples/auto_tuning.py

# Run multi-GPU training example
python examples/multi_gpu_training.py

# Run inference example
python examples/inference.py
```
