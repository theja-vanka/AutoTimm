# OOM Errors

Out of Memory (OOM) errors occur when GPU memory is exhausted.

## Memory Optimization Strategies

### 1. Reduce Batch Size

```python
from autotimm import ImageDataModule

data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=16,  # Reduce from 64 to 16
)
```

### 2. Use Gradient Accumulation

Simulate larger batches without increased memory:

```python
trainer = AutoTrainer(
    max_epochs=10,
    accumulate_grad_batches=4,  # Effective batch = 16 * 4 = 64
)
```

### 3. Enable Mixed Precision

```python
trainer = AutoTrainer(
    max_epochs=10,
    precision="16-mixed",  # Uses less memory than fp32
)
```

### 4. Use a Smaller Backbone

```python
# Instead of large models
model = ImageClassifier(backbone="resnet50", ...)  # ~98MB

# Use smaller alternatives
model = ImageClassifier(backbone="resnet34", ...)  # ~84MB
model = ImageClassifier(backbone="efficientnet_b0", ...)  # ~21MB
model = ImageClassifier(backbone="mobilenetv3_small_100", ...)  # ~10MB
```

### 5. Reduce Image Size

```python
data = ImageDataModule(
    data_dir="./data",
    image_size=160,  # Reduce from 224
)
```

## Memory Usage by Task

| Task | Typical Memory Usage | Recommended Batch Size |
|------|---------------------|----------------------|
| Classification | 4-8 GB | 32-64 |
| Object Detection | 8-16 GB | 8-16 |
| Semantic Segmentation | 8-12 GB | 8-16 |
| Instance Segmentation | 12-24 GB | 4-8 |

## Monitoring GPU Memory

```python
import torch

# Check current memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

## Related Issues

- [Slow Training](slow-training.md) - Performance optimization
- [Device Errors](../environment/device-errors.md) - CUDA problems
- [Profiling](profiling.md) - Finding memory bottlenecks
