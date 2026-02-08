# Slow Training

Training speed optimization and identifying bottlenecks.

## Data Loading Bottleneck

```python
# Check if CPU is the bottleneck (high GPU idle time)

# 1. Increase workers
data = ImageDataModule(
    data_dir="./data",
    num_workers=8,  # Increase based on CPU cores
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
)

# 2. Use faster storage (SSD over HDD)

# 3. Preprocess data once
# Convert images to a faster format like WebP or use lmdb
```

## GPU Utilization

```python
# 1. Increase batch size (use gradient accumulation if memory limited)
data = ImageDataModule(batch_size=64, ...)

# 2. Use mixed precision
trainer = AutoTrainer(precision="bf16-mixed", ...)

# 3. Compile the model (PyTorch 2.0+)
import torch

model = ImageClassifier(...)
model = torch.compile(model, mode="reduce-overhead")
```

## Validation Frequency

```python
# Reduce validation frequency for faster training
trainer = AutoTrainer(
    max_epochs=50,
    val_check_interval=0.5,  # Validate twice per epoch instead of every epoch
)

# Or validate every N steps
trainer = AutoTrainer(
    max_epochs=50,
    val_check_interval=500,  # Validate every 500 steps
)
```

## Related Issues

- [OOM Errors](oom-errors.md) - Memory constraints
- [Profiling](profiling.md) - Detailed performance analysis
- [Device Errors](../environment/device-errors.md) - Hardware issues
