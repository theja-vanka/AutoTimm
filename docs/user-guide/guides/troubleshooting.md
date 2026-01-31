# Troubleshooting

This guide covers common issues you may encounter when training models with AutoTimm and their solutions.

## NaN Losses

NaN (Not a Number) losses typically indicate numerical instability during training.

### Common Causes

| Cause | Symptom | Solution |
|-------|---------|----------|
| Learning rate too high | Loss spikes then becomes NaN | Reduce `lr` by 10x |
| Gradient explosion | Gradients grow unbounded | Enable gradient clipping |
| Division by zero | Specific layer outputs NaN | Check for empty batches |
| Log of zero/negative | Classification with wrong labels | Verify label range |
| FP16 underflow | Training with mixed precision | Use bf16 instead |

### Solutions

#### 1. Reduce Learning Rate

```python
from autotimm import ImageClassifier, MetricConfig

metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val"],
    )
]

model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    lr=1e-5,  # Start with a lower learning rate
)
```

#### 2. Enable Gradient Clipping

```python
from autotimm import AutoTrainer

trainer = AutoTrainer(
    max_epochs=10,
    gradient_clip_val=1.0,  # Clip gradients to max norm of 1.0
)
```

#### 3. Use Learning Rate Finder

```python
from autotimm import AutoTrainer, TunerConfig

trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,
        lr_find_kwargs={
            "min_lr": 1e-7,
            "max_lr": 1.0,
            "num_training": 100,
        },
    ),
)
```

#### 4. Switch to BF16 Precision

```python
trainer = AutoTrainer(
    max_epochs=10,
    precision="bf16-mixed",  # More stable than fp16
)
```

### Debugging NaN Losses

```python
import torch

# Enable anomaly detection to find the source
torch.autograd.set_detect_anomaly(True)

# Train with anomaly detection
trainer.fit(model, datamodule=data)

# Remember to disable after debugging
torch.autograd.set_detect_anomaly(False)
```

---

## OOM Errors

Out of Memory (OOM) errors occur when GPU memory is exhausted.

### Memory Optimization Strategies

#### 1. Reduce Batch Size

```python
from autotimm import ImageDataModule

data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=16,  # Reduce from 64 to 16
)
```

#### 2. Use Gradient Accumulation

Simulate larger batches without increased memory:

```python
trainer = AutoTrainer(
    max_epochs=10,
    accumulate_grad_batches=4,  # Effective batch = 16 * 4 = 64
)
```

#### 3. Enable Mixed Precision

```python
trainer = AutoTrainer(
    max_epochs=10,
    precision="16-mixed",  # Uses less memory than fp32
)
```

#### 4. Use a Smaller Backbone

```python
# Instead of large models
model = ImageClassifier(backbone="resnet50", ...)  # ~98MB

# Use smaller alternatives
model = ImageClassifier(backbone="resnet34", ...)  # ~84MB
model = ImageClassifier(backbone="efficientnet_b0", ...)  # ~21MB
model = ImageClassifier(backbone="mobilenetv3_small_100", ...)  # ~10MB
```

#### 5. Reduce Image Size

```python
data = ImageDataModule(
    data_dir="./data",
    image_size=160,  # Reduce from 224
)
```

### Memory Usage by Task

| Task | Typical Memory Usage | Recommended Batch Size |
|------|---------------------|----------------------|
| Classification | 4-8 GB | 32-64 |
| Object Detection | 8-16 GB | 8-16 |
| Semantic Segmentation | 8-12 GB | 8-16 |
| Instance Segmentation | 12-24 GB | 4-8 |

### Monitoring GPU Memory

```python
import torch

# Check current memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

---

## LR Tuning Failures

The learning rate finder may fail or produce suboptimal results.

### Common Issues

#### 1. Finder Doesn't Converge

```python
# Increase the number of training steps
trainer = AutoTrainer(
    tuner_config=TunerConfig(
        auto_lr=True,
        lr_find_kwargs={
            "num_training": 200,  # Increase from default 100
            "early_stop_threshold": None,  # Disable early stopping
        },
    ),
)
```

#### 2. Suggested LR Too High

```python
# Use a more conservative range
trainer = AutoTrainer(
    tuner_config=TunerConfig(
        auto_lr=True,
        lr_find_kwargs={
            "min_lr": 1e-6,
            "max_lr": 1e-2,  # Lower max
        },
    ),
)
```

#### 3. Manual LR Selection

If auto-tuning fails, use these guidelines:

| Backbone Type | Starting LR | With Pretrained |
|--------------|-------------|-----------------|
| CNN (ResNet, EfficientNet) | 1e-3 | 1e-4 |
| Transformer (ViT, Swin) | 1e-4 | 1e-5 |
| Detection models | 1e-4 | 1e-4 |
| Segmentation models | 1e-4 | 1e-4 |

### LR Schedule Recommendations

```python
# For CNN backbones
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    lr=1e-4,
    scheduler="cosineannealinglr",
    scheduler_kwargs={"T_max": 50, "eta_min": 1e-6},
)

# For Transformer backbones
model = ImageClassifier(
    backbone="vit_base_patch16_224",
    num_classes=10,
    metrics=metrics,
    lr=1e-5,
    scheduler="cosineannealinglr",
    scheduler_kwargs={"T_max": 50, "eta_min": 1e-7},
)
```

---

## Convergence Problems

### Overfitting

**Symptoms:** Training loss decreases, validation loss increases or plateaus.

**Solutions:**

```python
# 1. Add dropout
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    head_dropout=0.5,  # Add dropout before classification head
)

# 2. Use data augmentation
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    augmentation_preset="strong",  # Stronger augmentation
)

# 3. Add early stopping
from pytorch_lightning.callbacks import EarlyStopping

trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[
        EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min",
        ),
    ],
)

# 4. Use weight decay
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    weight_decay=1e-4,
)
```

### Underfitting

**Symptoms:** Both training and validation loss remain high.

**Solutions:**

```python
# 1. Train longer
trainer = AutoTrainer(max_epochs=100)

# 2. Use a larger model
model = ImageClassifier(
    backbone="resnet101",  # Larger backbone
    num_classes=10,
    metrics=metrics,
)

# 3. Increase learning rate
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    lr=1e-3,  # Higher LR
)

# 4. Reduce regularization
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    weight_decay=0,
    head_dropout=0,
)
```

### Oscillating Loss

**Symptoms:** Loss fluctuates significantly without converging.

**Solutions:**

```python
# 1. Reduce learning rate
model = ImageClassifier(lr=1e-5, ...)

# 2. Increase batch size
data = ImageDataModule(batch_size=128, ...)

# 3. Use a scheduler with warm restarts
model = ImageClassifier(
    scheduler="cosineannealingwarmrestarts",
    scheduler_kwargs={"T_0": 10, "T_mult": 2},
    ...
)
```

---

## Gradient Explosion

### Detection

```python
from autotimm import LoggingConfig

# Enable gradient norm logging
logging_config = LoggingConfig(
    log_learning_rate=True,
    log_gradient_norm=True,  # Monitor gradients
)

model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    logging_config=logging_config,
)
```

### Prevention

```python
# 1. Gradient clipping (always recommended for detection/segmentation)
trainer = AutoTrainer(
    max_epochs=10,
    gradient_clip_val=1.0,  # Clip by norm
)

# Alternative: clip by value
trainer = AutoTrainer(
    max_epochs=10,
    gradient_clip_val=0.5,
    gradient_clip_algorithm="value",
)

# 2. Layer-wise learning rates for transformers
# Use lower LR for early layers, higher for later layers
model = ImageClassifier(
    backbone="vit_base_patch16_224",
    num_classes=10,
    metrics=metrics,
    lr=1e-5,  # Conservative base LR
)
```

---

## Slow Training

### Data Loading Bottleneck

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

### GPU Utilization

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

### Validation Frequency

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

---

## Device Errors

### CUDA Errors

#### CUDA Out of Memory

See [OOM Errors](#oom-errors) section.

#### CUDA Device Not Found

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")

# Explicitly set device
trainer = AutoTrainer(
    accelerator="gpu",
    devices=[0],  # Use specific GPU
)
```

#### CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version if needed
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Multi-GPU Errors

```python
# Basic multi-GPU setup
trainer = AutoTrainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp",  # Distributed Data Parallel
)

# If DDP hangs, try different strategy
trainer = AutoTrainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp_spawn",  # Alternative DDP strategy
)

# For debugging, use single GPU first
trainer = AutoTrainer(
    accelerator="gpu",
    devices=1,
)
```

### MPS (Apple Silicon) Issues

```python
import torch

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")

# Use MPS
trainer = AutoTrainer(
    accelerator="mps",
    devices=1,
)

# Fallback to CPU if MPS has issues
trainer = AutoTrainer(
    accelerator="cpu",
)
```

---

## Common Error Reference

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `RuntimeError: CUDA out of memory` | Batch size too large | Reduce batch size, use gradient accumulation |
| `RuntimeError: CUDA error: device-side assert` | Invalid tensor values | Check labels are in valid range |
| `ValueError: Expected input batch_size to match target batch_size` | Mismatched batch dimensions | Check data loader output shapes |
| `RuntimeError: Given groups=1, weight of size [X], expected input[Y]` | Wrong input channels | Check image channels (RGB vs grayscale) |
| `IndexError: index out of range` | Label exceeds num_classes | Verify num_classes matches actual labels |
| `RuntimeError: element 0 of tensors does not require grad` | Frozen model or detached tensor | Check model.train() is called |
| `ValueError: optimizer got an empty parameter list` | No trainable parameters | Check model isn't fully frozen |
| `RuntimeError: Expected all tensors on same device` | Mixed CPU/GPU tensors | Ensure all inputs are on same device |
| `FileNotFoundError: No such file or directory` | Wrong data path | Verify data_dir path exists |
| `KeyError: 'images'` | Wrong COCO annotation format | Check annotation JSON structure |

---

## Getting Help

If you encounter an issue not covered here:

1. **Check the logs**: Enable verbose logging to get more details
   ```python
   logging_config = LoggingConfig(verbosity=2)
   ```

2. **Minimal reproduction**: Create a minimal example that reproduces the issue

3. **Report issues**: Open an issue at [GitHub Issues](https://github.com/theja-vanka/AutoTimm/issues)

---

## See Also

- [Training Guide](training.md) - Complete training documentation
- [Metrics Guide](metrics.md) - Metric configuration
- [Logging Guide](logging.md) - Logging configuration
