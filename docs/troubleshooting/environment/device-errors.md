# Device Errors

CUDA, MPS, and multi-GPU issues.

## CUDA Errors

### CUDA Out of Memory

See [OOM Errors](../performance/oom-errors.md) section.

### CUDA Device Not Found

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

### CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version if needed
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Multi-GPU Errors

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

## MPS (Apple Silicon) Issues

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

## Related Issues

- [OOM Errors](../performance/oom-errors.md) - Memory problems
- [Distributed Training](distributed.md) - Multi-GPU and multi-node issues
- [Installation](installation.md) - CUDA installation problems
