# Reproducibility Issues

Problems with deterministic training and seeding.

## Setting Random Seeds

```python
import torch
import random
import numpy as np
import pytorch_lightning as pl

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

set_seed(42)

# Use deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure trainer
trainer = AutoTrainer(
    max_epochs=10,
    deterministic=True,
)
```

## Non-Deterministic Operations

```python
# Some operations are non-deterministic by design
# To identify them:
import torch
torch.use_deterministic_algorithms(True)

# This will raise errors for non-deterministic operations
# Common culprits:
# - torch.nn.functional.interpolate (bilinear mode)
# - torch.scatter_add_
# - Atomic operations in CUDA

# Workaround: disable or replace non-deterministic ops
```

## Results Still Vary Slightly

**Problem:** Small variations despite setting seed

**Solutions:**

```python
# 1. Enable strict deterministic mode
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    seed=42,
    deterministic=True,  # Ensure this is True
)

# 2. Disable torch.compile if causing issues
model = ImageClassifier(
    backbone="resnet50",
    seed=42,
    deterministic=True,
    compile_model=False,
)

# 3. Accept hardware-dependent small differences
# Some operations vary slightly between GPU types
```

## Deterministic Mode Too Slow

**Solution:**

```python
# Disable for faster (but less reproducible) training
model = ImageClassifier(
    backbone="resnet50",
    seed=42,
    deterministic=False,  # Faster
)
```

## Related Issues

- [Convergence](../training/convergence.md) - Training consistency
- [Distributed Training](../environment/distributed.md) - Multi-GPU reproducibility
