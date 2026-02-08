# Gradient Issues

Problems with gradient explosion and gradient monitoring.

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

## Gradient Vanishing

### Symptoms
- Gradients become very small in early layers
- Model doesn't learn

### Solutions

```python
# 1. Use residual connections (built into most modern backbones)
model = ImageClassifier(backbone="resnet50", ...)

# 2. Use batch normalization (also built-in)

# 3. Increase learning rate slightly
model = ImageClassifier(lr=1e-3, ...)

# 4. Use better initialization (automatic in timm)
```

## Related Issues

- [NaN Losses](nan-losses.md) - Numerical instability
- [Convergence Problems](convergence.md) - Training doesn't improve
- [LR Tuning](lr-tuning.md) - Learning rate problems
