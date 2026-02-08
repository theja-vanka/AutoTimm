# LR Tuning Failures

The learning rate finder may fail or produce suboptimal results.

## Common Issues

### 1. Finder Doesn't Converge

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

### 2. Suggested LR Too High

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

### 3. Manual LR Selection

If auto-tuning fails, use these guidelines:

| Backbone Type | Starting LR | With Pretrained |
|--------------|-------------|-----------------|
| CNN (ResNet, EfficientNet) | 1e-3 | 1e-4 |
| Transformer (ViT, Swin) | 1e-4 | 1e-5 |
| Detection models | 1e-4 | 1e-4 |
| Segmentation models | 1e-4 | 1e-4 |

## LR Schedule Recommendations

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

## Related Issues

- [NaN Losses](nan-losses.md) - Too high learning rate
- [Convergence Problems](convergence.md) - Training doesn't improve
- [Gradient Issues](gradient-issues.md) - Gradient explosion from high LR
