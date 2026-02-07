# NaN Losses

NaN (Not a Number) losses typically indicate numerical instability during training.

## Common Causes

| Cause | Symptom | Solution |
|-------|---------|----------|
| Learning rate too high | Loss spikes then becomes NaN | Reduce `lr` by 10x |
| Gradient explosion | Gradients grow unbounded | Enable gradient clipping |
| Division by zero | Specific layer outputs NaN | Check for empty batches |
| Log of zero/negative | Classification with wrong labels | Verify label range |
| FP16 underflow | Training with mixed precision | Use bf16 instead |

## Solutions

### 1. Reduce Learning Rate

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

### 2. Enable Gradient Clipping

```python
from autotimm import AutoTrainer

trainer = AutoTrainer(
    max_epochs=10,
    gradient_clip_val=1.0,  # Clip gradients to max norm of 1.0
)
```

### 3. Use Learning Rate Finder

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

### 4. Switch to BF16 Precision

```python
trainer = AutoTrainer(
    max_epochs=10,
    precision="bf16-mixed",  # More stable than fp16
)
```

## Debugging NaN Losses

```python
import torch

# Enable anomaly detection to find the source
torch.autograd.set_detect_anomaly(True)

# Train with anomaly detection
trainer.fit(model, datamodule=data)

# Remember to disable after debugging
torch.autograd.set_detect_anomaly(False)
```

## Related Issues

- [Gradient Explosion](gradient-issues.md) - Related gradient problems
- [LR Tuning](lr-tuning.md) - Finding the right learning rate
- [Convergence Problems](convergence.md) - Other training instabilities
