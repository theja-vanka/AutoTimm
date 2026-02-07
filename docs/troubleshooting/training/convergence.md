# Convergence Problems

Common convergence issues and how to fix them.

## Overfitting

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

## Underfitting

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

## Oscillating Loss

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

## Related Issues

- [NaN Losses](nan-losses.md) - Numerical instability
- [LR Tuning](lr-tuning.md) - Learning rate problems
- [Gradient Issues](gradient-issues.md) - Gradient explosion/vanishing
