# AutoTrainer & TunerConfig

Configured PyTorch Lightning Trainer with auto-tuning support. For YAML-config-driven training from the command line, see [AutoTimmCLI](cli.md).

## AutoTrainer

A convenience wrapper around `pl.Trainer` with sensible defaults for AutoTimm.

### API Reference

::: autotimm.AutoTrainer
    options:
      show_source: true
      members:
        - __init__
        - fit
        - tuner_config

### Usage Examples

#### Basic Training

```python
from autotimm import AutoTrainer

trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

#### With Logging and Checkpointing

```python
from autotimm import AutoTrainer, LoggerConfig

trainer = AutoTrainer(
    max_epochs=10,
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
    checkpoint_monitor="val/accuracy",
    checkpoint_mode="max",
)
```

#### GPU Training with Mixed Precision

```python
trainer = AutoTrainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
)
```

#### Multi-GPU Training

```python
trainer = AutoTrainer(
    max_epochs=10,
    accelerator="gpu",
    devices=2,
    strategy="ddp",
)
```

#### With Auto-Tuning

```python
from autotimm import AutoTrainer, TunerConfig

trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,
        auto_batch_size=True,
    ),
)
trainer.fit(model, datamodule=data)  # Runs tuning first
```

#### With Reproducibility Settings

```python
# Default: seed=42, deterministic=True (Lightning's seeding)
trainer = AutoTrainer(max_epochs=10)

# Custom seed
trainer = AutoTrainer(max_epochs=10, seed=123)

# Faster training (disable deterministic mode)
trainer = AutoTrainer(max_epochs=10, deterministic=False)

# Use AutoTimm's custom seeding instead of Lightning's
trainer = AutoTrainer(max_epochs=10, use_autotimm_seeding=True)

# Disable seeding completely (set deterministic=False to avoid warning)
trainer = AutoTrainer(max_epochs=10, seed=None, deterministic=False)
```

#### With Gradient Accumulation

```python
trainer = AutoTrainer(
    max_epochs=10,
    accumulate_grad_batches=4,
    gradient_clip_val=1.0,
)
```

#### Fast Development Run

```python
# Run 1 batch for quick debugging
trainer = AutoTrainer(fast_dev_run=True)
trainer.fit(model, datamodule=data)

# Run 5 batches for testing
trainer = AutoTrainer(fast_dev_run=5)
trainer.fit(model, datamodule=data)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | `int` | `10` | Training epochs |
| `accelerator` | `str` | `"auto"` | `"auto"`, `"gpu"`, `"cpu"`, `"tpu"` |
| `devices` | `int \| str` | `"auto"` | Number of devices |
| `precision` | `str \| int` | `32` | `32`, `16`, `"bf16-mixed"`, `"16-mixed"` |
| `logger` | Various | `False` | Logger configuration |
| `tuner_config` | `TunerConfig \| None \| bool` | `None` | Auto-tuning config. `None`/`True` creates default config, `False` disables |
| `seed` | `int \| None` | `42` | Random seed for reproducibility. Set to `None` to disable seeding |
| `deterministic` | `bool` | `False` | Enable deterministic algorithms for reproducibility. May impact performance |
| `use_autotimm_seeding` | `bool` | `False` | Use AutoTimm's `seed_everything()` instead of Lightning's built-in seeding |
| `checkpoint_monitor` | `str \| None` | `None` | Metric for checkpointing |
| `checkpoint_mode` | `str` | `"max"` | `"max"` or `"min"` |
| `callbacks` | `list \| None` | `None` | Lightning callbacks |
| `default_root_dir` | `str` | `"."` | Root directory |
| `gradient_clip_val` | `float \| None` | `None` | Gradient clipping |
| `accumulate_grad_batches` | `int` | `1` | Gradient accumulation |
| `val_check_interval` | `float \| int` | `1.0` | Validation frequency |
| `enable_checkpointing` | `bool` | `True` | Save checkpoints |
| `fast_dev_run` | `bool \| int` | `False` | Run N batches for debugging |

---

## TunerConfig

Configuration for automatic hyperparameter tuning (LR and batch size finding).

### API Reference

::: autotimm.TunerConfig
    options:
      show_source: true

### Usage Examples

#### LR Finding Only

```python
from autotimm import TunerConfig

config = TunerConfig(
    auto_lr=True,
    auto_batch_size=False,
)
```

#### Full Auto-Tuning

```python
config = TunerConfig(
    auto_lr=True,
    auto_batch_size=True,
    lr_find_kwargs={
        "min_lr": 1e-6,
        "max_lr": 1.0,
        "num_training": 100,
    },
    batch_size_kwargs={
        "mode": "power",
        "init_val": 16,
    },
)
```

#### Custom LR Finder Settings

```python
config = TunerConfig(
    auto_lr=True,
    auto_batch_size=False,
    lr_find_kwargs={
        "min_lr": 1e-7,
        "max_lr": 10.0,
        "num_training": 200,
        "mode": "exponential",
        "early_stop_threshold": 4.0,
    },
)
```

#### Custom Batch Size Finder Settings

```python
config = TunerConfig(
    auto_lr=False,
    auto_batch_size=True,
    batch_size_kwargs={
        "mode": "binsearch",      # or "power"
        "steps_per_trial": 3,
        "init_val": 32,
        "max_trials": 25,
    },
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_lr` | `bool` | Required | Enable LR finder |
| `auto_batch_size` | `bool` | Required | Enable batch size finder |
| `lr_find_kwargs` | `dict \| None` | `None` | LR finder options |
| `batch_size_kwargs` | `dict \| None` | `None` | Batch size finder options |

### LR Finder Options (`lr_find_kwargs`)

| Option | Default | Description |
|--------|---------|-------------|
| `min_lr` | `1e-8` | Minimum learning rate |
| `max_lr` | `1.0` | Maximum learning rate |
| `num_training` | `100` | Training steps |
| `mode` | `"exponential"` | `"exponential"` or `"linear"` |
| `early_stop_threshold` | `4.0` | Stop if loss exceeds factor |

### Batch Size Finder Options (`batch_size_kwargs`)

| Option | Default | Description |
|--------|---------|-------------|
| `mode` | `"power"` | `"power"` (2x) or `"binsearch"` |
| `steps_per_trial` | `3` | Steps to run per trial |
| `init_val` | `2` | Initial batch size |
| `max_trials` | `25` | Maximum trials |

---

## Complete Example

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    TunerConfig,
)
from pytorch_lightning.callbacks import EarlyStopping

# Data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=64,
)

# Metrics
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
]

# Model
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)

# Trainer with all features
trainer = AutoTrainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    logger=[
        LoggerConfig(backend="tensorboard", params={"save_dir": "logs/tb"}),
        LoggerConfig(backend="csv", params={"save_dir": "logs/csv"}),
    ],
    tuner_config=TunerConfig(
        auto_lr=True,
        auto_batch_size=False,
        lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0},
    ),
    checkpoint_monitor="val/accuracy",
    checkpoint_mode="max",
    gradient_clip_val=1.0,
    callbacks=[
        EarlyStopping(monitor="val/loss", patience=10, mode="min"),
    ],
)

# Train
trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```
