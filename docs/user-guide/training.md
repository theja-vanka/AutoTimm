# Training

AutoTimm provides `AutoTrainer`, a configured PyTorch Lightning Trainer with sensible defaults and built-in support for auto-tuning, logging, and checkpointing.

## AutoTrainer

### Basic Training

```python
from autotimm import AutoTrainer

trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

### With Checkpointing

Save the best model based on a metric:

```python
trainer = AutoTrainer(
    max_epochs=10,
    checkpoint_monitor="val/accuracy",
    checkpoint_mode="max",  # "max" for accuracy, "min" for loss
)
```

### GPU Training

```python
trainer = AutoTrainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
)
```

### Mixed Precision

Speed up training with mixed precision:

```python
trainer = AutoTrainer(
    max_epochs=10,
    precision="bf16-mixed",  # or "16-mixed" for fp16
)
```

## Auto-Tuning

### TunerConfig

Automatically find optimal learning rate and batch size:

```python
from autotimm import AutoTrainer, TunerConfig

trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,
        auto_batch_size=True,
        lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0, "num_training": 100},
        batch_size_kwargs={"mode": "power", "init_val": 16},
    ),
)

trainer.fit(model, datamodule=data)  # Runs tuning before training
```

### LR Finding Only

```python
trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,
        auto_batch_size=False,
    ),
)
```

### TunerConfig Parameters

| Parameter | Description |
|-----------|-------------|
| `auto_lr` | Enable learning rate finder |
| `auto_batch_size` | Enable batch size finder |
| `lr_find_kwargs` | `min_lr`, `max_lr`, `num_training`, `mode`, `early_stop_threshold` |
| `batch_size_kwargs` | `mode` ("power" or "binsearch"), `steps_per_trial`, `init_val`, `max_trials` |

## Multi-GPU Training

### Data Parallel (DDP)

```python
trainer = AutoTrainer(
    max_epochs=10,
    accelerator="gpu",
    devices=2,              # Number of GPUs
    strategy="ddp",         # Distributed Data Parallel
    precision="bf16-mixed",
)
```

### Multi-Node Training

```python
trainer = AutoTrainer(
    max_epochs=10,
    accelerator="gpu",
    devices=4,
    num_nodes=2,            # 2 nodes with 4 GPUs each
    strategy="ddp",
)
```

## Gradient Accumulation

Simulate larger batches on limited memory:

```python
trainer = AutoTrainer(
    max_epochs=10,
    accumulate_grad_batches=4,  # Effective batch = batch_size * 4
)
```

## Gradient Clipping

Prevent exploding gradients:

```python
trainer = AutoTrainer(
    max_epochs=10,
    gradient_clip_val=1.0,      # Clip to max norm of 1.0
)
```

## Validation Frequency

Control how often validation runs:

```python
# Every epoch (default)
trainer = AutoTrainer(max_epochs=10, val_check_interval=1.0)

# Every 500 steps
trainer = AutoTrainer(max_epochs=10, val_check_interval=500)

# Every 25% of an epoch
trainer = AutoTrainer(max_epochs=10, val_check_interval=0.25)
```

## Callbacks

Add custom callbacks:

```python
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[
        EarlyStopping(monitor="val/loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ],
)
```

## Resume Training

Resume from a checkpoint:

```python
trainer = AutoTrainer(max_epochs=20)
trainer.fit(model, datamodule=data, ckpt_path="path/to/checkpoint.ckpt")
```

## Full Training Example

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggingConfig,
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
    num_workers=8,
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

# Model with enhanced logging
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    logging_config=LoggingConfig(
        log_learning_rate=True,
        log_gradient_norm=True,
    ),
)

# Trainer with all features
trainer = AutoTrainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
    tuner_config=TunerConfig(auto_lr=True, auto_batch_size=False),
    checkpoint_monitor="val/accuracy",
    gradient_clip_val=1.0,
    callbacks=[EarlyStopping(monitor="val/loss", patience=10, mode="min")],
)

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

## Full Parameter Reference

```python
AutoTrainer(
    max_epochs=10,                  # Training epochs
    accelerator="auto",             # "auto", "gpu", "cpu", "tpu"
    devices="auto",                 # Number of devices or "auto"
    precision=32,                   # 32, 16, "bf16-mixed", "16-mixed"
    logger=False,                   # LoggerManager, list of LoggerConfig, or False
    tuner_config=None,              # TunerConfig for auto-tuning
    checkpoint_monitor=None,        # Metric to monitor for checkpointing
    checkpoint_mode="max",          # "max" or "min"
    callbacks=None,                 # List of Lightning callbacks
    default_root_dir=".",           # Root directory for logs/checkpoints
    gradient_clip_val=None,         # Gradient clipping value
    accumulate_grad_batches=1,      # Gradient accumulation steps
    val_check_interval=1.0,         # Validation frequency
    enable_checkpointing=True,      # Save checkpoints
    # ... any other pl.Trainer argument
)
```
