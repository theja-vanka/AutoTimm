# Loggers

Logger configuration and management for experiment tracking.

## LoggerConfig

Configuration for a single logger backend.

### API Reference

::: autotimm.LoggerConfig
    options:
      show_source: true

### Usage Examples

#### TensorBoard

```python
from autotimm import LoggerConfig

tb = LoggerConfig(
    backend="tensorboard",
    params={"save_dir": "logs", "name": "experiment_1"},
)
```

#### Weights & Biases

```python
wandb = LoggerConfig(
    backend="wandb",
    params={
        "project": "my-project",
        "name": "run-1",
        "tags": ["resnet", "cifar10"],
    },
)
```

#### MLflow

```python
mlflow = LoggerConfig(
    backend="mlflow",
    params={
        "experiment_name": "cifar10-classification",
        "tracking_uri": "http://localhost:5000",
    },
)
```

#### CSV Logger

```python
csv = LoggerConfig(
    backend="csv",
    params={"save_dir": "logs/csv", "name": "metrics"},
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | Required | Logger type |
| `params` | `dict` | `{}` | Backend-specific params |

### Supported Backends

| Backend | Required Params | Install |
|---------|-----------------|---------|
| `tensorboard` | `save_dir` | `pip install autotimm[tensorboard]` |
| `csv` | `save_dir` | Built-in |
| `wandb` | `project` | `pip install autotimm[wandb]` |
| `mlflow` | `experiment_name` | `pip install autotimm[mlflow]` |

---

## LoggerManager

Manages multiple PyTorch Lightning loggers.

### API Reference

::: autotimm.LoggerManager
    options:
      show_source: true
      members:
        - __init__
        - loggers
        - configs
        - get_logger_by_backend

### Usage Examples

#### Basic Usage

```python
from autotimm import LoggerConfig, LoggerManager

manager = LoggerManager(configs=[
    LoggerConfig(backend="tensorboard", params={"save_dir": "logs/tb"}),
    LoggerConfig(backend="csv", params={"save_dir": "logs/csv"}),
])
```

#### With AutoTrainer

```python
from autotimm import AutoTrainer

trainer = AutoTrainer(max_epochs=10, logger=manager)
```

#### Access Loggers

```python
# Get all loggers
all_loggers = manager.loggers

# Get by backend
tb_logger = manager.get_logger_by_backend("tensorboard")
csv_logger = manager.get_logger_by_backend("csv")

# Iterate
for logger in manager:
    print(type(logger))

# Length
print(f"Number of loggers: {len(manager)}")
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `configs` | `list[LoggerConfig]` | List of logger configs |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `loggers` | `list[Logger]` | All instantiated loggers |
| `configs` | `list[LoggerConfig]` | Original configs |
| `get_logger_by_backend(name)` | `Logger \| None` | Find logger by backend |
| `len(manager)` | `int` | Number of loggers |
| `iter(manager)` | Iterator | Iterate over loggers |
| `manager[i]` | `Logger` | Get logger by index |

---

## Backend Parameters

### TensorBoard

```python
LoggerConfig(
    backend="tensorboard",
    params={
        "save_dir": "logs",           # Required
        "name": "experiment",         # Subdirectory
        "version": "v1",              # Version string
        "log_graph": True,            # Log model graph
        "default_hp_metric": False,   # HP metric logging
        "prefix": "",                 # Metric prefix
        "sub_dir": None,              # Additional subdirectory
    },
)
```

### Weights & Biases

```python
LoggerConfig(
    backend="wandb",
    params={
        "project": "my-project",      # Required
        "name": "run-1",              # Run name
        "id": None,                   # Run ID (for resuming)
        "tags": ["tag1", "tag2"],     # Tags
        "notes": "Experiment notes",  # Description
        "group": "experiment-group",  # Group runs
        "job_type": "training",       # Job type
        "entity": None,               # Team/user
        "save_dir": "wandb_logs",     # Local save directory
        "offline": False,             # Offline mode
        "log_model": False,           # Log model artifacts
        "prefix": "",                 # Metric prefix
    },
)
```

### MLflow

```python
LoggerConfig(
    backend="mlflow",
    params={
        "experiment_name": "exp",     # Required
        "run_name": "run-1",          # Run name
        "tracking_uri": None,         # MLflow server URL
        "tags": {"env": "dev"},       # Tags
        "save_dir": "mlruns",         # Local artifacts
        "log_model": False,           # Log model
        "prefix": "",                 # Metric prefix
        "artifact_location": None,    # Artifact storage
        "run_id": None,               # For resuming
    },
)
```

### CSV Logger

```python
LoggerConfig(
    backend="csv",
    params={
        "save_dir": "logs",           # Required
        "name": "metrics",            # Subdirectory
        "version": None,              # Auto-increment if None
        "prefix": "",                 # Metric prefix
        "flush_logs_every_n_steps": 100,
    },
)
```

---

## Full Example

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    LoggerManager,
    MetricConfig,
)

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

# Multiple loggers
logger_manager = LoggerManager(configs=[
    LoggerConfig(
        backend="tensorboard",
        params={"save_dir": "logs/tb", "name": "cifar10"},
    ),
    LoggerConfig(
        backend="csv",
        params={"save_dir": "logs/csv"},
    ),
    LoggerConfig(
        backend="wandb",
        params={"project": "cifar10-experiments", "name": "resnet50-run"},
    ),
])

# Trainer
trainer = AutoTrainer(
    max_epochs=10,
    logger=logger_manager,
    checkpoint_monitor="val/accuracy",
)

# Train
trainer.fit(model, datamodule=data)

# Access specific logger after training
tb = logger_manager.get_logger_by_backend("tensorboard")
print(f"TensorBoard log dir: {tb.log_dir}")
```
