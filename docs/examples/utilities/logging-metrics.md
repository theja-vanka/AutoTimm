# Logging and Metrics Examples

This page demonstrates logging configurations, MLflow tracking, and detailed evaluation metrics.

## Multiple Loggers

Log to TensorBoard and CSV simultaneously.

```python
from autotimm import AutoTrainer, LoggerConfig


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[
            LoggerConfig(backend="tensorboard", params={"save_dir": "logs/tb"}),
            LoggerConfig(backend="csv", params={"save_dir": "logs/csv"}),
        ],
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Available Loggers:**

| Logger | Backend | Key Parameters |
|--------|---------|----------------|
| TensorBoard | `tensorboard` | `save_dir`, `name`, `version` |
| Weights & Biases | `wandb` | `project`, `name`, `entity` |
| MLflow | `mlflow` | `experiment_name`, `tracking_uri` |
| CSV | `csv` | `save_dir`, `name`, `version` |

---

## MLflow Tracking

Track experiments with MLflow.

```python
from autotimm import AutoTrainer, LoggerConfig


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        logger=[
            LoggerConfig(
                backend="mlflow",
                params={
                    "experiment_name": "cifar10-experiments",
                    "tracking_uri": "http://localhost:5000",
                    "tags": {"model": "resnet50", "dataset": "cifar10"},
                },
            ),
        ],
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**MLflow Setup:**

```bash
# Install MLflow
pip install mlflow

# Start MLflow UI
mlflow ui --port 5000

# Access at http://localhost:5000
```

**MLflow Features:**
- Experiment tracking and comparison
- Model versioning and registry
- Hyperparameter logging
- Artifact storage (models, plots, etc.)
- Remote tracking server support

---

## Weights & Biases Integration

Track experiments with W&B for advanced visualization.

```python
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
    MetricManager,
)


def main():
    # Data
    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        batch_size=64,
    )

    # Metrics
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Model
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        metrics=metric_manager,
    )

    # Trainer with W&B
    trainer = AutoTrainer(
        max_epochs=20,
        logger=[
            LoggerConfig(
                backend="wandb",
                params={
                    "project": "my-project",
                    "name": "resnet50-cifar10",
                    "entity": "my-team",  # Optional
                    "tags": ["baseline", "resnet"],
                },
            ),
        ],
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**W&B Setup:**

```bash
# Install W&B
pip install wandb

# Login (first time only)
wandb login
```

---

## Detailed Evaluation

Log confusion matrix and detailed metrics with MetricManager.

```python
from autotimm import LoggingConfig, MetricConfig, MetricManager


def main():
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
        MetricConfig(
            name="precision",
            backend="torchmetrics",
            metric_class="Precision",
            params={"task": "multiclass", "average": "macro"},
            stages=["test"],
        ),
        MetricConfig(
            name="recall",
            backend="torchmetrics",
            metric_class="Recall",
            params={"task": "multiclass", "average": "macro"},
            stages=["test"],
        ),
    ]

    # Create MetricManager for programmatic access
    metric_manager = MetricManager(configs=metric_configs, num_classes=10)

    # Access specific metrics
    accuracy = metric_manager.get_metric_by_name("accuracy", stage="val")
    f1_config = metric_manager.get_config_by_name("f1")

    # Iterate over all configs
    for config in metric_manager:
        print(f"{config.name}: stages={config.stages}")

    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        metrics=metric_manager,
        logging_config=LoggingConfig(
            log_learning_rate=True,
            log_gradient_norm=True,
            log_confusion_matrix=True,
        ),
    )


if __name__ == "__main__":
    main()
```

**Common Metrics:**

| Metric | TorchMetrics Class | Use Case |
|--------|-------------------|----------|
| Accuracy | `Accuracy` | Overall correctness |
| F1 Score | `F1Score` | Balanced precision/recall |
| Precision | `Precision` | False positive rate |
| Recall | `Recall` | False negative rate |
| AUROC | `AUROC` | Classification confidence |
| Confusion Matrix | `ConfusionMatrix` | Per-class errors |

**Metric Averaging:**

| Average Mode | Description | Best For |
|--------------|-------------|----------|
| `macro` | Unweighted mean | Balanced datasets |
| `weighted` | Weighted by support | Imbalanced datasets |
| `micro` | Global average | Overall performance |
| `none` | Per-class metrics | Detailed analysis |

---

## Per-Class Metrics

Evaluate performance for each class individually.

```python
from autotimm import MetricConfig, MetricManager


def main():
    metric_configs = [
        # Overall metrics
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val", "test"],
        ),
        # Per-class metrics
        MetricConfig(
            name="per_class_accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass", "average": "none"},
            stages=["test"],
        ),
        MetricConfig(
            name="per_class_f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "none"},
            stages=["test"],
        ),
    ]

    metric_manager = MetricManager(configs=metric_configs, num_classes=10)


if __name__ == "__main__":
    main()
```

---

## Logging Configuration

Control what gets logged during training.

```python
from autotimm import ImageClassifier, LoggingConfig


def main():
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        logging_config=LoggingConfig(
            log_learning_rate=True,      # Log current LR
            log_gradient_norm=True,       # Log gradient norms
            log_confusion_matrix=True,    # Log confusion matrix
            log_train_metrics=True,       # Log metrics during training
            log_val_metrics=True,         # Log metrics during validation
        ),
    )


if __name__ == "__main__":
    main()
```

---

## Running Logging and Metrics Examples

Clone the repository and run examples:

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[all]"

# Run multiple loggers example
python examples/multiple_loggers.py

# Run MLflow tracking example
python examples/mlflow_tracking.py

# Run detailed evaluation example
python examples/detailed_evaluation.py
```
