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

```python
from autotimm import AutoTrainer, LoggerConfig

trainer = AutoTrainer(
    max_epochs=20,
    logger=[
        LoggerConfig(
            backend="wandb",
            params={
                "project": "my-project",
                "name": "resnet50-cifar10",
                "tags": ["baseline", "resnet"],
            },
        ),
    ],
)
```

---

## Detailed Evaluation

Configure metrics with MetricManager:

```python
from autotimm import ImageClassifier, LoggingConfig, MetricConfig, MetricManager

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
]

metric_manager = MetricManager(configs=metric_configs, num_classes=10)

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
```

---

## Running Examples

```bash
python examples/logging_inference/multiple_loggers.py
python examples/logging_inference/mlflow_tracking.py
python examples/logging_inference/detailed_evaluation.py
```

**See Also:**

- [Logging User Guide](../../user-guide/guides/logging.md) - Full logging documentation
- [Metrics User Guide](../../user-guide/evaluation/metrics.md) - Detailed metrics configuration
