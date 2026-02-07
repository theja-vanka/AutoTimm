# Hyperparameter Tuning with Optuna

Automate hyperparameter optimization using Optuna for finding optimal training configurations.

## Overview

Learn how to use Optuna for automated hyperparameter search, multi-objective optimization, and architecture selection with HuggingFace Hub models.

## What This Example Covers

- **Basic Optuna search** - Optimize learning rate, weight decay, batch size
- **Multi-objective optimization** - Accuracy + inference speed
- **Pruning** - Stop unpromising trials early
- **Architecture search** - Find the best model family
- **Visualization** - Analyze optimization results
- **Best practices** - Production-ready HPO pipelines

## Prerequisites

```bash
pip install optuna optuna-dashboard
```

## Basic Hyperparameter Search

```python
import optuna
from optuna.trial import Trial
from autotimm import ImageClassifier, ImageDataModule, AutoTrainer

def objective(trial: Trial) -> float:
    """Objective function to maximize validation accuracy."""

    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    backbone = trial.suggest_categorical("backbone", [
        "hf-hub:timm/resnet18.a1_in1k",
        "hf-hub:timm/resnet34.a1_in1k",
        "hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
    ])

    # Create model and data
    model = ImageClassifier(backbone=backbone, num_classes=10, lr=lr, weight_decay=weight_decay)
    data = ImageDataModule(data_dir="./data", dataset_name="CIFAR10", batch_size=batch_size)

    # Quick training
    trainer = AutoTrainer(max_epochs=3, enable_checkpointing=False, logger=False)
    trainer.fit(model, datamodule=data)

    # Return validation accuracy
    return trainer.callback_metrics["val_accuracy"].item()

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

## Multi-Objective Optimization

```python
def multi_objective(trial: Trial) -> tuple[float, float]:
    """Optimize both accuracy and inference speed."""

    backbone = trial.suggest_categorical("backbone", [
        "hf-hub:timm/resnet18.a1_in1k",
        "hf-hub:timm/efficientnet_b0.ra_in1k",
        "hf-hub:timm/mobilenetv3_small_100.lamb_in1k",
    ])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = ImageClassifier(backbone=backbone, num_classes=10, lr=lr)

    # Measure inference time
    dummy_input = torch.randn(1, 3, 224, 224)
    start = time.time()
    for _ in range(50):
        with torch.inference_mode():
            _ = model(dummy_input)
    inference_time = (time.time() - start) / 50 * 1000  # ms

    # Train and get accuracy
    # ... (training code)
    accuracy = 0.85  # placeholder

    # Return both objectives (maximize both by returning -time)
    return accuracy, -inference_time

# Multi-objective study
study = optuna.create_study(
    directions=["maximize", "maximize"],
    study_name="accuracy_speed_tradeoff",
)
study.optimize(multi_objective, n_trials=50)

# Analyze Pareto front
print(f"Pareto-optimal solutions: {len(study.best_trials)}")
for trial in study.best_trials:
    print(f"Accuracy: {trial.values[0]:.4f}, Time: {-trial.values[1]:.2f}ms")
```

## Pruning Unpromising Trials

```python
def objective_with_pruning(trial: Trial) -> float:
    """Stop unpromising trials early to save compute."""

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model = ImageClassifier(backbone="hf-hub:timm/resnet18.a1_in1k", num_classes=10, lr=lr)

    for epoch in range(10):
        # Train one epoch
        val_acc = train_one_epoch(model, data)

        # Report intermediate value
        trial.report(val_acc, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc

# Use MedianPruner
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2,
    ),
)
study.optimize(objective_with_pruning, n_trials=50)

# Compute savings
pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
print(f"Pruned {pruned} trials, saving ~{pruned * 60}% compute")
```

## Architecture-Specific Search Spaces

```python
def architecture_aware_objective(trial: Trial) -> float:
    """Different hyperparameters for CNNs vs ViTs."""

    arch_family = trial.suggest_categorical("arch_family", ["cnn", "vit"])

    if arch_family == "cnn":
        backbone = trial.suggest_categorical("backbone", [
            "hf-hub:timm/resnet34.a1_in1k",
            "hf-hub:timm/efficientnet_b1.ra_in1k",
        ])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)  # Higher LR
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Larger batches

    else:  # ViT
        backbone = trial.suggest_categorical("backbone", [
            "hf-hub:timm/vit_small_patch16_224.augreg_in1k",
        ])
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)  # Lower LR!
        batch_size = trial.suggest_categorical("batch_size", [16, 32])  # Smaller batches

    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = ImageClassifier(backbone=backbone, num_classes=10, lr=lr, weight_decay=weight_decay)
    # ... training code
    return val_accuracy
```

## Visualization

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
)

# After running study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 1. Optimization history
fig = plot_optimization_history(study)
fig.write_html("optimization_history.html")

# 2. Parameter importances
fig = plot_param_importances(study)
fig.write_html("param_importances.html")

# 3. Parallel coordinate plot
fig = plot_parallel_coordinate(study)
fig.write_html("parallel_coordinate.html")

# 4. Contour plot
fig = plot_contour(study, params=["lr", "weight_decay"])
fig.write_html("contour_plot.html")

# Optuna Dashboard (optional)
# Run: optuna-dashboard sqlite:///optuna.db
```

## Run the Example

```bash
python examples/data_training/hf_hyperparameter_tuning.py
```

## Search Space Guidelines

### CNN Models (ResNet, EfficientNet)
```python
{
    "lr": "1e-4 to 3e-3 (log scale)",
    "weight_decay": "1e-4 to 1e-2",
    "batch_size": "[32, 64, 128]",
    "dropout": "0.0 to 0.3",
    "optimizer": "['SGD', 'AdamW']",
}
```

### Vision Transformers
```python
{
    "lr": "1e-5 to 5e-4 (log scale, lower!)",
    "weight_decay": "0.01 to 0.1 (higher!)",
    "batch_size": "[16, 32] (smaller)",
    "warmup_epochs": "5 to 20",
    "optimizer": "['AdamW']",  # Don't use SGD
}
```

## Parallel Execution

```python
# Shared study across multiple workers/GPUs
study = optuna.create_study(
    study_name="distributed_hpo",
    storage="postgresql://user:pass@localhost/db",  # Shared database
    load_if_exists=True,
    direction="maximize",
)

# Each worker runs this simultaneously
study.optimize(objective, n_trials=50)
```

**Supported storage**:

- SQLite: `sqlite:///optuna.db`
- PostgreSQL: `postgresql://...`
- MySQL: `mysql://...`

## Computational Budget

### By Dataset Size
- **Small (<10k images)**: 50-100 trials
- **Medium (10k-100k)**: 30-50 trials
- **Large (>100k)**: 20-30 trials (use pruning!)

### Training Strategy
- Use **short epochs** for trials (3-5 epochs)
- Train **best model longer** afterward (full epochs)
- Use **validation set** for optimization
- **Test set** only for final evaluation

## Best Practices

### 1. Start Simple
- Optimize lr and weight_decay first
- Add more parameters gradually
- Don't optimize >7 parameters at once

### 2. Use Log Scale
- Always use log scale for learning rates
- Use log scale for weight decay
- Linear scale for batch size

### 3. Leverage Pruning
- MedianPruner for most cases
- HyperbandPruner for aggressive pruning
- Save 30-50% compute typically

### 4. Visualize Results
- Always plot optimization history
- Check parameter importances
- Use parallel coordinate for insights

### 5. Save and Resume
```python
import joblib

# Save study
joblib.dump(study, "study.pkl")

# Resume later
study = joblib.load("study.pkl")
study.optimize(objective, n_trials=50)  # Continue
```

## What to Optimize (Priority Order)

1. **Priority 1** (always): Learning rate, weight decay
2. **Priority 2** (recommended): Batch size, optimizer
3. **Priority 3** (if time allows): Architecture, dropout
4. **Priority 4** (advanced): Augmentation, scheduler params

## When to Use HPO

- **New dataset/domain**: Always!
- **Production deployment**: Optimize for speed too
- **Research**: Document search space and trials
- **Proof-of-concept**: Use defaults, optimize later

## Common Pitfalls

- **Too many parameters**: Stick to 5-7 max
- **Too few trials**: Need 30+ for reliable results
- **Not using pruning**: Wastes 30-50% compute
- **Wrong search space**: Use log scale for LR
- **Overfitting to validation**: Use separate test set

## Related Examples

- [HuggingFace Hub Models](../integration/huggingface-hub.md)
- [Transfer Learning](../integration/hf_transfer_learning.md)
- [Custom Data](hf_custom_data.md)

## See Also

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Training Guide](../../user-guide/training/training.md)
- [Advanced Customization](../../user-guide/training/advanced-customization.md)
