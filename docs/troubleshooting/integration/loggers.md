# Logger Integration Issues

Problems with TensorBoard, WandB, and other logging backends.

## Weights & Biases (WandB) Issues

### Login Issues

```python
# 1. Login issues
import wandb
wandb.login(key="your_api_key")

# 2. Disable online sync for offline training
trainer = AutoTrainer(
    max_epochs=10,
    logger="wandb",
    logger_kwargs={
        "project": "my-project",
        "offline": True,  # Save logs locally
    },
)

# 3. Resume run
trainer = AutoTrainer(
    logger_kwargs={
        "project": "my-project",
        "id": "run_id",
        "resume": "must",
    },
)
```

## TensorBoard Issues

```python
from autotimm import LoggingConfig

# Specify custom log directory
logging_config = LoggingConfig(
    log_dir="./custom_logs",
    log_hyperparameters=True,
)

# View logs
# tensorboard --logdir ./custom_logs

# If port is occupied
# tensorboard --logdir ./custom_logs --port 6007
```

## Related Issues

- [Installation](../environment/installation.md) - Missing logger dependencies
- [Reproducibility](reproducibility.md) - Logging reproducibility info
