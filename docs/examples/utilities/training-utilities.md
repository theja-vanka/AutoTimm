# Training Utilities Examples

This page demonstrates training optimization utilities including auto-tuning and multi-GPU training.

## Auto-Tuning

Automatically find optimal learning rate and batch size.

```python
from autotimm import AutoTrainer, TunerConfig


def main():
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


if __name__ == "__main__":
    main()
```

**TunerConfig Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `auto_lr` | Enable automatic learning rate finding | `False` |
| `auto_batch_size` | Enable automatic batch size finding | `False` |
| `lr_find_kwargs` | Arguments for LR finder | `{"min_lr": 1e-6, "max_lr": 1.0}` |
| `batch_size_kwargs` | Arguments for batch size finder | `{"mode": "power"}` |

---

## Multi-GPU Training

Distributed training across multiple GPUs.

```python
from autotimm import AutoTrainer


def main():
    trainer = AutoTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision="bf16-mixed",
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Multi-GPU Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `ddp` | Distributed Data Parallel | Most use cases |
| `ddp_spawn` | DDP with process spawning | Debugging |
| `fsdp` | Fully Sharded Data Parallel | Very large models |

**Precision Options:**

| Precision | Speed | Memory |
|-----------|-------|--------|
| `32` | Slowest | Highest |
| `16-mixed` | Faster | Lower |
| `bf16-mixed` | Faster | Lower |

---

## Running Examples

```bash
python examples/auto_tuning.py
python examples/multi_gpu_training.py
```

**See Also:**

- [Training User Guide](../../user-guide/training/training.md) - Full training documentation
- [Inference Guide](../../user-guide/inference/index.md) - Model inference and deployment
