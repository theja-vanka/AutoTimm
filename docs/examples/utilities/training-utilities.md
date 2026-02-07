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

## Preset Manager

Manage and reuse training configurations with preset templates.

```python
from autotimm import PresetManager, AutoTrainer, ImageClassifier
from autotimm.data import ImageDataModule


def main():
    # Create a preset manager
    preset_manager = PresetManager()
    
    # Save current configuration as preset
    preset_manager.save_preset(
        name="resnet18_baseline",
        model_configs={
            "backbone": "resnet18",
            "num_classes": 10,
            "lr": 1e-3,
            "optimizer": "adamw",
        },
        trainer_configs={
            "max_epochs": 50,
            "precision": "16-mixed",
        },
        data_configs={
            "batch_size": 32,
            "image_size": 224,
        }
    )
    
    # Load and use a preset
    configs = preset_manager.load_preset("resnet18_baseline")
    
    # Create model and datamodule from preset
    model = ImageClassifier(**configs["model_configs"])
    data = ImageDataModule(**configs["data_configs"], data_dir="./data")
    trainer = AutoTrainer(**configs["trainer_configs"])
    
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Preset Manager Features:**

- **Save configurations**: Store successful training setups
- **Reuse presets**: Quickly apply proven configurations
- **Share presets**: Export/import configurations across projects
- **Version control**: Track configuration changes over time

---

## Performance Optimization

Optimize training performance with various techniques.

```python
from autotimm import AutoTrainer, ImageClassifier, ImageDataModule
import torch


def main():
    # Enable performance optimizations
    torch.set_float32_matmul_precision('high')  # Use TensorFloat-32
    
    # Data optimizations
    data = ImageDataModule(
        data_dir="./data",
        batch_size=64,
        num_workers=8,  # Parallel data loading
        pin_memory=True,  # Fast GPU transfer
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2,  # Prefetch batches
    )
    
    # Model with optimizations
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=10,
        channels_last=True,  # Memory format optimization
        compile_model=True,  # torch.compile (PyTorch 2.0+)
    )
    
    # Trainer with performance settings
    trainer = AutoTrainer(
        max_epochs=50,
        precision="bf16-mixed",  # BFloat16 for speed
        accelerator="gpu",
        devices=1,
        benchmark=True,  # cuDNN autotuner
        deterministic=False,  # Allow non-deterministic ops
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Gradient accumulation
    )
    
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**Performance Optimization Techniques:**

| Technique | Speed Gain | Trade-off |
|-----------|------------|-----------|
| Mixed precision (bf16) | 2-3x | Minimal accuracy impact |
| Channels last | 10-20% | None |
| torch.compile | 20-40% | Longer startup time |
| Persistent workers | 5-15% | More memory |
| Gradient accumulation | Enable larger batch | Slower updates |
| cuDNN benchmark | 5-10% | Non-deterministic |

**Best Practices:**

1. **Start simple**: Enable one optimization at a time
2. **Profile first**: Use PyTorch Profiler to identify bottlenecks
3. **Monitor accuracy**: Ensure optimizations don't hurt model quality
4. **Test thoroughly**: Some optimizations are hardware-specific

---

## Running Examples

```bash
python examples/data_training/auto_tuning.py
python examples/data_training/multi_gpu_training.py
python examples/data_training/preset_manager.py
python examples/data_training/performance_optimization_demo.py
```

**See Also:**

- [Training User Guide](../../user-guide/training/training.md) - Full training documentation
- [Inference Guide](../../user-guide/inference/index.md) - Model inference and deployment
