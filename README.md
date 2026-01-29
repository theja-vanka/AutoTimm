<p align="center">
  <img src="autotimm.png" alt="AutoTimm" width="400">
</p>

<p align="center">
  <strong>Train state-of-the-art image classifiers with minimal code</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/autotimm/"><img src="https://img.shields.io/pypi/v/autotimm?color=blue&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/autotimm/"><img src="https://img.shields.io/pypi/pyversions/autotimm?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/theja-vanka/AutoTimm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://github.com/theja-vanka/AutoTimm/stargazers"><img src="https://img.shields.io/github/stars/theja-vanka/AutoTimm?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="https://theja-vanka.github.io/AutoTimm/">Documentation</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/getting-started/quickstart/">Quick Start</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/examples/">Examples</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/api/">API Reference</a>
</p>

---

AutoTimm combines the power of [timm](https://github.com/huggingface/pytorch-image-models) (1000+ pretrained models) with [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for a seamless training experience. Go from idea to trained model in minutes, not hours.

## Highlights

| | |
|---|---|
| **1000+ Backbones** | Access ResNet, EfficientNet, ViT, ConvNeXt, Swin, and more from timm |
| **Explicit Metrics** | Configure exactly what you track with MetricManager and torchmetrics |
| **Multi-Logger Support** | TensorBoard, MLflow, Weights & Biases, CSV — use them all at once |
| **Auto-Tuning** | Automatic learning rate and batch size finding before training |
| **Flexible Transforms** | Choose between torchvision (PIL) or albumentations (OpenCV) |
| **Production Ready** | Mixed precision, multi-GPU, gradient accumulation out of the box |

## Installation

```bash
pip install autotimm
```

<details>
<summary><strong>More installation options</strong></summary>

```bash
# With specific extras
pip install autotimm[albumentations]  # OpenCV-based transforms
pip install autotimm[tensorboard]     # TensorBoard logging
pip install autotimm[wandb]           # Weights & Biases
pip install autotimm[mlflow]          # MLflow tracking

# Everything
pip install autotimm[all]

# Development
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[dev,all]"
```

</details>

## Quick Start

```python
from autotimm import AutoTrainer, ImageClassifier, ImageDataModule, MetricConfig, MetricManager


def main():
    # Data
    data = ImageDataModule(data_dir="./data", dataset_name="CIFAR10", image_size=224, batch_size=64)

    # Metrics
    metrics = MetricManager(
        configs=[
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val", "test"],
                prog_bar=True,
            ),
        ],
        num_classes=10,
    )

    # Model & Train
    model = ImageClassifier(backbone="resnet18", num_classes=10, metrics=metrics, lr=1e-3)
    trainer = AutoTrainer(max_epochs=10)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

**[See the full documentation for more examples and features →](https://theja-vanka.github.io/AutoTimm/)**

## Examples

Ready-to-run scripts in the [`examples/`](examples/) directory:

| Example | Description |
|---------|-------------|
| [classify_cifar10.py](examples/classify_cifar10.py) | Basic training with MetricManager and auto-tuning |
| [classify_custom_folder.py](examples/classify_custom_folder.py) | Train on your own dataset |
| [vit_finetuning.py](examples/vit_finetuning.py) | Two-phase Vision Transformer fine-tuning |
| [multi_gpu_training.py](examples/multi_gpu_training.py) | Distributed training with DDP |
| [mlflow_tracking.py](examples/mlflow_tracking.py) | Experiment tracking with MLflow |

**[Browse all examples →](https://theja-vanka.github.io/AutoTimm/examples/)**

## Documentation

| Section | Description |
|---------|-------------|
| [Quick Start](https://theja-vanka.github.io/AutoTimm/getting-started/quickstart/) | Get up and running in 5 minutes |
| [User Guide](https://theja-vanka.github.io/AutoTimm/user-guide/data-loading/) | In-depth guides for all features |
| [API Reference](https://theja-vanka.github.io/AutoTimm/api/) | Complete API documentation |
| [Examples](https://theja-vanka.github.io/AutoTimm/examples/) | Runnable code examples |

## Explore Backbones

```python
import autotimm

# Search 1000+ models
autotimm.list_backbones("*efficientnet*", pretrained_only=True)
autotimm.list_backbones("*vit*")

# Inspect a model
backbone = autotimm.create_backbone("convnext_tiny")
print(f"Features: {backbone.num_features}, Params: {autotimm.count_parameters(backbone):,}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

```bash
# Setup development environment
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[dev,all]"

# Run tests
pytest tests/
```

## Citation

If you use AutoTimm in your research, please cite:

```bibtex
@software{autotimm,
  author = {Theja Vanka},
  title = {AutoTimm: Automated Deep Learning Image Classification},
  url = {https://github.com/theja-vanka/AutoTimm},
  year = {2024}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with <a href="https://github.com/huggingface/pytorch-image-models">timm</a> and <a href="https://github.com/Lightning-AI/pytorch-lightning">PyTorch Lightning</a>
</p>
