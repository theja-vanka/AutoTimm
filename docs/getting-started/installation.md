# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Install from PyPI

### Core Installation

```bash
pip install autotimm
```

This installs the core package with these dependencies:

- `torch>=2.0`
- `torchvision>=0.15`
- `timm>=1.0`
- `pytorch-lightning>=2.0`
- `torchmetrics>=1.0`
- `numpy>=1.23`

### Optional Dependencies

#### Albumentations (OpenCV-based transforms)

```bash
pip install autotimm[albumentations]
```

Adds:

- `albumentations>=1.3`
- `opencv-python-headless>=4.8`

#### Logger Backends

```bash
# TensorBoard
pip install autotimm[tensorboard]

# Weights & Biases
pip install autotimm[wandb]

# MLflow
pip install autotimm[mlflow]
```

#### Everything

```bash
pip install autotimm[all]
```

Includes all optional dependencies: tensorboard, mlflow, wandb, and albumentations.

## Development Installation

For contributing or developing locally:

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[dev,all]"
```

This installs:

- All optional dependencies
- Development tools: pytest, pytest-cov, ruff, black

## Verify Installation

```python
import autotimm

# Check available backbones
print(len(autotimm.list_backbones()))  # 1000+ models

# Check version
print(autotimm.__version__)
```

## GPU Support

AutoTimm automatically uses GPU when available. Ensure you have CUDA-compatible PyTorch:

```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is not available, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).

## Troubleshooting

### ImportError: No module named 'timm'

```bash
pip install timm>=1.0
```

### CUDA out of memory

Reduce batch size or use gradient accumulation:

```python
trainer = AutoTrainer(
    max_epochs=10,
    accumulate_grad_batches=4,  # Simulate larger batch
)
```

### Albumentations not found

```bash
pip install autotimm[albumentations]
```
