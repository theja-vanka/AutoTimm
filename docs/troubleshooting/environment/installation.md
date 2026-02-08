# Installation & Dependencies

Installation and dependency issues.

## ImportError: No module named 'timm'

**Solution:**

```bash
pip install timm>=1.0
```

## Albumentations not found

**Solution:**

```bash
pip install --upgrade autotimm  # Included by default
# Or if still missing:
pip install albumentations>=1.3
```

## Version Compatibility

```bash
# Check installed versions
pip list | grep torch
pip list | grep timm
pip list | grep autotimm

# Upgrade to compatible versions
pip install --upgrade torch torchvision timm autotimm
```

## CUDA Installation

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with matching CUDA version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Related Issues

- [Device Errors](device-errors.md) - CUDA runtime issues
- [Model Loading](../models/model-loading.md) - Version mismatches
