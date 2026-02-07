# Model Loading and Checkpoint Issues

Problems loading models and checkpoints.

## Checkpoint Compatibility

```python
import torch

# Check checkpoint contents
checkpoint = torch.load("path/to/checkpoint.ckpt")
print("Checkpoint keys:", checkpoint.keys())
print("State dict keys:", checkpoint["state_dict"].keys() if "state_dict" in checkpoint else "No state_dict")

# Load with strict=False to ignore mismatched keys
model = ImageClassifier.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    strict=False,  # Ignore missing/unexpected keys
)
```

## Pretrained Model Download Failures

```python
# If download fails, manually specify cache directory
import os
os.environ["TORCH_HOME"] = "/path/to/cache"

# Or disable pretrained weights
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    pretrained=False,  # Train from scratch
)
```

## Version Mismatch Errors

```bash
# Check installed versions
pip list | grep torch
pip list | grep timm
pip list | grep autotimm

# Upgrade to compatible versions
pip install --upgrade torch torchvision timm
```

## Related Issues

- [Installation](../environment/installation.md) - Dependencies and versions
- [HuggingFace](../integration/huggingface.md) - HF Hub model loading
