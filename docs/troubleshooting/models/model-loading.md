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

# Note: certain params are NOT saved in checkpoints (save_hyperparameters ignores them).
# You must re-supply these when loading:
#   ImageClassifier:   metrics, logging_config, transform_config, loss_fn
#   ObjectDetector:    metrics, logging_config, transform_config, cls_loss_fn, reg_loss_fn
#   SemanticSegmentor: metrics, logging_config, transform_config, class_weights, loss_fn
#   InstanceSegmentor: metrics, logging_config, transform_config, cls_loss_fn, reg_loss_fn, mask_loss_fn
#   YOLOXDetector:     metrics, logging_config, transform_config
#
# For inference/export, also pass compile_model=False to skip unnecessary compilation:
model = ImageClassifier.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    compile_model=False,  # skip compilation for inference
    metrics=metrics,      # not saved in checkpoint
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
