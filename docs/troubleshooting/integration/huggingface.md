# HuggingFace Integration Issues

Problems using HuggingFace Hub models.

## HuggingFace Hub Issues

### Model Not Found on Hub

**Solution:** Verify model name exists:

```python
from autotimm import list_hf_hub_backbones

# Search for model
models = list_hf_hub_backbones(model_name="resnet", limit=10)
print(models)
```

### Model Download is Slow

**Explanation:** Models are cached after first download. Subsequent runs are fast.

**Location:** Models cached in `~/.cache/huggingface/hub/`

### Checkpoint Loading Fails with HF Models

**Solution:** Must pass the same `backbone` argument:

```python
# Save
model = ImageClassifier(backbone="hf-hub:timm/resnet50.a1_in1k", ...)

# Load - must match
loaded = ImageClassifier.load_from_checkpoint(
    path,
    backbone="hf-hub:timm/resnet50.a1_in1k",  # Must match original
    metrics=metrics,
)
```

## HuggingFace Transformers Issues

### Model Expects 'pixel_values' Keyword Argument

**Problem:** HF Transformers models need specific input format

**Solution:**

```python
# ✗ Wrong
output = model(x)

# ✓ Correct
output = model(pixel_values=x)
```

### RuntimeError about Trainer Attachment

**Problem:** Calling `configure_optimizers()` without a trainer

**Solutions:**

```python
# Option 1: Attach model to trainer first
trainer.fit(model, datamodule=data)

# Option 2: Use scheduler=None for inference
model = ImageClassifier(
    backbone="hf-hub:timm/resnet50.a1_in1k",
    scheduler=None,  # No scheduler
)
```

## HuggingFace Hub Push Issues

```python
from huggingface_hub import login

# Login to HuggingFace
login(token="your_token")

# Push model with retry
model.push_to_hub(
    repo_id="username/model-name",
    commit_message="Initial commit",
    private=True,  # Make repository private
)

# If push fails, check permissions
# https://huggingface.co/settings/tokens
```

## Related Issues

- [Model Loading](../models/model-loading.md) - Checkpoint issues
- [Installation](../environment/installation.md) - HuggingFace dependencies
