# Common Warning Messages

Quick reference for common warning messages.

| Warning | Meaning | Action |
|---------|---------|--------|
| `UserWarning: The dataloader does not have many workers` | Slow data loading | Increase `num_workers` |
| `UserWarning: Trying to infer the batch_size` | Can't determine batch size | Explicitly set in datamodule |
| `UserWarning: The number of training batches is very small` | Epoch finishes quickly | Increase dataset size or reduce batch size |
| `FutureWarning: Passing (type, 1) for ndim` | Deprecated numpy usage | Update to latest version |
| `UserWarning: Mixed precision is not supported on CPU` | Using wrong accelerator | Switch to GPU or remove precision flag |

## Understanding Warnings

### UserWarning

User warnings are informational and usually don't break training:
- **Data loader warnings**: See [Slow Training](../performance/slow-training.md)
- **Batch size warnings**: Usually safe to ignore if training works
- **Mixed precision warnings**: See [Device Errors](../environment/device-errors.md)

### FutureWarning

Future warnings indicate deprecated features:
- Update dependencies to latest versions
- Check AutoTimm documentation for new APIs

### DeprecationWarning

Deprecation warnings suggest updating code:
- Review recent changes in package documentation
- Update to new recommended patterns

## Suppressing Warnings

```python
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", message=".*dataloader.*")

# Suppress all warnings (not recommended)
warnings.filterwarnings("ignore")
```

## Related Pages

- [Error Reference](error-reference.md) - Common errors
- [Data Loading](../data/data-loading.md) - Data loader warnings
- [Device Errors](../environment/device-errors.md) - Hardware warnings
