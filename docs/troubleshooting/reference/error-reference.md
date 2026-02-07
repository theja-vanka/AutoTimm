# Common Error Reference

Quick reference for common error messages and solutions.

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `RuntimeError: CUDA out of memory` | Batch size too large | Reduce batch size, use gradient accumulation |
| `RuntimeError: CUDA error: device-side assert` | Invalid tensor values | Check labels are in valid range |
| `ValueError: Expected input batch_size to match target batch_size` | Mismatched batch dimensions | Check data loader output shapes |
| `RuntimeError: Given groups=1, weight of size [X], expected input[Y]` | Wrong input channels | Check image channels (RGB vs grayscale) |
| `IndexError: index out of range` | Label exceeds num_classes | Verify num_classes matches actual labels |
| `RuntimeError: element 0 of tensors does not require grad` | Frozen model or detached tensor | Check model.train() is called |
| `ValueError: optimizer got an empty parameter list` | No trainable parameters | Check model isn't fully frozen |
| `RuntimeError: Expected all tensors on same device` | Mixed CPU/GPU tensors | Ensure all inputs are on same device |
| `FileNotFoundError: No such file or directory` | Wrong data path | Verify data_dir path exists |
| `KeyError: 'images'` | Wrong COCO annotation format | Check annotation JSON structure |

## Quick Solutions by Error Type

### RuntimeError

Most runtime errors are related to:
- **Memory**: See [OOM Errors](../performance/oom-errors.md)
- **Device**: See [Device Errors](../environment/device-errors.md)
- **Gradients**: See [Gradient Issues](../training/gradient-issues.md)

### ValueError

Value errors typically indicate:
- **Data format issues**: See [Data Loading](../data/data-loading.md)
- **Metric configuration**: See [Metrics](../models/metrics.md)
- **Shape mismatches**: Check data dimensions

### ImportError

Import errors usually mean:
- **Missing dependencies**: See [Installation](../environment/installation.md)
- **Version mismatches**: Check package versions

### KeyError

Key errors commonly occur with:
- **Data loading**: See [Data Loading](../data/data-loading.md)
- **COCO annotations**: Verify annotation format
- **Config dictionaries**: Check parameter spelling

## Related Pages

- [Common Warnings](warnings.md) - Warning messages reference
- All troubleshooting sections for detailed solutions
