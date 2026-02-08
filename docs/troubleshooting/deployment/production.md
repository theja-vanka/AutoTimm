# Production Deployment Issues

Issues deploying models to production environments.

## C++ Deployment

### Undefined Symbol Errors

**Solution:** Match LibTorch ABI version:

```bash
# Check compiler version
g++ --version

# Use cxx11-abi for modern compilers
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
```

### CUDA Error: No Kernel Image Available

**Solution:** Match LibTorch CUDA version:

```bash
# Check CUDA version
nvidia-smi

# Download matching LibTorch (e.g., CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip
```

## iOS Deployment

### Module File Doesn't Exist

**Solution:** Ensure model is in app bundle:
- Check "Copy items if needed" when adding to Xcode
- Verify target membership in Xcode

## Android Deployment

### Native Method Not Found

**Solution:** Update PyTorch Android version:

```gradle
implementation 'org.pytorch:pytorch_android:1.13.1'
implementation 'org.pytorch:pytorch_android_torchvision:1.13.1'
```

## ONNX Deployment

### Export Fails

**Solutions:**

```python
# 1. Use lower opset version
torch.onnx.export(..., opset_version=11)

# 2. Simplify model (remove custom ops)

# 3. Use TorchScript instead
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.pt")
```

## Model Size Optimization

### Exported Model Too Large

**Solutions:**

```python
# 1. Use quantization
from autotimm.export import quantize_dynamic
quantize_dynamic("model.onnx", "model_quant.onnx")

# 2. Use smaller backbone
model = ImageClassifier(backbone="resnet34")  # Instead of resnet50

# 3. Use FP16 during export
```

## Related Issues

- [Export & Inference](export-inference.md) - Export problems
- [Device Errors](../environment/device-errors.md) - Hardware issues
