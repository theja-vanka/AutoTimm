# Deployment Examples

Export trained AutoTimm models for production deployment.

## Scripts

| Script | Description |
|--------|-------------|
| `export_to_onnx.py` | Export models to ONNX format for cross-platform deployment (ONNX Runtime, TensorRT, OpenVINO, CoreML) |
| `export_to_torchscript.py` | Export models to TorchScript format for PyTorch ecosystem deployment (LibTorch, C++, mobile) |
| `deploy_torchscript_cpp.py` | C++ deployment with LibTorch |

## Quick Start

```bash
# Install ONNX dependencies
pip install onnx onnxruntime onnxscript

# Run ONNX export examples
python examples/deployment/export_to_onnx.py

# Run TorchScript export examples
python examples/deployment/export_to_torchscript.py
```

## See Also

- [ONNX Export Guide](https://theja-vanka.github.io/AutoTimm/user-guide/deployment/onnx-export/)
- [TorchScript Export Guide](https://theja-vanka.github.io/AutoTimm/user-guide/deployment/torchscript-export/)
- [Production Deployment Guide](https://theja-vanka.github.io/AutoTimm/user-guide/deployment/deployment/)
