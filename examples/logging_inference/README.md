# Logging, Inference & Utilities

Experiment tracking, model inference, and utility examples.

## Logging & Tracking (3 examples)

- **`multiple_loggers.py`** - TensorBoard + CSV logging simultaneously
- **`mlflow_tracking.py`** - MLflow experiment tracking
- **`detailed_evaluation.py`** - Confusion matrix and per-class metrics

## Inference (4 examples)

- **`inference.py`** - Model inference and batch prediction
- **`inference_without_metrics.py`** - Production-ready inference
- **`detection_inference.py`** - Object detection inference and visualization
- **`segmentation_inference.py`** - Semantic segmentation inference

## Utilities (1 example)

- **`backbone_discovery.py`** - Explore and compare timm backbones

## Quick Start

```bash
# Experiment tracking with MLflow
python logging_inference/mlflow_tracking.py

# Model inference
python logging_inference/inference.py

# Explore available backbones
python logging_inference/backbone_discovery.py
```

## Logging Backends

- **TensorBoard** - Real-time training visualization
- **CSV** - Simple text-based logging
- **MLflow** - Experiment tracking and model registry
- **Weights & Biases** - Cloud-based experiment tracking

## Inference Features

- Single image and batch prediction
- Visualization of results
- Model export (ONNX, TorchScript)
- Production-optimized code
