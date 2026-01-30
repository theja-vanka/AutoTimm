# Examples

The [`examples/`](https://github.com/theja-vanka/AutoTimm/tree/main/examples) directory contains runnable scripts demonstrating AutoTimm features.

## Quick Reference

Browse examples by topic:

- **[Classification](classification.md)** - Image classification with CIFAR-10, custom datasets, and ViT fine-tuning
- **[Object Detection](object-detection.md)** - FCOS, transformer-based detection, and RT-DETR
- **[Data Handling](data-handling.md)** - Balanced sampling, augmentation, and custom transforms
- **[Training Utilities](training-utilities.md)** - Auto-tuning, multi-GPU training, and inference
- **[Logging & Metrics](logging-metrics.md)** - Multiple loggers, MLflow, and detailed evaluation
- **[Backbone Utilities](backbone-utilities.md)** - Discovering and comparing timm backbones

## All Examples

| Script | Description | Category |
|--------|-------------|----------|
| [`classify_cifar10.py`](classification.md#cifar-10-classification) | ResNet-18 on CIFAR-10 with MetricManager | Classification |
| [`classify_custom_folder.py`](classification.md#custom-folder-dataset) | EfficientNet on a custom folder dataset with W&B | Classification |
| [`vit_finetuning.py`](classification.md#vit-fine-tuning) | Two-phase ViT fine-tuning | Classification |
| [`object_detection_coco.py`](object-detection.md#object-detection-on-coco) | FCOS-style object detection on COCO dataset | Object Detection |
| [`object_detection_transformers.py`](object-detection.md#transformer-based-object-detection) | Vision Transformers for object detection | Object Detection |
| [`object_detection_rtdetr.py`](object-detection.md#rt-detr-real-time-detection-transformer) | RT-DETR end-to-end transformer detection | Object Detection |
| [`balanced_sampling.py`](data-handling.md#balanced-sampling) | Weighted sampling for imbalanced data | Data Handling |
| [`albumentations_cifar10.py`](data-handling.md#albumentations-strong-augmentation) | Albumentations strong augmentation | Data Handling |
| [`albumentations_custom_folder.py`](data-handling.md#custom-albumentations-pipeline) | Custom albumentations pipeline | Data Handling |
| [`auto_tuning.py`](training-utilities.md#auto-tuning) | Automatic LR and batch size finding | Training Utilities |
| [`multi_gpu_training.py`](training-utilities.md#multi-gpu-training) | Multi-GPU and distributed training | Training Utilities |
| [`inference.py`](training-utilities.md#inference) | Model inference and batch prediction | Training Utilities |
| [`multiple_loggers.py`](logging-metrics.md#multiple-loggers) | TensorBoard + CSV logging simultaneously | Logging & Metrics |
| [`mlflow_tracking.py`](logging-metrics.md#mlflow-tracking) | MLflow experiment tracking | Logging & Metrics |
| [`detailed_evaluation.py`](logging-metrics.md#detailed-evaluation) | Confusion matrix and per-class metrics | Logging & Metrics |
| [`backbone_discovery.py`](backbone-utilities.md#backbone-discovery) | Explore timm backbones | Backbone Utilities |

---

## Getting Started

To run any example:

```bash
# Clone the repository
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm

# Install AutoTimm with all dependencies
pip install -e ".[all]"

# Run an example
python examples/classify_cifar10.py
```

## Example Categories

### Classification
Learn how to train image classification models on standard and custom datasets. Includes examples for basic training, transfer learning, and Vision Transformer fine-tuning.

👉 [View Classification Examples](classification.md)

### Object Detection
Explore object detection with FCOS architecture, Vision Transformer backbones, and RT-DETR. Covers COCO format datasets and various detection strategies.

👉 [View Object Detection Examples](object-detection.md)

### Data Handling
Master data loading techniques including balanced sampling for imbalanced datasets, albumentations augmentation presets, and custom transform pipelines.

👉 [View Data Handling Examples](data-handling.md)

### Training Utilities
Optimize training with auto-tuning for learning rates and batch sizes, multi-GPU distributed training, and efficient inference patterns.

👉 [View Training Utilities Examples](training-utilities.md)

### Logging & Metrics
Track experiments effectively with multiple logger support, MLflow integration, and comprehensive metric evaluation including per-class analysis.

👉 [View Logging & Metrics Examples](logging-metrics.md)

### Backbone Utilities
Discover and compare timm backbones for your use case. Learn about different model families and how to select the right backbone.

👉 [View Backbone Utilities Examples](backbone-utilities.md)



