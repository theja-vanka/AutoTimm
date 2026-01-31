# Examples

The [`examples/`](https://github.com/theja-vanka/AutoTimm/tree/main/examples) directory contains runnable scripts demonstrating AutoTimm features.

## Quick Reference

Browse examples by topic:

- **[Classification](tasks/classification.md)** - Image classification with CIFAR-10, custom datasets, and ViT fine-tuning
- **[Object Detection](tasks/object-detection.md)** - FCOS, transformer-based detection, and RT-DETR
- **[Semantic Segmentation](tasks/semantic-segmentation.md)** - DeepLabV3+ and FCN architectures for semantic segmentation
- **[Instance Segmentation](tasks/instance-segmentation.md)** - Mask R-CNN style instance segmentation
- **[Hugging Face Hub](integration/huggingface-hub.md)** - Using HF Hub models for all tasks with comprehensive examples
- **[Data Handling](utilities/data-handling.md)** - Balanced sampling, augmentation, and custom transforms
- **[Training Utilities](utilities/training-utilities.md)** - Auto-tuning, multi-GPU training, and inference
- **[Logging & Metrics](utilities/logging-metrics.md)** - Multiple loggers, MLflow, and detailed evaluation
- **[Backbone Utilities](utilities/backbone-utilities.md)** - Discovering and comparing timm backbones

## All Examples

| Script | Description | Category |
|--------|-------------|----------|
| [`classify_cifar10.py`](tasks/classification.md#cifar-10-classification) | ResNet-18 on CIFAR-10 with MetricManager | Classification |
| [`classify_custom_folder.py`](tasks/classification.md#custom-folder-dataset) | Train on custom ImageFolder dataset with W&B | Classification |
| [`vit_finetuning.py`](tasks/classification.md#vit-fine-tuning) | Two-phase ViT fine-tuning | Classification |
| [`semantic_segmentation.py`](tasks/semantic-segmentation.md#basic-example-cityscapes) | DeepLabV3+ semantic segmentation | Semantic Segmentation |
| [`instance_segmentation.py`](tasks/instance-segmentation.md#basic-example-coco) | Mask R-CNN style instance segmentation | Instance Segmentation |
| [`object_detection_coco.py`](tasks/object-detection.md#object-detection-on-coco) | FCOS-style object detection on COCO dataset | Object Detection |
| [`object_detection_transformers.py`](tasks/object-detection.md#transformer-based-object-detection) | Vision Transformers for object detection | Object Detection |
| [`object_detection_rtdetr.py`](tasks/object-detection.md#rt-detr-real-time-detection-transformer) | RT-DETR end-to-end transformer detection | Object Detection |
| [`balanced_sampling.py`](utilities/data-handling.md#balanced-sampling) | Weighted sampling for imbalanced data | Data Handling |
| [`albumentations_cifar10.py`](utilities/data-handling.md#albumentations-strong-augmentation) | Albumentations strong augmentation | Data Handling |
| [`albumentations_custom_folder.py`](utilities/data-handling.md#custom-albumentations-pipeline) | Custom albumentations pipeline | Data Handling |
| [`auto_tuning.py`](utilities/training-utilities.md#auto-tuning) | Automatic LR and batch size finding | Training Utilities |
| [`multi_gpu_training.py`](utilities/training-utilities.md#multi-gpu-training) | Multi-GPU and distributed training | Training Utilities |
| [`inference.py`](utilities/training-utilities.md#inference) | Model inference and batch prediction | Training Utilities |
| [`multiple_loggers.py`](utilities/logging-metrics.md#multiple-loggers) | TensorBoard + CSV logging simultaneously | Logging & Metrics |
| [`mlflow_tracking.py`](utilities/logging-metrics.md#mlflow-tracking) | MLflow experiment tracking | Logging & Metrics |
| [`detailed_evaluation.py`](utilities/logging-metrics.md#detailed-evaluation) | Confusion matrix and per-class metrics | Logging & Metrics |
| [`backbone_discovery.py`](utilities/backbone-utilities.md#backbone-discovery) | Explore timm backbones | Backbone Utilities |
| [`huggingface_hub_models.py`](integration/huggingface-hub.md#1-introduction-to-hf-hub-models) | Introduction to HF Hub integration | HF Hub |
| [`hf_hub_classification.py`](integration/huggingface-hub.md#2-image-classification) | Classification with HF Hub models | HF Hub |
| [`hf_hub_segmentation.py`](integration/huggingface-hub.md#3-semantic-segmentation) | Segmentation with HF Hub models | HF Hub |
| [`hf_hub_object_detection.py`](integration/huggingface-hub.md#4-object-detection) | Detection with HF Hub models | HF Hub |
| [`hf_hub_instance_segmentation.py`](integration/huggingface-hub.md#5-instance-segmentation) | Instance segmentation with HF Hub models | HF Hub |
| [`hf_hub_advanced.py`](integration/huggingface-hub.md#6-advanced-usage) | Advanced HF Hub features | HF Hub |
| [`hf_hub_lightning_integration.py`](../user-guide/integration/huggingface-hub-integration.md#pytorch-lightning-compatibility) | Lightning compatibility with HF Hub | HF Hub |
| [`hf_direct_models_lightning.py`](../user-guide/integration/huggingface-transformers-integration.md) | Direct HF Transformers models with Lightning | HF Transformers |

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

👉 [View Classification Examples](tasks/classification.md)

### Object Detection
Explore object detection with FCOS architecture, Vision Transformer backbones, and RT-DETR. Covers COCO format datasets and various detection strategies.

👉 [View Object Detection Examples](tasks/object-detection.md)

### Semantic Segmentation
Learn semantic segmentation with DeepLabV3+ and FCN architectures. Covers multiple dataset formats (PNG, Cityscapes, COCO, VOC) and various loss functions.

👉 [View Semantic Segmentation Examples](tasks/semantic-segmentation.md)

### Instance Segmentation
Master instance segmentation with Mask R-CNN style architecture. Includes COCO format datasets, detection + mask prediction, and comprehensive evaluation metrics.

👉 [View Instance Segmentation Examples](tasks/instance-segmentation.md)

### Data Handling
Master data loading techniques including balanced sampling for imbalanced datasets, albumentations augmentation presets, and custom transform pipelines.

👉 [View Data Handling Examples](utilities/data-handling.md)

### Training Utilities
Optimize training with auto-tuning for learning rates and batch sizes, multi-GPU distributed training, and efficient inference patterns.

👉 [View Training Utilities Examples](utilities/training-utilities.md)

### Logging & Metrics
Track experiments effectively with multiple logger support, MLflow integration, and comprehensive metric evaluation including per-class analysis.

👉 [View Logging & Metrics Examples](utilities/logging-metrics.md)

### Backbone Utilities
Discover and compare timm backbones, optimizers, and schedulers for your use case. Learn about different model families and how to select the right backbone, optimizer, and learning rate schedule.

👉 [View Backbone Utilities Examples](utilities/backbone-utilities.md)

### Hugging Face Hub
Learn how to use thousands of timm-compatible models from Hugging Face Hub with AutoTimm. Covers model discovery, all tasks (classification, detection, segmentation), PyTorch Lightning compatibility, and advanced features.

👉 [View Hugging Face Hub Examples](integration/huggingface-hub.md)

### HuggingFace Transformers
Use HuggingFace Transformers vision models (ViT, DeiT, BEiT, Swin) directly with PyTorch Lightning. Learn how to use specific model classes for full control without Auto classes.

👉 [View HuggingFace Transformers Integration](../user-guide/integration/huggingface-transformers-integration.md)



