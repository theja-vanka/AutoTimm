# Examples

The [`examples/`](https://github.com/theja-vanka/AutoTimm/tree/main/examples) directory contains runnable scripts demonstrating AutoTimm features.

## Quick Reference

### 🚀 Getting Started
- **[Classification](tasks/classification.md)** - Start with CIFAR-10 and custom datasets
- **[HuggingFace Hub Basics](integration/huggingface-hub.md)** - Load and use HF Hub models

### 🎯 Computer Vision Tasks
- **[Object Detection](tasks/object-detection.md)** - FCOS, YOLOX, RT-DETR, and transformers
- **[Semantic Segmentation](tasks/semantic-segmentation.md)** - DeepLabV3+ and FCN
- **[Instance Segmentation](tasks/instance-segmentation.md)** - Mask R-CNN style

### 🤗 HuggingFace Advanced
- **[Model Interpretation](integration/hf_interpretation.md)** - GradCAM, attention visualization, metrics
- **[Transfer Learning](integration/hf_transfer_learning.md)** - LLRD, progressive unfreezing
- **[Ensemble & Distillation](integration/hf_ensemble.md)** - Model ensembles and knowledge distillation
- **[Deployment](integration/hf_deployment.md)** - ONNX, quantization, serving
- **[Custom Data](utilities/hf_custom_data.md)** - Advanced augmentation and data handling
- **[Hyperparameter Tuning](utilities/hf_hyperparameter_tuning.md)** - Optuna integration

### 📊 Training & Optimization
- **[Training Utilities](utilities/training-utilities.md)** - Auto-tuning, multi-GPU, presets
- **[Data Handling](utilities/data-handling.md)** - Balanced sampling, augmentation
- **[Logging & Metrics](utilities/logging-metrics.md)** - TensorBoard, MLflow, evaluation

### 🔍 Model Understanding
- **[Interpretation Methods](utilities/interpretation.md)** - Comprehensive interpretation toolkit
- **[Backbone Utilities](utilities/backbone-utilities.md)** - Discover and compare models

## All Examples by Category

### 🚀 Getting Started

| Script | Description |
|--------|-------------|
| [`getting_started/classify_cifar10.py`](tasks/classification.md#cifar-10-classification) | ResNet-18 on CIFAR-10 with MetricManager |
| [`getting_started/classify_custom_folder.py`](tasks/classification.md#custom-folder-dataset) | Train on custom ImageFolder dataset with W&B |
| [`getting_started/vit_finetuning.py`](tasks/classification.md#vit-fine-tuning) | Two-phase ViT fine-tuning |

### 🎯 Computer Vision Tasks

**Object Detection:**

| Script | Description |
|--------|-------------|
| [`computer_vision/object_detection_coco.py`](tasks/object-detection.md#object-detection-on-coco) | FCOS-style object detection on COCO |
| [`computer_vision/object_detection_yolox.py`](tasks/object-detection.md#yolox-object-detection) | YOLOX object detection |
| [`computer_vision/object_detection_rtdetr.py`](tasks/object-detection.md#rt-detr-real-time-detection-transformer) | RT-DETR transformer detection |
| [`computer_vision/object_detection_transformers.py`](tasks/object-detection.md#transformer-based-object-detection) | Vision Transformers for detection |
| [`computer_vision/explore_yolox_models.py`](tasks/object-detection.md#explore-yolox-models) | Explore YOLOX model variants |
| [`computer_vision/yolox_official.py`](tasks/object-detection.md#yolox-official) | Official YOLOX implementation |

**Segmentation:**

| Script | Description |
|--------|-------------|
| [`computer_vision/semantic_segmentation.py`](tasks/semantic-segmentation.md#basic-example-cityscapes) | DeepLabV3+ semantic segmentation |
| [`computer_vision/instance_segmentation.py`](tasks/instance-segmentation.md#basic-example-coco) | Mask R-CNN style instance segmentation |

### 🤗 HuggingFace Hub Integration

**Basic Integration:**

| Script | Description |
|--------|-------------|
| [`huggingface/huggingface_hub_models.py`](integration/huggingface-hub.md#1-introduction-to-hf-hub-models) | Introduction to HF Hub |
| [`huggingface/hf_hub_classification.py`](integration/huggingface-hub.md#2-image-classification) | Classification with HF Hub |
| [`huggingface/hf_hub_segmentation.py`](integration/huggingface-hub.md#3-semantic-segmentation) | Segmentation with HF Hub |
| [`huggingface/hf_hub_object_detection.py`](integration/huggingface-hub.md#4-object-detection) | Detection with HF Hub |
| [`huggingface/hf_hub_instance_segmentation.py`](integration/huggingface-hub.md#5-instance-segmentation) | Instance segmentation with HF Hub |
| [`huggingface/hf_hub_advanced.py`](integration/huggingface-hub.md#6-advanced-usage) | Advanced HF Hub features |
| [`huggingface/hf_hub_lightning_integration.py`](../user-guide/integration/huggingface-hub-integration.md#pytorch-lightning-compatibility) | Lightning compatibility |
| [`huggingface/hf_direct_models_lightning.py`](../user-guide/integration/huggingface-transformers-integration.md) | Direct HF Transformers models |

**Advanced HF Hub:**

| Script | Description |
|--------|-------------|
| [`huggingface/hf_interpretation.py`](integration/hf_interpretation.md) | Model interpretation (GradCAM, attention, metrics) |
| [`huggingface/hf_transfer_learning.py`](integration/hf_transfer_learning.md) | LLRD, progressive unfreezing |
| [`huggingface/hf_ensemble.py`](integration/hf_ensemble.md) | Ensembles and knowledge distillation |
| [`huggingface/hf_deployment.py`](integration/hf_deployment.md) | ONNX export, quantization, serving |

### 📊 Data & Augmentation

| Script | Description |
|--------|-------------|
| [`data_training/balanced_sampling.py`](utilities/data-handling.md#balanced-sampling) | Weighted sampling for imbalanced data |
| [`data_training/albumentations_cifar10.py`](utilities/data-handling.md#albumentations-strong-augmentation) | Albumentations strong augmentation |
| [`data_training/albumentations_custom_folder.py`](utilities/data-handling.md#custom-albumentations-pipeline) | Custom albumentations pipeline |
| [`data_training/hf_custom_data.py`](utilities/hf_custom_data.md) | Advanced augmentation, multi-label, validation |

### ⚙️ Training & Optimization

| Script | Description |
|--------|-------------|
| [`data_training/auto_tuning.py`](utilities/training-utilities.md#auto-tuning) | Automatic LR and batch size finding |
| [`data_training/multi_gpu_training.py`](utilities/training-utilities.md#multi-gpu-training) | Multi-GPU and distributed training |
| [`data_training/preset_manager.py`](utilities/training-utilities.md#preset-manager) | Training presets and configurations |
| [`data_training/performance_optimization_demo.py`](utilities/training-utilities.md#performance-optimization) | Performance optimization techniques |
| [`data_training/hf_hyperparameter_tuning.py`](utilities/hf_hyperparameter_tuning.md) | Optuna hyperparameter optimization |

### 🔍 Model Interpretation

| Script | Description |
|--------|-------------|
| [`interpretation/interpretation_demo.py`](utilities/interpretation.md#basic-interpretation) | Basic interpretation and visualization |
| [`interpretation/interpretation_metrics_demo.py`](utilities/interpretation.md#interpretation-metrics) | Interpretation with metrics analysis |
| [`interpretation/interpretation_phase2_demo.py`](utilities/interpretation.md#interpretation-phase-2) | Advanced techniques (Phase 2) |
| [`interpretation/interpretation_phase3_demo.py`](utilities/interpretation.md#interpretation-phase-3) | Advanced techniques (Phase 3) |
| [`interpretation/interactive_visualization_demo.py`](utilities/interpretation.md#interactive-visualization) | Interactive Plotly visualizations |
| [`interpretation/comprehensive_interpretation_tutorial.ipynb`](utilities/interpretation.md#comprehensive-tutorial) | Comprehensive tutorial (Notebook) |

### 📈 Logging & Experiment Tracking

| Script | Description |
|--------|-------------|
| [`logging_inference/multiple_loggers.py`](utilities/logging-metrics.md#multiple-loggers) | TensorBoard + CSV logging |
| [`logging_inference/mlflow_tracking.py`](utilities/logging-metrics.md#mlflow-tracking) | MLflow experiment tracking |
| [`logging_inference/detailed_evaluation.py`](utilities/logging-metrics.md#detailed-evaluation) | Confusion matrix and per-class metrics |

### 🚢 Inference & Utilities

| Script | Description |
|--------|-------------|
| [`logging_inference/inference.py`](../user-guide/inference/classification-inference.md) | Model inference and batch prediction |
| [`logging_inference/inference_without_metrics.py`](../user-guide/inference/classification-inference.md#inference-without-metrics) | Inference without metrics |
| [`logging_inference/detection_inference.py`](../user-guide/inference/object-detection-inference.md) | Object detection inference |
| [`logging_inference/segmentation_inference.py`](../user-guide/inference/semantic-segmentation-inference.md) | Segmentation inference |
| [`logging_inference/backbone_discovery.py`](utilities/backbone-utilities.md#backbone-discovery) | Explore timm backbones |

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
python examples/getting_started/classify_cifar10.py
```

## Example Categories Summary

### 🚀 Getting Started (3 examples)
Start your journey with CIFAR-10 classification and custom datasets.

### 🎯 Computer Vision Tasks (8 examples)
- **Classification** - Basic to advanced classification
- **Object Detection** - FCOS, YOLOX, RT-DETR, transformers (6 examples)
- **Segmentation** - Semantic and instance segmentation (2 examples)

### 🤗 HuggingFace Hub (12 examples)
- **Basic Integration** - 8 task-specific examples
- **Advanced Techniques** - Interpretation, transfer learning, ensemble, deployment (4 examples)

### 📊 Data & Training (9 examples)
- **Data & Augmentation** - Balanced sampling, augmentation strategies (4 examples)
- **Training & Optimization** - Auto-tuning, multi-GPU, HPO (5 examples)

### 🔍 Understanding & Deployment (12 examples)
- **Model Interpretation** - GradCAM, attention, interactive viz (6 examples)
- **Logging & Tracking** - TensorBoard, MLflow, evaluation (3 examples)
- **Inference & Utilities** - Model inference and backbone discovery (3 examples)

**Total: 44 runnable examples**



