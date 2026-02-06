# Examples

The [`examples/`](https://github.com/theja-vanka/AutoTimm/tree/main/examples) directory contains runnable scripts demonstrating AutoTimm features.

## Quick Reference

Browse examples by topic:

- **[Classification](tasks/classification.md)** - Image classification with CIFAR-10, custom datasets, and ViT fine-tuning
- **[Object Detection](tasks/object-detection.md)** - FCOS, YOLOX, transformer-based detection, and RT-DETR
- **[Semantic Segmentation](tasks/semantic-segmentation.md)** - DeepLabV3+ and FCN architectures for semantic segmentation
- **[Instance Segmentation](tasks/instance-segmentation.md)** - Mask R-CNN style instance segmentation
- **[Hugging Face Hub](integration/huggingface-hub.md)** - Using HF Hub models for all tasks with comprehensive examples
- **[Data Handling](utilities/data-handling.md)** - Balanced sampling, augmentation, and custom transforms
- **[Training Utilities](utilities/training-utilities.md)** - Auto-tuning, multi-GPU training, presets, and performance optimization
- **[Logging & Metrics](utilities/logging-metrics.md)** - Multiple loggers, MLflow, and detailed evaluation
- **[Backbone Utilities](utilities/backbone-utilities.md)** - Discovering and comparing timm backbones
- **[Interpretation](utilities/interpretation.md)** - Model interpretation, visualization, and analysis techniques

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
| [`object_detection_yolox.py`](tasks/object-detection.md#yolox-object-detection) | YOLOX object detection | Object Detection |
| [`explore_yolox_models.py`](tasks/object-detection.md#explore-yolox-models) | Explore and compare YOLOX model variants | Object Detection |
| [`yolox_official.py`](tasks/object-detection.md#yolox-official) | Official YOLOX implementation example | Object Detection |
| [`balanced_sampling.py`](utilities/data-handling.md#balanced-sampling) | Weighted sampling for imbalanced data | Data Handling |
| [`albumentations_cifar10.py`](utilities/data-handling.md#albumentations-strong-augmentation) | Albumentations strong augmentation | Data Handling |
| [`albumentations_custom_folder.py`](utilities/data-handling.md#custom-albumentations-pipeline) | Custom albumentations pipeline | Data Handling |
| [`auto_tuning.py`](utilities/training-utilities.md#auto-tuning) | Automatic LR and batch size finding | Training Utilities |
| [`multi_gpu_training.py`](utilities/training-utilities.md#multi-gpu-training) | Multi-GPU and distributed training | Training Utilities |
| [`preset_manager.py`](utilities/training-utilities.md#preset-manager) | Managing training presets and configurations | Training Utilities |
| [`performance_optimization_demo.py`](utilities/training-utilities.md#performance-optimization) | Performance optimization techniques | Training Utilities |
| [`inference.py`](../user-guide/inference/classification-inference.md) | Model inference and batch prediction | Inference |
| [`inference_without_metrics.py`](../user-guide/inference/classification-inference.md#inference-without-metrics) | Inference without computing metrics | Inference |
| [`detection_inference.py`](../user-guide/inference/object-detection-inference.md) | Object detection inference and visualization | Inference |
| [`segmentation_inference.py`](../user-guide/inference/semantic-segmentation-inference.md) | Semantic segmentation inference and visualization | Inference |
| [`multiple_loggers.py`](utilities/logging-metrics.md#multiple-loggers) | TensorBoard + CSV logging simultaneously | Logging & Metrics |
| [`mlflow_tracking.py`](utilities/logging-metrics.md#mlflow-tracking) | MLflow experiment tracking | Logging & Metrics |
| [`detailed_evaluation.py`](utilities/logging-metrics.md#detailed-evaluation) | Confusion matrix and per-class metrics | Logging & Metrics |
| [`backbone_discovery.py`](utilities/backbone-utilities.md#backbone-discovery) | Explore timm backbones | Backbone Utilities |
| [`interpretation_demo.py`](utilities/interpretation.md#basic-interpretation) | Model interpretation and visualization | Interpretation |
| [`interpretation_metrics_demo.py`](utilities/interpretation.md#interpretation-metrics) | Interpretation with metrics analysis | Interpretation |
| [`interpretation_phase2_demo.py`](utilities/interpretation.md#interpretation-phase-2) | Advanced interpretation techniques (Phase 2) | Interpretation |
| [`interpretation_phase3_demo.py`](utilities/interpretation.md#interpretation-phase-3) | Advanced interpretation techniques (Phase 3) | Interpretation |
| [`interactive_visualization_demo.py`](utilities/interpretation.md#interactive-visualization) | Interactive visualization for model interpretation | Interpretation |
| [`comprehensive_interpretation_tutorial.ipynb`](utilities/interpretation.md#comprehensive-tutorial) | Comprehensive interpretation tutorial (Notebook) | Interpretation |
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

### Tasks
- **[Classification](tasks/classification.md)** - CIFAR-10, custom datasets, ViT fine-tuning
- **[Object Detection](tasks/object-detection.md)** - FCOS, YOLOX, transformer detection, RT-DETR
- **[Semantic Segmentation](tasks/semantic-segmentation.md)** - DeepLabV3+, FCN architectures
- **[Instance Segmentation](tasks/instance-segmentation.md)** - Mask R-CNN style segmentation

### Utilities
- **[Data Handling](utilities/data-handling.md)** - Balanced sampling, augmentation, transforms
- **[Training Utilities](utilities/training-utilities.md)** - Auto-tuning, multi-GPU training, presets, performance optimization
- **[Logging & Metrics](utilities/logging-metrics.md)** - Multiple loggers, MLflow, metrics
- **[Backbone Utilities](utilities/backbone-utilities.md)** - Discover and compare backbones
- **[Interpretation](utilities/interpretation.md)** - Model interpretation, visualization, and analysis

### Integrations
- **[Hugging Face Hub](integration/huggingface-hub.md)** - Using HF Hub models
- **[HuggingFace Transformers](../user-guide/integration/huggingface-transformers-integration.md)** - Direct HF Transformers integration



