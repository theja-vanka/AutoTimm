<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1IO383lY97phOg9qDVARnG9HQ2McOF7Zj" alt="AutoTimm" width="400">
</p>

<p align="center">
  <strong>Train state-of-the-art vision models with minimal code</strong><br>
  From prototype to production in minutes, not hours
</p>

<p align="center">
  <a href="https://pypi.org/project/autotimm/"><img src="https://img.shields.io/pypi/v/autotimm?color=blue&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/autotimm/"><img src="https://img.shields.io/pypi/pyversions/autotimm?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/theja-vanka/AutoTimm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://github.com/theja-vanka/AutoTimm/stargazers"><img src="https://img.shields.io/github/stars/theja-vanka/AutoTimm?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="https://theja-vanka.github.io/AutoTimm/">Documentation</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/getting-started/quickstart/">Quick Start</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/examples/">Examples</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/api/">API Reference</a>
</p>

---

## What is AutoTimm?

AutoTimm is a **production-ready** computer vision framework that combines [timm](https://github.com/huggingface/pytorch-image-models) (1000+ pretrained models) with [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). Train image classifiers, object detectors, and segmentation models with any timm backbone using a simple, intuitive API.

**Perfect for:**
- **Researchers** needing reproducible experiments and quick iterations
- **Engineers** building production ML systems with minimal boilerplate
- **Students** learning computer vision with modern best practices
- **Startups** rapidly prototyping vision applications

## Why AutoTimm?

<table>
<tr>
<td width="33%">
<h3 align="center">Fast</h3>
<p align="center">
From idea to trained model in minutes. Auto-tuning, mixed precision, and multi-GPU out of the box.
</p>
</td>
<td width="33%">
<h3 align="center">Flexible</h3>
<p align="center">
1000+ backbones, 4 vision tasks, multiple transform backends. Use what works best.
</p>
</td>
<td width="33%">
<h3 align="center">Production Ready</h3>
<p align="center">
410+ tests, comprehensive logging, checkpoint management. Deploy with confidence.
</p>
</td>
</tr>
</table>

## What's New in v0.7.3

- **CSV Data Loading** - Load data from CSV files for all task types: classification, object detection, semantic segmentation, and instance segmentation
- **CSVImageDataset** - Single-label classification from CSV with auto class detection and both torchvision/albumentations backends
- **CSVDetectionDataset** - Object detection from CSV with multi-row-per-image grouping and `xyxy` bbox format
- **CSVInstanceDataset** - Instance segmentation from CSV with binary mask PNGs (no pycocotools required)
- **CSV Segmentation** - Semantic segmentation from CSV via `format="csv"` in `SemanticSegmentationDataset`
- **All DataModules Updated** - `ImageDataModule`, `DetectionDataModule`, `SegmentationDataModule`, and `InstanceSegmentationDataModule` all support `train_csv`/`val_csv`/`test_csv` parameters
- **Multi-Label Classification** - Native multi-label support in `ImageClassifier` with `multi_label=True`, using `BCEWithLogitsLoss` and sigmoid predictions
- **MultiLabelImageDataModule** - New data module for loading multi-label datasets from CSV files with auto-detected label columns, validation splits, and rich summary tables
- **Multi-Label Metrics** - `MetricManager` now auto-injects `num_labels` and resolves `torchmetrics.classification` metrics (e.g., `MultilabelAccuracy`, `MultilabelF1Score`)

<details>
<summary><strong>v0.7.2</strong></summary>

- **torch.inference_mode** - Faster inference across all tasks, export, and interpretation using `torch.inference_mode()` instead of `torch.no_grad()`
- **Reproducibility by Default** - Automatic seeding with `seed=42` and deterministic mode enabled out-of-the-box for fully reproducible training and inference
- **torch.compile by Default** - Automatic PyTorch 2.0+ optimization enabled out-of-the-box for faster training and inference
- **TorchScript Export** - Export trained models to TorchScript (.pt) for production deployment without Python dependencies
- **Model Interpretation** - Complete explainability toolkit with 6 interpretation methods, 6 quality metrics, interactive Plotly visualizations, and up to 100x speedup with optimization
- **Tutorial Notebook** - Comprehensive Jupyter notebook covering all interpretation features end-to-end
- **YOLOX Models** - Official YOLOX implementation (nano to X) with CSPDarknet backbone
- **Smart Backend Selection** - AI-powered recommendation for optimal transform backends
- **TransformConfig** - Unified transform configuration with presets and model-specific normalization
- **Optional Metrics** - Metrics now optional for inference-only deployments
- **Python 3.10-3.14** - Latest Python support

</details>

## Quick Start

### Installation

```bash
pip install autotimm
```

**Everything included:** PyTorch, timm, PyTorch Lightning, torchmetrics, albumentations, pycocotools, and more.

<details>
<summary><strong>Optional extras</strong></summary>

```bash
# Logging backends
pip install autotimm[tensorboard]  # TensorBoard
pip install autotimm[wandb]        # Weights & Biases
pip install autotimm[mlflow]       # MLflow

# Interpretation
pip install autotimm[interactive]  # Interactive Plotly visualizations

# All extras
pip install autotimm[all]          # Everything
```

</details>

### Your First Model in 30 Seconds

```python
from autotimm import AutoTrainer, ImageClassifier, ImageDataModule, MetricConfig

# Data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=64,
)

# Metrics
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
        prog_bar=True,
    )
]

# Model
model = ImageClassifier(
    backbone="resnet18",  # Try efficientnet_b0, vit_base_patch16_224, etc.
    num_classes=10,
    metrics=metrics,
    lr=1e-3,
)

# Train with auto-tuning (finds optimal LR and batch size automatically!)
trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
```

> **Auto-tuning is enabled by default.** Disable with `tuner_config=False` for manual control.

## Key Features

<table>
<tr>
<td><strong>4 Vision Tasks</strong></td>
<td>Classification (single & multi-label) • Object Detection • Semantic Segmentation • Instance Segmentation</td>
</tr>
<tr>
<td><strong>1000+ Backbones</strong></td>
<td>ResNet • EfficientNet • ViT • ConvNeXt • Swin • DeiT • BEiT • and more from timm</td>
</tr>
<tr>
<td><strong>Model Interpretation</strong></td>
<td>6 explanation methods • 6 quality metrics • Interactive visualizations • Up to 100x speedup</td>
</tr>
<tr>
<td><strong>HuggingFace Integration</strong></td>
<td>Load models from HF Hub with <code>hf-hub:</code> prefix + Direct Transformers support</td>
</tr>
<tr>
<td><strong>YOLOX Support</strong></td>
<td>Official YOLOX models (nano → X) + YOLOX-style heads with any timm backbone</td>
</tr>
<tr>
<td><strong>Advanced Architectures</strong></td>
<td>DeepLabV3+ • FCOS • YOLOX • Mask R-CNN • Feature Pyramids</td>
</tr>
<tr>
<td><strong>Auto-Tuning</strong></td>
<td>Automatic LR and batch size finding—enabled by default</td>
</tr>
<tr>
<td><strong>Smart Transforms</strong></td>
<td>AI-powered backend recommendations + unified TransformConfig with presets</td>
</tr>
<tr>
<td><strong>Multi-Logger Support</strong></td>
<td>TensorBoard • MLflow • Weights & Biases • CSV—use simultaneously</td>
</tr>
<tr>
<td><strong>torch.compile Support</strong></td>
<td>Automatic PyTorch 2.0+ optimization • Enabled by default • Configurable modes</td>
</tr>
<tr>
<td><strong>CSV Data Loading</strong></td>
<td>Load any task from CSV files — classification, detection, segmentation, instance segmentation</td>
</tr>
<tr>
<td><strong>Production Ready</strong></td>
<td>Mixed precision • Multi-GPU • Gradient accumulation • 410+ tests</td>
</tr>
</table>

## Task Examples

### Image Classification

```python
from autotimm import ImageClassifier

# Use any timm backbone or HuggingFace model
model = ImageClassifier(
    backbone="efficientnet_b0",  # or "hf-hub:timm/resnet50.a1_in1k"
    num_classes=10,
    metrics=metrics,  # Optional for inference!
)

trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
```

### Multi-Label Classification

```python
from autotimm import ImageClassifier, MultiLabelImageDataModule, MetricConfig

# CSV data with columns: image_path, cat, dog, outdoor, indoor
data = MultiLabelImageDataModule(
    train_csv="train.csv",
    image_dir="./images",
    val_csv="val.csv",
    image_size=224,
    batch_size=32,
)
data.setup("fit")

model = ImageClassifier(
    backbone="resnet50",
    num_classes=data.num_labels,
    multi_label=True,       # BCEWithLogitsLoss + sigmoid
    threshold=0.5,
    metrics=[
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="MultilabelAccuracy",
            params={"num_labels": data.num_labels},
            stages=["train", "val"],
            prog_bar=True,
        ),
    ],
)

trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
```

### Object Detection with YOLOX

**Official YOLOX (matches paper benchmarks):**

```python
from autotimm import YOLOXDetector, DetectionDataModule

model = YOLOXDetector(
    model_name="yolox-s",  # nano, tiny, s, m, l, x
    num_classes=80,
    lr=0.01,
    optimizer="sgd",
    scheduler="yolox",
    total_epochs=300,
)

trainer = AutoTrainer(max_epochs=300, precision="16-mixed")
trainer.fit(model, datamodule=DetectionDataModule(data_dir="./coco", image_size=640))
```

**YOLOX-style head with any timm backbone:**

```python
from autotimm import ObjectDetector

model = ObjectDetector(
    backbone="resnet50",  # Experiment with any backbone!
    num_classes=80,
    detection_arch="yolox",
    fpn_channels=256,
)
```

**[Complete YOLOX Guide](https://theja-vanka.github.io/AutoTimm/user-guide/models/yolox-detector/)** • **[Quick Reference](https://theja-vanka.github.io/AutoTimm/user-guide/guides/yolox-quick-reference/)**

### Semantic Segmentation

```python
from autotimm import SemanticSegmentor, SegmentationDataModule

model = SemanticSegmentor(
    backbone="resnet50",
    num_classes=19,
    head_type="deeplabv3plus",
    loss_type="combined",  # CE + Dice for better boundaries
)

data = SegmentationDataModule(
    data_dir="./cityscapes",
    format="cityscapes",  # or "coco", "voc", "png"
    image_size=512,
)

trainer = AutoTrainer(max_epochs=100)
trainer.fit(model, datamodule=data)
```

### Instance Segmentation

```python
from autotimm import InstanceSegmentor, InstanceSegmentationDataModule

model = InstanceSegmentor(
    backbone="resnet50",
    num_classes=80,
    mask_loss_weight=1.0,
)

trainer = AutoTrainer(max_epochs=100)
trainer.fit(model, datamodule=InstanceSegmentationDataModule(data_dir="./coco"))
```

### CSV Data Loading

Load data from CSV files instead of folder structures or COCO JSON:

```python
from autotimm import ImageClassifier, ImageDataModule, AutoTrainer

# Classification from CSV (columns: image_path, label)
data = ImageDataModule(
    train_csv="train.csv",
    val_csv="val.csv",
    image_dir="./images",
    image_size=224,
    batch_size=32,
)

model = ImageClassifier(backbone="resnet50", num_classes=10)
trainer = AutoTrainer(max_epochs=10)
trainer.fit(model, datamodule=data)
```

```python
from autotimm import ObjectDetector, DetectionDataModule

# Detection from CSV (columns: image_path, x_min, y_min, x_max, y_max, label)
data = DetectionDataModule(
    train_csv="annotations.csv",
    image_dir="./images",
    image_size=640,
    batch_size=8,
)
```

CSV loading is supported for all tasks: classification, object detection, semantic segmentation, and instance segmentation.

**[CSV Data Loading Guide](https://theja-vanka.github.io/AutoTimm/user-guide/data-loading/csv-data/)**

## Model Interpretation & Explainability

Understand what your models learn and how they make decisions with comprehensive interpretation tools.

### Quick Explanation

```python
from autotimm.interpretation import quick_explain

# One-line explanation
result = quick_explain(
    model,
    image,
    method="gradcam",
    save_path="explanation.png"
)
```

### 6 Interpretation Methods

```python
from autotimm.interpretation import (
    GradCAM,                # Fast, class-discriminative (CNNs)
    GradCAMPlusPlus,        # Better for multiple objects
    IntegratedGradients,    # Theoretically sound, pixel-level
    SmoothGrad,             # Noise-reduced gradients
    AttentionRollout,       # Vision Transformers
    AttentionFlow,          # Vision Transformers
)

# Use any method
explainer = GradCAM(model)
heatmap = explainer.explain(image, target_class=5)
explainer.visualize(image, heatmap, save_path="gradcam.png")
```

### Quantitative Evaluation

```python
from autotimm.interpretation import ExplanationMetrics

metrics = ExplanationMetrics(model, explainer)

# Faithfulness metrics
deletion = metrics.deletion(image, target_class=5, steps=50)
insertion = metrics.insertion(image, target_class=5, steps=50)

# Stability metric
sensitivity = metrics.sensitivity_n(image, n_samples=50)

# Sanity checks
param_test = metrics.model_parameter_randomization_test(image)
data_test = metrics.data_randomization_test(image)

# Localization metric
pointing = metrics.pointing_game(image, bbox=(50, 50, 150, 150))

print(f"Deletion AUC: {deletion['auc']:.4f}")  # Lower = better
print(f"Insertion AUC: {insertion['auc']:.4f}")  # Higher = better
print(f"Sensitivity: {sensitivity['sensitivity']:.4f}")  # Lower = more stable
```

### Interactive Visualizations

```python
from autotimm.interpretation import InteractiveVisualizer

viz = InteractiveVisualizer(model)

# Create interactive HTML with zoom/pan/hover
fig = viz.visualize_explanation(
    image,
    explainer,
    colorscale="Viridis",
    save_path="interactive.html"
)

# Compare methods side-by-side
explainers = {
    'GradCAM': GradCAM(model),
    'GradCAM++': GradCAMPlusPlus(model),
    'Integrated Gradients': IntegratedGradients(model),
}
viz.compare_methods(image, explainers, save_path="comparison.html")

# Generate comprehensive report
viz.create_report(image, explainer, save_path="report.html")
```

### Performance Optimization

```python
from autotimm.interpretation.optimization import (
    ExplanationCache,        # 10-50x speedup
    BatchProcessor,          # 2-5x speedup
    PerformanceProfiler,     # Identify bottlenecks
    optimize_for_inference,  # 1.5-3x speedup
)

# Enable caching
cache = ExplanationCache(cache_dir="./cache", max_size_mb=5000)

# Optimize model
model = optimize_for_inference(model, use_fp16=True)

# Batch processing
processor = BatchProcessor(model, explainer, batch_size=32)
heatmaps = processor.process_batch(images)

# Profile performance
profiler = PerformanceProfiler(enabled=True)
with profiler.profile("explanation"):
    heatmap = explainer.explain(image)
profiler.print_stats()
```

### Training Integration

```python
from autotimm import AutoTrainer
from autotimm.interpretation import InterpretationCallback

# Monitor interpretations during training
callback = InterpretationCallback(
    sample_images=val_images,
    method="gradcam",
    log_every_n_epochs=5,
)

trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[callback],
    logger="tensorboard",
)
trainer.fit(model, datamodule=data)
```

**Features:**

- **6 interpretation methods** for different use cases
- **6 quality metrics** for quantitative evaluation
- **Interactive visualizations** with Plotly (zoom/pan/hover)
- **Up to 100x speedup** with caching and optimization
- **Feature visualization** and receptive field analysis
- **Training callbacks** for automatic monitoring
- **Comprehensive tutorial** notebook included

**[Interpretation Guide](https://theja-vanka.github.io/AutoTimm/user-guide/interpretation/)** • **[Tutorial Notebook](examples/comprehensive_interpretation_tutorial.ipynb)**

## HuggingFace Integration

### Three Approaches

<table>
<thead>
<tr>
<th>Approach</th>
<th>Best For</th>
<th>Example</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>HF Hub timm</strong></td>
<td>CNNs, Production</td>
<td><code>"hf-hub:timm/resnet50.a1_in1k"</code></td>
</tr>
<tr>
<td><strong>HF Transformers Direct</strong></td>
<td>Vision Transformers</td>
<td><code>ViTModel.from_pretrained(...)</code></td>
</tr>
<tr>
<td><strong>HF Transformers Auto</strong></td>
<td>Quick Prototyping</td>
<td><code>AutoModel.from_pretrained(...)</code></td>
</tr>
</tbody>
</table>

**All approaches fully support AutoTrainer** (checkpointing, early stopping, mixed precision, multi-GPU, auto-tuning).

```python
from autotimm import ImageClassifier, list_hf_hub_backbones

# Discover models
models = list_hf_hub_backbones(model_name="resnet", limit=5)

# Use any HF Hub model (just add 'hf-hub:' prefix!)
model = ImageClassifier(
    backbone="hf-hub:timm/convnext_base.fb_in22k_ft_in1k",
    num_classes=100,
)
```

**[HF Integration Comparison](https://theja-vanka.github.io/AutoTimm/user-guide/integration/huggingface-integration-comparison/)** • **[HF Hub Guide](https://theja-vanka.github.io/AutoTimm/user-guide/integration/huggingface-hub-integration/)** • **[HF Transformers Guide](https://theja-vanka.github.io/AutoTimm/user-guide/integration/huggingface-transformers-integration/)**

## Smart Features

### torch.compile Optimization

**Enabled by default** for all tasks with PyTorch 2.0+:

```python
# Default: torch.compile enabled for faster training/inference
model = ImageClassifier(backbone="resnet50", num_classes=10)

# Disable if needed
model = ImageClassifier(backbone="resnet50", num_classes=10, compile_model=False)

# Custom compile options
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    compile_kwargs={"mode": "reduce-overhead", "fullgraph": True}
)
```

**Compile modes:**
- `"default"` - Balanced performance (default)
- `"reduce-overhead"` - Lower latency, better for smaller batches
- `"max-autotune"` - Maximum optimization, longer compile time

**What gets compiled:**
- Classification: backbone + head
- Detection: backbone + FPN/neck + head
- Segmentation: backbone + segmentation head
- Instance Segmentation: backbone + FPN + detection head + mask head

Gracefully falls back on PyTorch < 2.0 with a warning.

### Reproducibility by Default

**Automatic seeding** for reproducible experiments:

```python
# Default: seed=42, deterministic=True for full reproducibility
model = ImageClassifier(backbone="resnet50", num_classes=10)
trainer = AutoTrainer(max_epochs=10)

# Custom seed
model = ImageClassifier(backbone="resnet50", num_classes=10, seed=123)
trainer = AutoTrainer(max_epochs=10, seed=123)

# Faster training (disable deterministic mode)
model = ImageClassifier(backbone="resnet50", num_classes=10, deterministic=False)
trainer = AutoTrainer(max_epochs=10, deterministic=False)

# Manual seeding
from autotimm import seed_everything
seed_everything(42, deterministic=True)
```

**What's seeded:**
- Python's `random` module
- NumPy's random number generator
- PyTorch (CPU & CUDA)
- Environment variables for reproducibility
- cuDNN deterministic algorithms (when `deterministic=True`)

**Seeding options:**
- **Model-level:** Seeds when model is created
- **Trainer-level:** Seeds before training starts (uses Lightning's seeding by default)
- **Manual:** Use `seed_everything()` for custom control

Perfect for research papers, debugging, and ensuring consistent results across runs!

### TorchScript Export

Export trained models for production deployment:

```python
from autotimm import ImageClassifier, export_to_torchscript
import torch

# Load trained model
model = ImageClassifier.load_from_checkpoint("model.ckpt")

# Export to TorchScript
example_input = torch.randn(1, 3, 224, 224)
export_to_torchscript(
    model,
    "model.pt",
    example_input=example_input,
    method="trace"  # Recommended
)

# Or use the convenience method
model.to_torchscript("model.pt")

# Load and use in production
scripted_model = torch.jit.load("model.pt")
output = scripted_model(image)
```

**Benefits:**
- No Python dependencies required
- Deploy to C++, mobile, or edge devices
- Faster inference with `torch.inference_mode()` and JIT optimizations
- Single-file deployment
- Graceful fallback if JIT optimization fails on your platform

### Smart Backend Selection

```python
from autotimm import recommend_backend, compare_backends

# Get AI-powered recommendation
rec = recommend_backend(task="detection")
config = rec.to_config(image_size=640)

# Compare backends side-by-side
compare_backends()
```

### Unified Transform Configuration

```python
from autotimm import TransformConfig, list_transform_presets

# Discover presets
list_transform_presets()  # ['default', 'autoaugment', 'randaugment', ...]

# Configure with model-specific normalization
config = TransformConfig(
    preset="randaugment",
    image_size=384,
    use_timm_config=True,  # Auto-detect mean/std from backbone
)

model = ImageClassifier(
    backbone="efficientnet_b4",
    num_classes=10,
    transform_config=config,
)
```

### Custom Auto-Tuning

```python
from autotimm import AutoTrainer, TunerConfig

# Default: Full auto-tuning
trainer = AutoTrainer(max_epochs=10)

# Disable auto-tuning
trainer = AutoTrainer(max_epochs=10, tuner_config=False)

# Custom configuration
trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,
        auto_batch_size=True,
        lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0},
    ),
)
```

### Optional Metrics for Inference

```python
# Training with metrics
model = ImageClassifier(backbone="resnet50", num_classes=10, metrics=metrics)

# Inference without metrics
model = ImageClassifier(backbone="resnet50", num_classes=10)
model = model.load_from_checkpoint("checkpoint.ckpt")
predictions = model(image)
```

## Explore Models

### YOLOX Models

```python
import autotimm

# List all YOLOX variants
autotimm.list_yolox_models()  # ['yolox-nano', 'yolox-tiny', 'yolox-s', ...]

# Get detailed specs (params, FLOPs, mAP)
autotimm.list_yolox_models(verbose=True)

# Get model info
info = autotimm.get_yolox_model_info("yolox-s")
print(f"Params: {info['params']}, mAP: {info['mAP']}")  # Params: 9.0M, mAP: 40.5

# List components
autotimm.list_yolox_backbones()
autotimm.list_yolox_necks()
autotimm.list_yolox_heads()
```

### timm Backbones

```python
# Search 1000+ timm models
autotimm.list_backbones("*efficientnet*", pretrained_only=True)
autotimm.list_backbones("*vit*")

# Search HuggingFace Hub
autotimm.list_hf_hub_backbones(model_name="resnet", limit=10)

# Inspect a model
backbone = autotimm.create_backbone("convnext_tiny")
print(f"Features: {backbone.num_features}, Params: {autotimm.count_parameters(backbone):,}")
```

## Documentation & Examples

### Documentation

Comprehensive documentation with **interactive diagrams**, search optimization, and fast navigation:

| Section | Description |
|---------|-------------|
| [Quick Start](https://theja-vanka.github.io/AutoTimm/getting-started/quickstart/) | Get up and running in 5 minutes |
| [User Guide](https://theja-vanka.github.io/AutoTimm/user-guide/data-loading/) | In-depth guides for all features |
| [Interpretation Guide](https://theja-vanka.github.io/AutoTimm/user-guide/interpretation/) | Model explainability and visualization |
| [YOLOX Guide](https://theja-vanka.github.io/AutoTimm/user-guide/models/yolox-detector/) | Complete YOLOX implementation guide |
| [API Reference](https://theja-vanka.github.io/AutoTimm/api/) | Complete API documentation |
| [Examples](https://theja-vanka.github.io/AutoTimm/examples/) | 50 runnable code examples |

### Ready-to-Run Examples

**🚀 Getting Started**
- [classify_cifar10.py](examples/classify_cifar10.py) - Basic classification with auto-tuning
- [classify_custom_folder.py](examples/classify_custom_folder.py) - Train on custom dataset
- [vit_finetuning.py](examples/vit_finetuning.py) - Two-phase ViT fine-tuning

**🎯 Computer Vision Tasks**
- Object Detection: [yolox_official.py](examples/computer_vision/yolox_official.py), [object_detection_yolox.py](examples/computer_vision/object_detection_yolox.py), [object_detection_coco.py](examples/computer_vision/object_detection_coco.py), [object_detection_rtdetr.py](examples/computer_vision/object_detection_rtdetr.py)
- Segmentation: [semantic_segmentation.py](examples/computer_vision/semantic_segmentation.py), [instance_segmentation.py](examples/computer_vision/instance_segmentation.py)

**🤗 HuggingFace Hub (14 examples)**
- Basic: [huggingface_hub_models.py](examples/huggingface/huggingface_hub_models.py), [hf_hub_*.py](examples/huggingface/) (8 task-specific files)
- Advanced: [hf_interpretation.py](examples/huggingface/hf_interpretation.py), [hf_transfer_learning.py](examples/huggingface/hf_transfer_learning.py), [hf_ensemble.py](examples/huggingface/hf_ensemble.py), [hf_deployment.py](examples/huggingface/hf_deployment.py)

**📊 Data & Training**
- CSV Data: [csv_classification.py](examples/data_training/csv_classification.py), [csv_detection.py](examples/data_training/csv_detection.py), [csv_segmentation.py](examples/data_training/csv_segmentation.py), [csv_instance_segmentation.py](examples/data_training/csv_instance_segmentation.py)
- Data: [multilabel_classification.py](examples/data_training/multilabel_classification.py) - Multi-label from CSV, [hf_custom_data.py](examples/data_training/hf_custom_data.py) - Advanced augmentation
- Training: [multi_gpu_training.py](examples/data_training/multi_gpu_training.py), [hf_hyperparameter_tuning.py](examples/data_training/hf_hyperparameter_tuning.py)
- Optimization: [preset_manager.py](examples/data_training/preset_manager.py), [performance_optimization_demo.py](examples/data_training/performance_optimization_demo.py)

**🔍 Model Understanding**
- Interpretation: [comprehensive_interpretation_tutorial.ipynb](examples/interpretation/comprehensive_interpretation_tutorial.ipynb) (40+ cells), [interpretation_metrics_demo.py](examples/interpretation/interpretation_metrics_demo.py)
- Visualization: [interactive_visualization_demo.py](examples/interpretation/interactive_visualization_demo.py) - Interactive Plotly
- Tracking: [mlflow_tracking.py](examples/logging_inference/mlflow_tracking.py) - MLflow experiment tracking

**[Browse all examples →](https://theja-vanka.github.io/AutoTimm/examples/)**

## Supported Architectures

**Classification**
- Models: Any timm backbone (1000+)
- Losses: CrossEntropy with label smoothing, Mixup; BCEWithLogitsLoss for multi-label

**Object Detection**
- Architectures: FCOS, YOLOX (official & custom)
- Losses: Focal Loss, GIoU Loss, Centerness Loss

**Semantic Segmentation**
- Architectures: DeepLabV3+, FCN
- Losses: CrossEntropy, Dice, Focal, Combined, Tversky
- Formats: PNG masks, COCO stuff, Cityscapes, Pascal VOC, CSV

**Instance Segmentation**
- Architecture: FCOS + Mask R-CNN style mask head
- Losses: Detection losses + Binary mask loss
- Formats: COCO JSON, CSV with binary mask PNGs

## Testing

Comprehensive test suite with **410+ tests**:

```bash
# Run all tests
pytest tests/ -v

# Specific modules
pytest tests/test_classification.py
pytest tests/test_yolox.py
pytest tests/test_interpretation.py
pytest tests/test_csv_datamodules.py

# With coverage
pytest tests/ --cov=autotimm --cov-report=html
```

## Contributing

We welcome contributions!

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[dev,all]"
pytest tests/ -v
```

To build the documentation locally:
```bash
./scripts/build_docs.sh
```

For more details, see [scripts/README.md](scripts/README.md).

For major changes, please open an issue first.

## Citation

```bibtex
@software{autotimm,
  author = {Krishnatheja Vanka},
  title = {AutoTimm: Automatic PyTorch Image Models},
  url = {https://github.com/theja-vanka/AutoTimm},
  year = {2026},
  version = {0.7.3}
}
```

---

<p align="center">
  <strong>Built with ❤️ using <a href="https://github.com/huggingface/pytorch-image-models">timm</a> and <a href="https://github.com/Lightning-AI/pytorch-lightning">PyTorch Lightning</a></strong>
</p>

<p align="center">
  <a href="https://github.com/theja-vanka/AutoTimm">Star us on GitHub</a> •
  <a href="https://github.com/theja-vanka/AutoTimm/issues">Report Issues</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/">Read the Docs</a>
</p>
