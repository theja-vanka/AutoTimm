<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1IO383lY97phOg9qDVARnG9HQ2McOF7Zj" alt="AutoTimm" width="400">
</p>

<h1 align="center">AutoTimm</h1>

<p align="center">
  <strong>🚀 Train state-of-the-art vision models with minimal code</strong><br>
  From prototype to production in minutes, not hours
</p>

<p align="center">
  <a href="https://pypi.org/project/autotimm/"><img src="https://img.shields.io/pypi/v/autotimm?color=blue&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/autotimm/"><img src="https://img.shields.io/pypi/pyversions/autotimm?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/theja-vanka/AutoTimm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://github.com/theja-vanka/AutoTimm/stargazers"><img src="https://img.shields.io/github/stars/theja-vanka/AutoTimm?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="https://theja-vanka.github.io/AutoTimm/">📖 Documentation</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/getting-started/quickstart/">⚡ Quick Start</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/examples/">💡 Examples</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/api/">🔧 API Reference</a>
</p>

---

## 🎯 What is AutoTimm?

AutoTimm is a **production-ready** computer vision framework that combines [timm](https://github.com/huggingface/pytorch-image-models) (1000+ pretrained models) with [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). Train image classifiers, object detectors, and segmentation models with any timm backbone using a simple, intuitive API.

**Perfect for:**
- 🧑‍🔬 **Researchers** needing reproducible experiments and quick iterations
- 👨‍💻 **Engineers** building production ML systems with minimal boilerplate
- 🎓 **Students** learning computer vision with modern best practices
- 🚀 **Startups** rapidly prototyping vision applications

## ✨ What's New in v0.6.2

- **YOLOX Models** 🎯 Official YOLOX implementation (nano to X) with CSPDarknet backbone
- **Smart Backend Selection** 🧠 AI-powered recommendation for optimal transform backends
- **TransformConfig** ⚙️ Unified transform configuration with presets and model-specific normalization
- **Optional Metrics** 🔧 Metrics now optional for inference-only deployments
- **Python 3.10-3.14** 🐍 Latest Python support

## 🚀 Quick Start

### Installation

```bash
pip install autotimm
```

**Everything included:** PyTorch, timm, PyTorch Lightning, torchmetrics, albumentations, pycocotools, and more.

<details>
<summary><strong>Optional logging backends</strong></summary>

```bash
pip install autotimm[tensorboard]  # TensorBoard
pip install autotimm[wandb]        # Weights & Biases
pip install autotimm[mlflow]       # MLflow
pip install autotimm[all]          # All extras
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

> 💡 **Auto-tuning is enabled by default.** Disable with `tuner_config=False` for manual control.

## 🎨 Key Features

<table>
<tr>
<td><strong>🎯 4 Vision Tasks</strong></td>
<td>Classification • Object Detection • Semantic Segmentation • Instance Segmentation</td>
</tr>
<tr>
<td><strong>🧠 1000+ Backbones</strong></td>
<td>ResNet • EfficientNet • ViT • ConvNeXt • Swin • DeiT • BEiT • and more from timm</td>
</tr>
<tr>
<td><strong>🤗 HuggingFace Integration</strong></td>
<td>Load models from HF Hub with <code>hf-hub:</code> prefix + Direct Transformers support</td>
</tr>
<tr>
<td><strong>🎯 YOLOX Support</strong></td>
<td>Official YOLOX models (nano → X) + YOLOX-style heads with any timm backbone</td>
</tr>
<tr>
<td><strong>🏗️ Advanced Architectures</strong></td>
<td>DeepLabV3+ • FCOS • YOLOX • Mask R-CNN • Feature Pyramids</td>
</tr>
<tr>
<td><strong>⚡ Auto-Tuning</strong></td>
<td>Automatic LR and batch size finding—enabled by default</td>
</tr>
<tr>
<td><strong>🧠 Smart Transforms</strong></td>
<td>AI-powered backend recommendations + unified TransformConfig with presets</td>
</tr>
<tr>
<td><strong>📈 Multi-Logger Support</strong></td>
<td>TensorBoard • MLflow • Weights & Biases • CSV—use simultaneously</td>
</tr>
<tr>
<td><strong>🏭 Production Ready</strong></td>
<td>Mixed precision • Multi-GPU • Gradient accumulation • 200+ tests</td>
</tr>
</table>

## 📚 Task Examples

### 🖼️ Image Classification

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

### 🎯 Object Detection with YOLOX

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

📖 **[Complete YOLOX Guide](https://theja-vanka.github.io/AutoTimm/user-guide/models/yolox-detector/)** • ⚡ **[Quick Reference](https://theja-vanka.github.io/AutoTimm/user-guide/guides/yolox-quick-reference/)**

### 🗺️ Semantic Segmentation

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

### 🎭 Instance Segmentation

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

## 🤗 HuggingFace Integration

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

📖 **[HF Integration Comparison](https://theja-vanka.github.io/AutoTimm/user-guide/integration/huggingface-integration-comparison/)** • **[HF Hub Guide](https://theja-vanka.github.io/AutoTimm/user-guide/integration/huggingface-hub-integration/)** • **[HF Transformers Guide](https://theja-vanka.github.io/AutoTimm/user-guide/integration/huggingface-transformers-integration/)**

## 🧠 Smart Features

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

## 🔧 Explore Models

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

## 📖 Documentation & Examples

### Documentation

| Section | Description |
|---------|-------------|
| [Quick Start](https://theja-vanka.github.io/AutoTimm/getting-started/quickstart/) | Get up and running in 5 minutes |
| [User Guide](https://theja-vanka.github.io/AutoTimm/user-guide/data-loading/) | In-depth guides for all features |
| [YOLOX Guide](https://theja-vanka.github.io/AutoTimm/user-guide/models/yolox-detector/) | Complete YOLOX implementation guide |
| [API Reference](https://theja-vanka.github.io/AutoTimm/api/) | Complete API documentation |
| [Examples](https://theja-vanka.github.io/AutoTimm/examples/) | 30+ runnable code examples |

### Ready-to-Run Examples

**Classification**
- [classify_cifar10.py](examples/classify_cifar10.py) - Basic classification with auto-tuning
- [classify_custom_folder.py](examples/classify_custom_folder.py) - Train on custom dataset
- [vit_finetuning.py](examples/vit_finetuning.py) - Two-phase ViT fine-tuning
- [inference_without_metrics.py](examples/inference_without_metrics.py) - Production deployment

**Object Detection**
- [yolox_official.py](examples/yolox_official.py) - Official YOLOX models
- [object_detection_yolox.py](examples/object_detection_yolox.py) - YOLOX-style with timm
- [object_detection_coco.py](examples/object_detection_coco.py) - FCOS detection
- [object_detection_rtdetr.py](examples/object_detection_rtdetr.py) - RT-DETR (no NMS!)
- [explore_yolox_models.py](examples/explore_yolox_models.py) - Interactive YOLOX explorer

**Segmentation**
- [semantic_segmentation.py](examples/semantic_segmentation.py) - DeepLabV3+
- [instance_segmentation.py](examples/instance_segmentation.py) - Mask R-CNN style

**HuggingFace & Advanced**
- [huggingface_hub_models.py](examples/huggingface_hub_models.py) - HF Hub basics
- [hf_hub_*.py](examples/) - Comprehensive HF examples
- [multi_gpu_training.py](examples/multi_gpu_training.py) - Distributed training
- [mlflow_tracking.py](examples/mlflow_tracking.py) - MLflow tracking
- [preset_manager.py](examples/preset_manager.py) - Smart backend selection

**[Browse all examples →](https://theja-vanka.github.io/AutoTimm/examples/)**

## 🏗️ Supported Architectures

**Classification**
- Models: Any timm backbone (1000+)
- Losses: CrossEntropy with label smoothing, Mixup

**Object Detection**
- Architectures: FCOS, YOLOX (official & custom)
- Losses: Focal Loss, GIoU Loss, Centerness Loss

**Semantic Segmentation**
- Architectures: DeepLabV3+, FCN
- Losses: CrossEntropy, Dice, Focal, Combined, Tversky
- Formats: PNG masks, COCO stuff, Cityscapes, Pascal VOC

**Instance Segmentation**
- Architecture: FCOS + Mask R-CNN style mask head
- Losses: Detection losses + Binary mask loss

## 🧪 Testing

Comprehensive test suite with **200+ tests**:

```bash
# Run all tests
pytest tests/ -v

# Specific modules
pytest tests/test_classification.py
pytest tests/test_yolox.py

# With coverage
pytest tests/ --cov=autotimm --cov-report=html
```

## 🤝 Contributing

We welcome contributions!

```bash
git clone https://github.com/theja-vanka/AutoTimm.git
cd AutoTimm
pip install -e ".[dev,all]"
pytest tests/ -v
```

For major changes, please open an issue first.

## 📄 Citation

```bibtex
@software{autotimm2026,
  author = {Krishnatheja Vanka},
  title = {AutoTimm: Automatic PyTorch Image Models},
  url = {https://github.com/theja-vanka/AutoTimm},
  year = {2026},
  version = {0.6.2}
}
```

## 🌟 Why AutoTimm?

<table>
<tr>
<td width="33%">
<h3 align="center">🚀 Fast</h3>
<p align="center">
From idea to trained model in minutes. Auto-tuning, mixed precision, and multi-GPU out of the box.
</p>
</td>
<td width="33%">
<h3 align="center">🔧 Flexible</h3>
<p align="center">
1000+ backbones, 4 vision tasks, multiple transform backends. Use what works best.
</p>
</td>
<td width="33%">
<h3 align="center">🏭 Production Ready</h3>
<p align="center">
200+ tests, comprehensive logging, checkpoint management. Deploy with confidence.
</p>
</td>
</tr>
</table>

---

<p align="center">
  <strong>Built with ❤️ using <a href="https://github.com/huggingface/pytorch-image-models">timm</a> and <a href="https://github.com/Lightning-AI/pytorch-lightning">PyTorch Lightning</a></strong>
</p>

<p align="center">
  <a href="https://github.com/theja-vanka/AutoTimm">⭐ Star us on GitHub</a> •
  <a href="https://github.com/theja-vanka/AutoTimm/issues">🐛 Report Issues</a> •
  <a href="https://theja-vanka.github.io/AutoTimm/">📖 Read the Docs</a>
</p>
