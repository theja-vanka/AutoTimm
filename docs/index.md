---
title: 
---

<div align="center">

<img src="autotimm.png" alt="AutoTimm" width="600" class="animated-logo" />

<h3>Automated Deep Learning for Computer Vision</h3>

<p><em>Powered by <a href="https://github.com/huggingface/pytorch-image-models">timm</a> and <a href="https://github.com/Lightning-AI/pytorch-lightning">PyTorch Lightning</a></em></p>

<hr/>

<p>Train state-of-the-art vision models with <strong>1000+ backbones</strong> in just a few lines of Python</p>

<div class="getting-started-grid">

<div class="getting-started-card">
<div class="number">1</div>
<h3>Install</h3>
<div class="code-box">pip install autotimm</div>
<a href="getting-started/installation/">Installation Guide →</a>
</div>

<div class="getting-started-card">
<div class="number">2</div>
<h3>Quick Start</h3>
<p>Train your first model in minutes</p>
<a href="getting-started/quickstart/">Quick Start Guide →</a>
</div>

<div class="getting-started-card">
<div class="number">3</div>
<h3>Explore</h3>
<p>Browse examples and dive deeper</p>
<a href="examples/">View Examples →</a>
</div>

</div>


</div>

---

## Key Features

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0;">

<div style="padding: 20px; border-left: 4px solid #1976D2;">
<strong>4 Vision Tasks</strong><br/>
Image classification, object detection, semantic segmentation, and instance segmentation
</div>

<div style="padding: 20px; border-left: 4px solid #1565C0;">
<strong>1000+ Backbones</strong><br/>
Any timm model: CNNs (ResNet, EfficientNet, ConvNeXt) and Transformers (ViT, Swin, DeiT)
</div>

<div style="padding: 20px; border-left: 4px solid #0D47A1;">
<strong>Flexible Architectures</strong><br/>
Built-in FCOS detector, DeepLabV3+ segmentation, Mask R-CNN style instance segmentation
</div>

<div style="padding: 20px; border-left: 4px solid #01579B;">
<strong>Advanced Losses</strong><br/>
Focal, Dice, Tversky, Combined CE+Dice, GIoU for bbox regression
</div>

<div style="padding: 20px; border-left: 4px solid #1976D2;">
<strong>Configurable Metrics</strong><br/>
Use torchmetrics or custom metrics with full control
</div>

<div style="padding: 20px; border-left: 4px solid #00838F;">
<strong>Multiple Loggers</strong><br/>
TensorBoard, MLflow, W&B, CSV - use them all simultaneously
</div>

<div style="padding: 20px; border-left: 4px solid #006064;">
<strong>Auto-Tuning</strong><br/>
Automatic learning rate and batch size finding
</div>

<div style="padding: 20px; border-left: 4px solid #1565C0;">
<strong>Enhanced Logging</strong><br/>
Track learning rate, gradient norms, confusion matrices and more
</div>

<div style="padding: 20px; border-left: 4px solid #0277BD;">
<strong>Flexible Transforms</strong><br/>
Torchvision (PIL) or albumentations (OpenCV) with bbox and mask support
</div>

</div>

---

## Quick Example

```python
from autotimm import (
    AutoTrainer, ImageClassifier, ImageDataModule,
    LoggerConfig, MetricConfig,
)

# Data
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=64,
)

# Metrics (explicit configuration required)
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val", "test"],
        prog_bar=True,
    ),
]

# Model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=10,
    metrics=metrics,
    lr=1e-3,
)

# Trainer with logging
trainer = AutoTrainer(
    max_epochs=10,
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
    checkpoint_monitor="val/accuracy",
)

trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

<div align="center" style="margin: 20px 0;">
That's it! Train production-ready models in <strong>~20 lines of code</strong>
</div>

---

## Why Choose AutoTimm?

<div align="center">
<table>
<thead>
<tr>
<th>Feature</th>
<th align="center">AutoTimm</th>
<th align="center">Raw PyTorch</th>
<th align="center">Lightning</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>1000+ backbones</strong></td>
<td align="center">Yes</td>
<td align="center">Manual</td>
<td align="center">Manual</td>
</tr>
<tr>
<td><strong>Configurable metrics</strong></td>
<td align="center">Yes</td>
<td align="center">Manual</td>
<td align="center">Manual</td>
</tr>
<tr>
<td><strong>Multi-logger support</strong></td>
<td align="center">Yes</td>
<td align="center">Manual</td>
<td align="center">Partial</td>
</tr>
<tr>
<td><strong>Auto LR/batch finding</strong></td>
<td align="center">Yes</td>
<td align="center">No</td>
<td align="center">Yes</td>
</tr>
<tr>
<td><strong>Lines of code</strong></td>
<td align="center"><strong>~20</strong></td>
<td align="center">~200+</td>
<td align="center">~100</td>
</tr>
</tbody>
</table>
</div>

---

<div align="center" style="margin: 50px 0;">

<a href="getting-started/installation/" style="padding: 15px 30px; background-color: #1976D2; color: white; text-decoration: none; border-radius: 4px; font-size: 18px; font-weight: bold; display: inline-block; margin: 10px;">Start Now →</a>

</div>
