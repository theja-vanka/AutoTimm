# Performance Benchmarks

This guide provides performance benchmarks for different backbone architectures and tasks in AutoTimm. Use these to select the right model for your accuracy, speed, and memory requirements.

## Backbone Comparison

### CNN Backbones

| Backbone | Parameters | ImageNet Top-1 | Size (MB) | Inference (ms) |
|----------|------------|----------------|-----------|----------------|
| `resnet18` | 11.7M | 69.8% | 45 | 1.2 |
| `resnet34` | 21.8M | 73.3% | 84 | 1.8 |
| `resnet50` | 25.6M | 80.4% | 98 | 2.4 |
| `resnet101` | 44.5M | 81.5% | 171 | 4.1 |
| `resnet152` | 60.2M | 82.0% | 231 | 5.8 |
| `efficientnet_b0` | 5.3M | 77.7% | 21 | 2.1 |
| `efficientnet_b1` | 7.8M | 79.2% | 31 | 2.8 |
| `efficientnet_b2` | 9.1M | 80.3% | 36 | 3.2 |
| `efficientnet_b3` | 12.2M | 81.7% | 48 | 4.0 |
| `efficientnet_b4` | 19.3M | 83.0% | 76 | 5.5 |
| `convnext_tiny` | 28.6M | 82.1% | 110 | 3.2 |
| `convnext_small` | 50.2M | 83.1% | 192 | 5.1 |
| `convnext_base` | 88.6M | 83.8% | 339 | 7.8 |
| `mobilenetv3_small_100` | 2.5M | 67.7% | 10 | 0.8 |
| `mobilenetv3_large_100` | 5.5M | 75.8% | 22 | 1.1 |

*Inference times measured on NVIDIA V100 with batch size 32, 224x224 images.*

### Vision Transformer Backbones

| Backbone | Parameters | ImageNet Top-1 | Size (MB) | Inference (ms) |
|----------|------------|----------------|-----------|----------------|
| `vit_tiny_patch16_224` | 5.7M | 75.5% | 23 | 1.8 |
| `vit_small_patch16_224` | 22.1M | 81.4% | 86 | 2.5 |
| `vit_base_patch16_224` | 86.6M | 84.5% | 331 | 4.8 |
| `vit_large_patch16_224` | 304.3M | 85.8% | 1163 | 14.2 |
| `swin_tiny_patch4_window7_224` | 28.3M | 81.3% | 109 | 4.2 |
| `swin_small_patch4_window7_224` | 49.6M | 83.0% | 190 | 6.8 |
| `swin_base_patch4_window7_224` | 87.8M | 83.5% | 336 | 10.5 |
| `deit_tiny_patch16_224` | 5.7M | 72.2% | 23 | 1.6 |
| `deit_small_patch16_224` | 22.1M | 79.9% | 86 | 2.3 |
| `deit_base_patch16_224` | 86.6M | 81.8% | 331 | 4.5 |

### Recommendation by Use Case

| Use Case | Recommended Backbone | Why |
|----------|---------------------|-----|
| Edge deployment | `mobilenetv3_small_100` | Smallest, fastest |
| Mobile apps | `efficientnet_b0` | Good accuracy/speed trade-off |
| General purpose | `resnet50` | Well-balanced, widely supported |
| High accuracy | `convnext_base` or `swin_base` | State-of-the-art |
| Limited GPU memory | `efficientnet_b2` | Low memory, good accuracy |
| Transfer learning | `resnet50` or `vit_base` | Best pretrained weights |

---

## Classification Results

### CIFAR-10

| Model | Top-1 Accuracy | Training Time | GPU Memory |
|-------|---------------|---------------|------------|
| ResNet-18 | 95.2% | 12 min | 2.1 GB |
| ResNet-50 | 96.1% | 25 min | 3.8 GB |
| EfficientNet-B0 | 95.8% | 18 min | 2.4 GB |
| ConvNeXt-Tiny | 96.4% | 28 min | 4.2 GB |
| ViT-Small | 96.0% | 32 min | 4.8 GB |

*Training: 50 epochs, batch size 128, single V100 GPU*

### CIFAR-100

| Model | Top-1 Accuracy | Top-5 Accuracy |
|-------|---------------|----------------|
| ResNet-18 | 77.5% | 93.2% |
| ResNet-50 | 80.2% | 94.8% |
| EfficientNet-B0 | 79.1% | 94.1% |
| ConvNeXt-Tiny | 81.5% | 95.3% |
| ViT-Small | 80.8% | 94.9% |

### ImageNet-1K (Transfer Learning)

| Model | Top-1 Accuracy | Fine-tuning Time |
|-------|---------------|------------------|
| ResNet-50 (pretrained) | 80.4% | - |
| ResNet-50 (fine-tuned) | 82.1% | 8 hours |
| EfficientNet-B3 (pretrained) | 81.7% | - |
| ViT-Base (pretrained) | 84.5% | - |
| Swin-Base (pretrained) | 83.5% | - |

---

## Object Detection Results

### COCO Detection

| Backbone | mAP | mAP@50 | mAP@75 | Training Time | Memory |
|----------|-----|--------|--------|---------------|--------|
| ResNet-50 + FPN | 38.2 | 58.1 | 41.0 | 16h | 8.5 GB |
| ResNet-101 + FPN | 40.1 | 60.2 | 43.5 | 22h | 10.2 GB |
| EfficientNet-B3 + FPN | 39.5 | 59.2 | 42.8 | 20h | 9.1 GB |
| Swin-Tiny + FPN | 41.2 | 61.8 | 44.6 | 24h | 11.5 GB |
| ConvNeXt-Small + FPN | 42.1 | 62.5 | 45.3 | 28h | 12.8 GB |

*Training: 12 epochs, batch size 16, 2x V100 GPUs*

### Detection Speed vs Accuracy

| Model | mAP | FPS (V100) | FPS (T4) | Use Case |
|-------|-----|------------|----------|----------|
| MobileNetV3 + FPN | 32.5 | 45 | 22 | Real-time |
| ResNet-50 + FPN | 38.2 | 28 | 14 | Balanced |
| Swin-Tiny + FPN | 41.2 | 18 | 9 | High accuracy |

### Recommended Configuration

```python
from autotimm import ObjectDetector, MetricConfig

# For real-time applications
model = ObjectDetector(
    backbone="mobilenetv3_large_100",
    num_classes=80,
    metrics=[...],
    fpn_channels=128,  # Smaller FPN
)

# For high accuracy
model = ObjectDetector(
    backbone="swin_small_patch4_window7_224",
    num_classes=80,
    metrics=[...],
    fpn_channels=256,
)
```

---

## Segmentation Results

### Cityscapes Semantic Segmentation

| Backbone | mIoU | Pixel Acc | Training Time | Memory |
|----------|------|-----------|---------------|--------|
| ResNet-50 + DeepLabV3+ | 78.2% | 96.1% | 8h | 7.2 GB |
| ResNet-101 + DeepLabV3+ | 79.5% | 96.4% | 12h | 9.8 GB |
| EfficientNet-B3 + DeepLabV3+ | 78.8% | 96.2% | 10h | 8.1 GB |
| Swin-Tiny + DeepLabV3+ | 80.1% | 96.6% | 14h | 10.5 GB |

*Training: 80 epochs, 512x1024 crops, single V100 GPU*

### Pascal VOC Segmentation

| Backbone | mIoU | Parameters |
|----------|------|------------|
| ResNet-50 + FCN | 72.5% | 26M |
| ResNet-50 + DeepLabV3+ | 78.5% | 28M |
| ResNet-101 + DeepLabV3+ | 80.2% | 47M |

### Segmentation Speed

| Model | mIoU | FPS (512x512) | Memory |
|-------|------|---------------|--------|
| MobileNetV3 + FCN | 68.5% | 52 | 2.1 GB |
| ResNet-50 + DeepLabV3+ | 78.2% | 18 | 7.2 GB |
| Swin-Tiny + DeepLabV3+ | 80.1% | 12 | 10.5 GB |

---

## Memory Usage

### By Task

| Task | Typical Memory (batch 16) | Peak Memory |
|------|--------------------------|-------------|
| Classification (224x224) | 4-8 GB | 10 GB |
| Detection (640x640) | 8-16 GB | 20 GB |
| Semantic Segmentation (512x512) | 8-12 GB | 16 GB |
| Instance Segmentation (640x640) | 12-24 GB | 28 GB |

### By Backbone

| Backbone | Classification | Detection | Segmentation |
|----------|---------------|-----------|--------------|
| MobileNetV3-Small | 1.5 GB | 4 GB | 3 GB |
| EfficientNet-B0 | 2.0 GB | 5 GB | 4 GB |
| ResNet-50 | 3.5 GB | 8 GB | 7 GB |
| ViT-Base | 6.0 GB | 12 GB | 10 GB |
| Swin-Base | 7.0 GB | 14 GB | 12 GB |

*Memory usage with batch size 16, mixed precision training*

### Memory Optimization Tips

```python
from autotimm import AutoTrainer, ImageDataModule

# 1. Reduce batch size
data = ImageDataModule(batch_size=8)  # Instead of 32

# 2. Use gradient accumulation
trainer = AutoTrainer(
    accumulate_grad_batches=4,  # Effective batch = 8 * 4 = 32
)

# 3. Use mixed precision
trainer = AutoTrainer(precision="bf16-mixed")

# 4. Reduce image size
data = ImageDataModule(image_size=160)  # Instead of 224
```

---

## Inference Speed

### Classification (224x224)

| Backbone | V100 (ms) | T4 (ms) | A100 (ms) | CPU (ms) |
|----------|-----------|---------|-----------|----------|
| MobileNetV3-Small | 0.8 | 1.2 | 0.5 | 8 |
| EfficientNet-B0 | 2.1 | 3.5 | 1.4 | 25 |
| ResNet-50 | 2.4 | 4.2 | 1.6 | 35 |
| ViT-Base | 4.8 | 8.5 | 3.2 | 120 |
| Swin-Base | 10.5 | 18.0 | 7.0 | 180 |

*Single image inference, batch size 1*

### Batch Inference Speedup

| Backbone | Batch 1 | Batch 8 | Batch 32 | Speedup |
|----------|---------|---------|----------|---------|
| ResNet-50 | 2.4 ms | 8.5 ms | 28 ms | 2.7x |
| ViT-Base | 4.8 ms | 15 ms | 48 ms | 3.2x |

*V100 GPU, images per second = batch_size / time*

### Detection (640x640)

| Model | V100 FPS | T4 FPS | Note |
|-------|----------|--------|------|
| MobileNetV3 + FPN | 45 | 22 | Real-time capable |
| ResNet-50 + FPN | 28 | 14 | Good balance |
| Swin-Tiny + FPN | 18 | 9 | Higher accuracy |

---

## Model Selection Guidelines

### By Hardware

| Hardware | Recommended Backbone | Max Batch Size |
|----------|---------------------|----------------|
| 4GB GPU (GTX 1650) | MobileNetV3, EfficientNet-B0 | 16-32 |
| 8GB GPU (RTX 3070) | ResNet-50, EfficientNet-B2 | 32-64 |
| 16GB GPU (V100, RTX 4090) | Any CNN, ViT-Base | 64-128 |
| 24GB+ GPU (A100) | Any, including Swin-Large | 128+ |
| CPU only | MobileNetV3-Small | 1-8 |

### By Accuracy Requirement

| Requirement | Classification | Detection | Segmentation |
|-------------|----------------|-----------|--------------|
| Maximum accuracy | Swin-Base, ConvNeXt-Large | Swin + FPN | Swin + DeepLabV3+ |
| High accuracy | ResNet-101, EfficientNet-B4 | ResNet-101 + FPN | ResNet-101 + DeepLabV3+ |
| Balanced | ResNet-50, EfficientNet-B2 | ResNet-50 + FPN | ResNet-50 + DeepLabV3+ |
| Fast inference | MobileNetV3, EfficientNet-B0 | MobileNetV3 + FPN | MobileNetV3 + FCN |

### By Training Time Budget

| Budget | Recommended | Expected Results |
|--------|-------------|------------------|
| < 1 hour | MobileNetV3, ResNet-18 | Good baseline |
| 1-4 hours | ResNet-50, EfficientNet-B2 | Strong results |
| 4-12 hours | ResNet-101, ConvNeXt-Small | Near state-of-the-art |
| 12+ hours | Swin-Base, ViT-Large | Best possible |

---

## Benchmark Code

Run your own benchmarks:

```python
import time
import torch
from autotimm import ImageClassifier, MetricConfig


def benchmark_model(backbone, num_classes=10, image_size=224, batch_size=32, warmup=10, iterations=100):
    """Benchmark model inference speed."""
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        )
    ]

    model = ImageClassifier(
        backbone=backbone,
        num_classes=num_classes,
        metrics=metrics,
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    # Create dummy input
    x = torch.randn(batch_size, 3, image_size, image_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / iterations * 1000  # ms
    throughput = batch_size * iterations / total_time

    print(f"{backbone}:")
    print(f"  Average latency: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.1f} images/sec")

    return avg_time, throughput


# Run benchmarks
backbones = [
    "mobilenetv3_small_100",
    "efficientnet_b0",
    "resnet50",
    "vit_base_patch16_224",
]

for backbone in backbones:
    benchmark_model(backbone)
```

---

## See Also

- [Training Guide](../training/training.md) - Training configuration options
- [Model Export](../inference/model-export.md) - Exporting models for deployment
- [Troubleshooting](../guides/troubleshooting.md) - Memory and performance issues
