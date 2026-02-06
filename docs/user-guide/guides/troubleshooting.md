# Troubleshooting

This guide covers common issues you may encounter when training models with AutoTimm and their solutions.

## NaN Losses

NaN (Not a Number) losses typically indicate numerical instability during training.

### Common Causes

| Cause | Symptom | Solution |
|-------|---------|----------|
| Learning rate too high | Loss spikes then becomes NaN | Reduce `lr` by 10x |
| Gradient explosion | Gradients grow unbounded | Enable gradient clipping |
| Division by zero | Specific layer outputs NaN | Check for empty batches |
| Log of zero/negative | Classification with wrong labels | Verify label range |
| FP16 underflow | Training with mixed precision | Use bf16 instead |

### Solutions

#### 1. Reduce Learning Rate

```python
from autotimm import ImageClassifier, MetricConfig

metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val"],
    )
]

model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    lr=1e-5,  # Start with a lower learning rate
)
```

#### 2. Enable Gradient Clipping

```python
from autotimm import AutoTrainer

trainer = AutoTrainer(
    max_epochs=10,
    gradient_clip_val=1.0,  # Clip gradients to max norm of 1.0
)
```

#### 3. Use Learning Rate Finder

```python
from autotimm import AutoTrainer, TunerConfig

trainer = AutoTrainer(
    max_epochs=10,
    tuner_config=TunerConfig(
        auto_lr=True,
        lr_find_kwargs={
            "min_lr": 1e-7,
            "max_lr": 1.0,
            "num_training": 100,
        },
    ),
)
```

#### 4. Switch to BF16 Precision

```python
trainer = AutoTrainer(
    max_epochs=10,
    precision="bf16-mixed",  # More stable than fp16
)
```

### Debugging NaN Losses

```python
import torch

# Enable anomaly detection to find the source
torch.autograd.set_detect_anomaly(True)

# Train with anomaly detection
trainer.fit(model, datamodule=data)

# Remember to disable after debugging
torch.autograd.set_detect_anomaly(False)
```

---

## OOM Errors

Out of Memory (OOM) errors occur when GPU memory is exhausted.

### Memory Optimization Strategies

#### 1. Reduce Batch Size

```python
from autotimm import ImageDataModule

data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    batch_size=16,  # Reduce from 64 to 16
)
```

#### 2. Use Gradient Accumulation

Simulate larger batches without increased memory:

```python
trainer = AutoTrainer(
    max_epochs=10,
    accumulate_grad_batches=4,  # Effective batch = 16 * 4 = 64
)
```

#### 3. Enable Mixed Precision

```python
trainer = AutoTrainer(
    max_epochs=10,
    precision="16-mixed",  # Uses less memory than fp32
)
```

#### 4. Use a Smaller Backbone

```python
# Instead of large models
model = ImageClassifier(backbone="resnet50", ...)  # ~98MB

# Use smaller alternatives
model = ImageClassifier(backbone="resnet34", ...)  # ~84MB
model = ImageClassifier(backbone="efficientnet_b0", ...)  # ~21MB
model = ImageClassifier(backbone="mobilenetv3_small_100", ...)  # ~10MB
```

#### 5. Reduce Image Size

```python
data = ImageDataModule(
    data_dir="./data",
    image_size=160,  # Reduce from 224
)
```

### Memory Usage by Task

| Task | Typical Memory Usage | Recommended Batch Size |
|------|---------------------|----------------------|
| Classification | 4-8 GB | 32-64 |
| Object Detection | 8-16 GB | 8-16 |
| Semantic Segmentation | 8-12 GB | 8-16 |
| Instance Segmentation | 12-24 GB | 4-8 |

### Monitoring GPU Memory

```python
import torch

# Check current memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

---

## LR Tuning Failures

The learning rate finder may fail or produce suboptimal results.

### Common Issues

#### 1. Finder Doesn't Converge

```python
# Increase the number of training steps
trainer = AutoTrainer(
    tuner_config=TunerConfig(
        auto_lr=True,
        lr_find_kwargs={
            "num_training": 200,  # Increase from default 100
            "early_stop_threshold": None,  # Disable early stopping
        },
    ),
)
```

#### 2. Suggested LR Too High

```python
# Use a more conservative range
trainer = AutoTrainer(
    tuner_config=TunerConfig(
        auto_lr=True,
        lr_find_kwargs={
            "min_lr": 1e-6,
            "max_lr": 1e-2,  # Lower max
        },
    ),
)
```

#### 3. Manual LR Selection

If auto-tuning fails, use these guidelines:

| Backbone Type | Starting LR | With Pretrained |
|--------------|-------------|-----------------|
| CNN (ResNet, EfficientNet) | 1e-3 | 1e-4 |
| Transformer (ViT, Swin) | 1e-4 | 1e-5 |
| Detection models | 1e-4 | 1e-4 |
| Segmentation models | 1e-4 | 1e-4 |

### LR Schedule Recommendations

```python
# For CNN backbones
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    lr=1e-4,
    scheduler="cosineannealinglr",
    scheduler_kwargs={"T_max": 50, "eta_min": 1e-6},
)

# For Transformer backbones
model = ImageClassifier(
    backbone="vit_base_patch16_224",
    num_classes=10,
    metrics=metrics,
    lr=1e-5,
    scheduler="cosineannealinglr",
    scheduler_kwargs={"T_max": 50, "eta_min": 1e-7},
)
```

---

## Convergence Problems

### Overfitting

**Symptoms:** Training loss decreases, validation loss increases or plateaus.

**Solutions:**

```python
# 1. Add dropout
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    head_dropout=0.5,  # Add dropout before classification head
)

# 2. Use data augmentation
data = ImageDataModule(
    data_dir="./data",
    dataset_name="CIFAR10",
    image_size=224,
    augmentation_preset="strong",  # Stronger augmentation
)

# 3. Add early stopping
from pytorch_lightning.callbacks import EarlyStopping

trainer = AutoTrainer(
    max_epochs=100,
    callbacks=[
        EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min",
        ),
    ],
)

# 4. Use weight decay
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    weight_decay=1e-4,
)
```

### Underfitting

**Symptoms:** Both training and validation loss remain high.

**Solutions:**

```python
# 1. Train longer
trainer = AutoTrainer(max_epochs=100)

# 2. Use a larger model
model = ImageClassifier(
    backbone="resnet101",  # Larger backbone
    num_classes=10,
    metrics=metrics,
)

# 3. Increase learning rate
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    lr=1e-3,  # Higher LR
)

# 4. Reduce regularization
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    weight_decay=0,
    head_dropout=0,
)
```

### Oscillating Loss

**Symptoms:** Loss fluctuates significantly without converging.

**Solutions:**

```python
# 1. Reduce learning rate
model = ImageClassifier(lr=1e-5, ...)

# 2. Increase batch size
data = ImageDataModule(batch_size=128, ...)

# 3. Use a scheduler with warm restarts
model = ImageClassifier(
    scheduler="cosineannealingwarmrestarts",
    scheduler_kwargs={"T_0": 10, "T_mult": 2},
    ...
)
```

---

## Gradient Explosion

### Detection

```python
from autotimm import LoggingConfig

# Enable gradient norm logging
logging_config = LoggingConfig(
    log_learning_rate=True,
    log_gradient_norm=True,  # Monitor gradients
)

model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    logging_config=logging_config,
)
```

### Prevention

```python
# 1. Gradient clipping (always recommended for detection/segmentation)
trainer = AutoTrainer(
    max_epochs=10,
    gradient_clip_val=1.0,  # Clip by norm
)

# Alternative: clip by value
trainer = AutoTrainer(
    max_epochs=10,
    gradient_clip_val=0.5,
    gradient_clip_algorithm="value",
)

# 2. Layer-wise learning rates for transformers
# Use lower LR for early layers, higher for later layers
model = ImageClassifier(
    backbone="vit_base_patch16_224",
    num_classes=10,
    metrics=metrics,
    lr=1e-5,  # Conservative base LR
)
```

---

## Slow Training

### Data Loading Bottleneck

```python
# Check if CPU is the bottleneck (high GPU idle time)

# 1. Increase workers
data = ImageDataModule(
    data_dir="./data",
    num_workers=8,  # Increase based on CPU cores
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
)

# 2. Use faster storage (SSD over HDD)

# 3. Preprocess data once
# Convert images to a faster format like WebP or use lmdb
```

### GPU Utilization

```python
# 1. Increase batch size (use gradient accumulation if memory limited)
data = ImageDataModule(batch_size=64, ...)

# 2. Use mixed precision
trainer = AutoTrainer(precision="bf16-mixed", ...)

# 3. Compile the model (PyTorch 2.0+)
import torch

model = ImageClassifier(...)
model = torch.compile(model, mode="reduce-overhead")
```

### Validation Frequency

```python
# Reduce validation frequency for faster training
trainer = AutoTrainer(
    max_epochs=50,
    val_check_interval=0.5,  # Validate twice per epoch instead of every epoch
)

# Or validate every N steps
trainer = AutoTrainer(
    max_epochs=50,
    val_check_interval=500,  # Validate every 500 steps
)
```

---

## Device Errors

### CUDA Errors

#### CUDA Out of Memory

See [OOM Errors](#oom-errors) section.

#### CUDA Device Not Found

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")

# Explicitly set device
trainer = AutoTrainer(
    accelerator="gpu",
    devices=[0],  # Use specific GPU
)
```

#### CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version if needed
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Multi-GPU Errors

```python
# Basic multi-GPU setup
trainer = AutoTrainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp",  # Distributed Data Parallel
)

# If DDP hangs, try different strategy
trainer = AutoTrainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp_spawn",  # Alternative DDP strategy
)

# For debugging, use single GPU first
trainer = AutoTrainer(
    accelerator="gpu",
    devices=1,
)
```

### MPS (Apple Silicon) Issues

```python
import torch

# Check MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")

# Use MPS
trainer = AutoTrainer(
    accelerator="mps",
    devices=1,
)

# Fallback to CPU if MPS has issues
trainer = AutoTrainer(
    accelerator="cpu",
)
```

---

## Common Error Reference

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `RuntimeError: CUDA out of memory` | Batch size too large | Reduce batch size, use gradient accumulation |
| `RuntimeError: CUDA error: device-side assert` | Invalid tensor values | Check labels are in valid range |
| `ValueError: Expected input batch_size to match target batch_size` | Mismatched batch dimensions | Check data loader output shapes |
| `RuntimeError: Given groups=1, weight of size [X], expected input[Y]` | Wrong input channels | Check image channels (RGB vs grayscale) |
| `IndexError: index out of range` | Label exceeds num_classes | Verify num_classes matches actual labels |
| `RuntimeError: element 0 of tensors does not require grad` | Frozen model or detached tensor | Check model.train() is called |
| `ValueError: optimizer got an empty parameter list` | No trainable parameters | Check model isn't fully frozen |
| `RuntimeError: Expected all tensors on same device` | Mixed CPU/GPU tensors | Ensure all inputs are on same device |
| `FileNotFoundError: No such file or directory` | Wrong data path | Verify data_dir path exists |
| `KeyError: 'images'` | Wrong COCO annotation format | Check annotation JSON structure |

---

## Getting Help

If you encounter an issue not covered here:

1. **Check the logs**: Enable verbose logging to get more details
   ```python
   logging_config = LoggingConfig(verbosity=2)
   ```

2. **Minimal reproduction**: Create a minimal example that reproduces the issue

3. **Report issues**: Open an issue at [GitHub Issues](https://github.com/theja-vanka/AutoTimm/issues)

---

## Data Loading Issues

### Corrupted or Missing Images

```python
# Enable validation to skip corrupted images
from autotimm import ImageDataModule

data = ImageDataModule(
    data_dir="./data",
    dataset_name="custom",
    validate_images=True,  # Skip corrupted images
)
```

### Dataset Not Found

```python
import os

# Verify data directory structure
print("Data directory contents:")
print(os.listdir("./data"))

# For COCO format detection
# Expected structure:
# data/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   └── annotations/
#       ├── instances_train.json
#       └── instances_val.json
```

### Annotation Format Issues

**Symptoms:** KeyError, ValueError when loading annotations

**Solutions:**

```python
# For COCO format, verify structure
import json

with open("data/annotations/instances_train.json") as f:
    coco_data = json.load(f)

# Check required keys
required_keys = ["images", "annotations", "categories"]
for key in required_keys:
    assert key in coco_data, f"Missing key: {key}"

# Verify image IDs match
image_ids = {img["id"] for img in coco_data["images"]}
ann_image_ids = {ann["image_id"] for ann in coco_data["annotations"]}
print(f"Images: {len(image_ids)}, Annotated: {len(ann_image_ids)}")
```

### Class Imbalance Warnings

```python
from collections import Counter

# Check class distribution
def check_class_distribution(datamodule):
    train_labels = []
    for batch in datamodule.train_dataloader():
        labels = batch["labels"] if isinstance(batch, dict) else batch[1]
        train_labels.extend(labels.tolist())

    counts = Counter(train_labels)
    print("Class distribution:", counts)

    # If imbalanced, use weighted loss
    class_weights = [1.0 / count for count in counts.values()]
    return class_weights

# Apply weighted loss
from autotimm import ImageClassifier
class_weights = check_class_distribution(data)
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    loss_fn="crossentropyloss",
    loss_kwargs={"weight": torch.tensor(class_weights)},
)
```

---

## Model Loading and Checkpoint Issues

### Checkpoint Compatibility

```python
import torch

# Check checkpoint contents
checkpoint = torch.load("path/to/checkpoint.ckpt")
print("Checkpoint keys:", checkpoint.keys())
print("State dict keys:", checkpoint["state_dict"].keys() if "state_dict" in checkpoint else "No state_dict")

# Load with strict=False to ignore mismatched keys
model = ImageClassifier.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    strict=False,  # Ignore missing/unexpected keys
)
```

### Pretrained Model Download Failures

```python
# If download fails, manually specify cache directory
import os
os.environ["TORCH_HOME"] = "/path/to/cache"

# Or disable pretrained weights
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    pretrained=False,  # Train from scratch
)
```

### Version Mismatch Errors

```bash
# Check installed versions
pip list | grep torch
pip list | grep timm
pip list | grep autotimm

# Upgrade to compatible versions
pip install --upgrade torch torchvision timm
```

---

## Metric Calculation Issues

### Unexpected Metric Values

**Problem:** Metrics return 0, NaN, or unexpected values

**Solutions:**

```python
from autotimm import MetricConfig

# 1. Verify metric configuration matches task
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={
            "task": "multiclass",  # Must match: binary, multiclass, multilabel
            "num_classes": 10,
        },
        stages=["train", "val"],
    )
]

# 2. Check prediction format
def debug_predictions(model, batch):
    outputs = model(batch["images"])
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Label shape: {batch['labels'].shape}")
    print(f"Label range: [{batch['labels'].min()}, {batch['labels'].max()}]")

# 3. Verify label encoding
# For classification: labels should be integers 0 to num_classes-1
# For detection: check bbox format (xyxy, xywh, cxcywh)
```

### Detection mAP Issues

```python
# Common issue: bbox format mismatch
from autotimm import ObjectDetector

# Specify bbox format explicitly
model = ObjectDetector(
    backbone="resnet50",
    num_classes=10,
    bbox_format="xyxy",  # Options: xyxy, xywh, cxcywh
)

# Verify annotation format
# COCO format uses [x, y, width, height]
# Model expects format specified in bbox_format parameter
```

---

## Export and Inference Issues

### ONNX Export Failures

```python
import torch

# 1. Export with dynamic axes for variable input sizes
model = ImageClassifier.load_from_checkpoint("checkpoint.ckpt")
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["output"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=14,  # Use higher opset for better compatibility
)

# 2. If export fails, simplify model
model.to_torchscript(
    file_path="model.pt",
    method="trace",  # Try "script" if trace fails
)
```

### TorchScript Issues

```python
# Some operations don't support TorchScript
# Try tracing instead of scripting
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# Or freeze the model
frozen_model = torch.jit.freeze(traced_model)
frozen_model.save("model_frozen.pt")
```

### Inference Optimization

```python
# 1. Compile model for faster inference (PyTorch 2.0+)
import torch
model = torch.compile(model, mode="reduce-overhead")

# 2. Use half precision for inference
model = model.half()
input_tensor = input_tensor.half()

# 3. Disable gradient computation
with torch.no_grad():
    with torch.cuda.amp.autocast():  # Automatic mixed precision
        outputs = model(inputs)
```

---

## Integration Issues

### Weights & Biases (WandB) Issues

```python
# 1. Login issues
import wandb
wandb.login(key="your_api_key")

# 2. Disable online sync for offline training
trainer = AutoTrainer(
    max_epochs=10,
    logger="wandb",
    logger_kwargs={
        "project": "my-project",
        "offline": True,  # Save logs locally
    },
)

# 3. Resume run
trainer = AutoTrainer(
    logger_kwargs={
        "project": "my-project",
        "id": "run_id",
        "resume": "must",
    },
)
```

### TensorBoard Issues

```python
from autotimm import LoggingConfig

# Specify custom log directory
logging_config = LoggingConfig(
    log_dir="./custom_logs",
    log_hyperparameters=True,
)

# View logs
# tensorboard --logdir ./custom_logs

# If port is occupied
# tensorboard --logdir ./custom_logs --port 6007
```

### HuggingFace Hub Issues

```python
from huggingface_hub import login

# Login to HuggingFace
login(token="your_token")

# Push model with retry
model.push_to_hub(
    repo_id="username/model-name",
    commit_message="Initial commit",
    private=True,  # Make repository private
)

# If push fails, check permissions
# https://huggingface.co/settings/tokens
```

---

## Reproducibility Issues

### Setting Random Seeds

```python
import torch
import random
import numpy as np
import pytorch_lightning as pl

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

set_seed(42)

# Use deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure trainer
trainer = AutoTrainer(
    max_epochs=10,
    deterministic=True,
)
```

### Non-Deterministic Operations

```python
# Some operations are non-deterministic by design
# To identify them:
import torch
torch.use_deterministic_algorithms(True)

# This will raise errors for non-deterministic operations
# Common culprits:
# - torch.nn.functional.interpolate (bilinear mode)
# - torch.scatter_add_
# - Atomic operations in CUDA

# Workaround: disable or replace non-deterministic ops
```

---

## Data Augmentation Issues

### Augmentation Too Strong

**Symptoms:** Training accuracy remains low, loss doesn't converge

```python
# Use weaker augmentation preset
data = ImageDataModule(
    data_dir="./data",
    augmentation_preset="light",  # Instead of "strong"
)

# Or disable augmentation temporarily
data = ImageDataModule(
    data_dir="./data",
    augmentation_preset=None,
)
```

### Custom Transform Errors

```python
from autotimm import TransformConfig

# Debug transforms
transform_config = TransformConfig(
    train_preset="light",
    additional_transforms=[
        {
            "transform": "ColorJitter",
            "params": {"brightness": 0.2, "contrast": 0.2},
        }
    ],
)

# Test transform on single image
from PIL import Image
img = Image.open("test_image.jpg")
transforms = transform_config.get_train_transforms(image_size=224)

try:
    transformed = transforms(img)
    print(f"Transform successful: {transformed.shape}")
except Exception as e:
    print(f"Transform failed: {e}")
```

### Bbox Transforms for Detection

```python
# Ensure bbox transforms are compatible
from autotimm import DetectionDataModule

data = DetectionDataModule(
    data_dir="./data",
    image_size=640,
    bbox_format="xyxy",  # Must match your annotations
    # Geometric transforms automatically handle bboxes
    augmentation_preset="medium",
)
```

---

## Performance Profiling

### Identifying Bottlenecks

```python
import torch.profiler

# Profile training
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Run a few training steps
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Profile 10 batches
            break
        outputs = model(batch["images"])
        loss = criterion(outputs, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")
# View in chrome://tracing
```

### Data Loading Profiling

```python
import time

# Measure data loading time
loader = datamodule.train_dataloader()
times = []

for i, batch in enumerate(loader):
    start = time.time()
    # Just iterate, don't process
    end = time.time()
    times.append(end - start)
    if i >= 100:
        break

print(f"Average batch load time: {sum(times)/len(times):.4f}s")
print(f"Max batch load time: {max(times):.4f}s")

# If slow, increase num_workers
data = ImageDataModule(
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)
```

---

## Distributed Training Issues

### DDP Hangs or Deadlocks

```python
# 1. Set environment variables
import os
os.environ["NCCL_DEBUG"] = "INFO"  # Debug NCCL issues

# 2. Use timeout to detect hangs
trainer = AutoTrainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp",
    plugins=[
        {"timeout": 1800}  # 30 minute timeout
    ],
)

# 3. If still hangs, try ddp_spawn
trainer = AutoTrainer(
    strategy="ddp_spawn",
    devices=2,
)
```

### Multi-Node Training Issues

```bash
# Set master node address
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500

# Run on each node
python train.py --num_nodes=2 --node_rank=0  # Master node
python train.py --num_nodes=2 --node_rank=1  # Worker node
```

```python
# In code
trainer = AutoTrainer(
    accelerator="gpu",
    devices=4,
    num_nodes=2,
    strategy="ddp",
)
```

---

## Callback and Hook Errors

### Custom Callback Issues

```python
from pytorch_lightning.callbacks import Callback

class DebugCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if outputs is correct
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: loss = {outputs['loss']:.4f}")

# Add to trainer
trainer = AutoTrainer(
    callbacks=[DebugCallback()],
)
```

### Hook Registration Issues

```python
# Register hooks for debugging
def forward_hook(module, input, output):
    print(f"{module.__class__.__name__}: {output.shape}")

# Attach to specific layer
model.backbone.register_forward_hook(forward_hook)

# Remove hook after debugging
handle = model.backbone.register_forward_hook(forward_hook)
handle.remove()
```

---

## Common Warning Messages

| Warning | Meaning | Action |
|---------|---------|--------|
| `UserWarning: The dataloader does not have many workers` | Slow data loading | Increase `num_workers` |
| `UserWarning: Trying to infer the batch_size` | Can't determine batch size | Explicitly set in datamodule |
| `UserWarning: The number of training batches is very small` | Epoch finishes quickly | Increase dataset size or reduce batch size |
| `FutureWarning: Passing (type, 1) for ndim` | Deprecated numpy usage | Update to latest version |
| `UserWarning: Mixed precision is not supported on CPU` | Using wrong accelerator | Switch to GPU or remove precision flag |

---

## See Also

- [Training Guide](../training/training.md) - Complete training documentation
- [Metrics Guide](../evaluation/metrics.md) - Metric configuration
- [Logging Guide](logging.md) - Logging configuration
