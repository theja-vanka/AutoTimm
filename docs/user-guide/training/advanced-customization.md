# Advanced Customization

This guide covers advanced customization options in AutoTimm, including custom heads, loss functions, metrics, backbone modifications, and Lightning callbacks.

## Custom Classification Heads

### Basic Custom Head

```python
import torch
import torch.nn as nn
from autotimm import ImageClassifier, MetricConfig


class CustomClassificationHead(nn.Module):
    """Custom classification head with multiple layers."""

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# Using custom head with ImageClassifier
class CustomImageClassifier(ImageClassifier):
    """ImageClassifier with custom head."""

    def __init__(self, *args, hidden_dim: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the default head
        self.head = CustomClassificationHead(
            in_features=self.backbone.num_features,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
        )


# Usage
metrics = [
    MetricConfig(
        name="accuracy",
        backend="torchmetrics",
        metric_class="Accuracy",
        params={"task": "multiclass"},
        stages=["train", "val"],
    )
]

model = CustomImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
    hidden_dim=1024,
)
```

### Attention-Based Head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """Classification head with self-attention."""

    def __init__(self, in_features: int, num_classes: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, features] -> [B, 1, features] for attention
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x.squeeze(1)
        return self.fc(x)
```

---

## Custom Detection Heads

### Modified FCOS Head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from autotimm import DetectionHead


class CustomDetectionHead(DetectionHead):
    """Detection head with additional scale prediction."""

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_convs: int = 4,
        predict_scale: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            num_convs=num_convs,
        )
        self.predict_scale = predict_scale

        if predict_scale:
            # Additional scale prediction branch
            self.scale_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, features):
        cls_outputs, reg_outputs, centerness_outputs = super().forward(features)

        if self.predict_scale:
            scale_outputs = []
            for feat in features:
                reg_feat = self.reg_convs(feat)
                scale_out = self.scale_pred(reg_feat)
                scale_outputs.append(scale_out)
            return cls_outputs, reg_outputs, centerness_outputs, scale_outputs

        return cls_outputs, reg_outputs, centerness_outputs
```

---

## Custom Segmentation Heads

### UNet-Style Decoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    """UNet-style decoder for segmentation."""

    def __init__(self, encoder_channels: list, num_classes: int, decoder_channels: int = 256):
        super().__init__()
        self.num_classes = num_classes

        # Decoder blocks (bottom-up)
        self.decoder_blocks = nn.ModuleList()
        in_channels = encoder_channels[-1]

        for enc_ch in reversed(encoder_channels[:-1]):
            self.decoder_blocks.append(
                self._make_decoder_block(in_channels + enc_ch, decoder_channels)
            )
            in_channels = decoder_channels

        # Final classifier
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list) -> torch.Tensor:
        # features: [C1, C2, C3, C4, C5] from encoder
        x = features[-1]

        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

            # Skip connection
            skip = features[-(i + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            # Concatenate and decode
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)

        return self.classifier(x)
```

---

## Custom Loss Functions

### Focal Tversky Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for highly imbalanced segmentation."""

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Weight for false negatives
        self.beta = beta  # Weight for false positives
        self.gamma = gamma  # Focal parameter
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)

        # Create valid mask
        valid_mask = targets != self.ignore_index

        # One-hot encode targets
        targets_clamped = targets.clamp(0, self.num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Apply valid mask
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_expanded
        targets_one_hot = targets_one_hot * valid_mask_expanded

        # Flatten
        probs = probs.flatten(2)
        targets_one_hot = targets_one_hot.flatten(2)

        # Tversky components
        tp = (probs * targets_one_hot).sum(dim=2)
        fp = (probs * (1 - targets_one_hot)).sum(dim=2)
        fn = ((1 - probs) * targets_one_hot).sum(dim=2)

        # Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        # Focal Tversky loss
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky.mean()
```

### Label Smoothing Cross Entropy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
```

### Boundary Loss for Segmentation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    """Boundary loss for segmentation tasks."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def _compute_distance_map(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute distance transform for a binary mask."""
        mask_np = mask.cpu().numpy()
        dist = distance_transform_edt(mask_np) + distance_transform_edt(1 - mask_np)
        return torch.from_numpy(dist).to(mask.device).float()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        total_loss = 0.0

        for c in range(self.num_classes):
            # Get class probability and target
            class_prob = probs[:, c]
            class_target = (targets == c).float()

            # Compute distance map for target
            dist_maps = []
            for b in range(class_target.size(0)):
                dist_map = self._compute_distance_map(class_target[b])
                dist_maps.append(dist_map)
            dist_map = torch.stack(dist_maps)

            # Boundary loss
            total_loss += (dist_map * class_prob).mean()

        return total_loss / self.num_classes
```

---

## Custom Metrics

### Creating Torchmetrics Subclass

```python
import torch
import torchmetrics
from autotimm import MetricConfig


class WeightedAccuracy(torchmetrics.Metric):
    """Accuracy weighted by class frequency."""

    def __init__(self, num_classes: int, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes

        if class_weights is None:
            class_weights = torch.ones(num_classes)
        self.register_buffer("class_weights", class_weights)

        self.add_state("correct", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.argmax(dim=1)

        for c in range(self.num_classes):
            mask = target == c
            self.correct[c] += (preds[mask] == target[mask]).sum()
            self.total[c] += mask.sum()

    def compute(self) -> torch.Tensor:
        per_class_acc = self.correct / self.total.clamp(min=1)
        weighted_acc = (per_class_acc * self.class_weights).sum() / self.class_weights.sum()
        return weighted_acc


# Usage in MetricConfig
weighted_accuracy = MetricConfig(
    name="weighted_accuracy",
    backend="custom",
    metric_class="mymodule.WeightedAccuracy",  # Full path to your class
    params={"num_classes": 10},
    stages=["val", "test"],
)
```

### F-Beta Score

```python
import torch
import torchmetrics


class FBetaScore(torchmetrics.Metric):
    """F-beta score with configurable beta."""

    def __init__(self, num_classes: int, beta: float = 1.0, average: str = "macro"):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.average = average

        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.argmax(dim=1)

        for c in range(self.num_classes):
            pred_c = preds == c
            target_c = target == c

            self.tp[c] += (pred_c & target_c).sum()
            self.fp[c] += (pred_c & ~target_c).sum()
            self.fn[c] += (~pred_c & target_c).sum()

    def compute(self) -> torch.Tensor:
        beta_sq = self.beta ** 2
        precision = self.tp / (self.tp + self.fp).clamp(min=1e-7)
        recall = self.tp / (self.tp + self.fn).clamp(min=1e-7)

        fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall).clamp(min=1e-7)

        if self.average == "macro":
            return fbeta.mean()
        elif self.average == "micro":
            tp_sum = self.tp.sum()
            fp_sum = self.fp.sum()
            fn_sum = self.fn.sum()
            precision = tp_sum / (tp_sum + fp_sum).clamp(min=1e-7)
            recall = tp_sum / (tp_sum + fn_sum).clamp(min=1e-7)
            return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall).clamp(min=1e-7)
        else:
            return fbeta
```

---

## Backbone Modifications

### Freezing Backbone Layers

```python
from autotimm import ImageClassifier, MetricConfig


def freeze_backbone_layers(model, num_layers_to_freeze: int):
    """Freeze the first N layers of the backbone."""
    # Get all backbone modules
    backbone_modules = list(model.backbone.children())

    for i, module in enumerate(backbone_modules):
        if i < num_layers_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
            print(f"Froze layer {i}: {module.__class__.__name__}")


# Usage
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
)

# Freeze first 6 layers (up to and including layer2)
freeze_backbone_layers(model, 6)
```

### Layer-Wise Learning Rate Decay

```python
from autotimm import ImageClassifier


class LayerWiseLRClassifier(ImageClassifier):
    """Classifier with layer-wise learning rate decay."""

    def __init__(self, *args, lr_decay: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_decay = lr_decay

    def configure_optimizers(self):
        # Get backbone layers
        backbone_layers = list(self.backbone.children())
        num_layers = len(backbone_layers)

        # Create parameter groups with decaying LR
        param_groups = []

        for i, layer in enumerate(backbone_layers):
            layer_lr = self._lr * (self.lr_decay ** (num_layers - i - 1))
            param_groups.append(
                {"params": layer.parameters(), "lr": layer_lr}
            )

        # Head with base LR
        param_groups.append({"params": self.head.parameters(), "lr": self._lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self._weight_decay)

        # Add scheduler if configured
        if self._scheduler:
            scheduler = self._create_scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer
```

### Custom Feature Extraction

```python
import timm
import torch
import torch.nn as nn


class MultiScaleBackbone(nn.Module):
    """Backbone that returns features at multiple scales."""

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4],  # C2, C3, C4, C5
        )
        self.feature_channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list:
        return self.backbone(x)


# Usage
backbone = MultiScaleBackbone("resnet50")
features = backbone(torch.randn(1, 3, 224, 224))
for i, feat in enumerate(features):
    print(f"Feature {i}: {feat.shape}")
```

---

## Extending Task Classes

### Custom Training Step

```python
import torch
from autotimm import ImageClassifier


class MixupClassifier(ImageClassifier):
    """Classifier with Mixup augmentation during training."""

    def __init__(self, *args, mixup_alpha: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixup_alpha = mixup_alpha

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Apply mixup
        if self.training and self.mixup_alpha > 0:
            images, targets_a, targets_b, lam = self._mixup_data(images, targets)

            # Forward pass
            logits = self(images)

            # Mixed loss
            loss = lam * self.criterion(logits, targets_a) + (1 - lam) * self.criterion(
                logits, targets_b
            )
        else:
            logits = self(images)
            loss = self.criterion(logits, targets)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _mixup_data(self, x, y):
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam
```

### Custom Validation with Visualization

```python
import torch
from autotimm import ImageClassifier


class VisualizingClassifier(ImageClassifier):
    """Classifier that logs sample predictions during validation."""

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        # Log metrics
        self.log("val/loss", loss, prog_bar=True)

        # Update metrics
        for name, metric in self.val_metrics.items():
            metric.update(logits, targets)

        # Log sample predictions (first batch only)
        if batch_idx == 0 and self.logger:
            self._log_predictions(images, logits, targets)

        return loss

    def _log_predictions(self, images, logits, targets, num_samples=8):
        preds = logits.argmax(dim=1)

        # Create visualization
        for i in range(min(num_samples, images.size(0))):
            img = images[i].cpu()
            pred = preds[i].item()
            target = targets[i].item()

            # Denormalize image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = img.clamp(0, 1)

            # Log to tensorboard
            self.logger.experiment.add_image(
                f"val/sample_{i}_pred{pred}_true{target}",
                img,
                self.current_epoch,
            )
```

---

## Custom Augmentation Pipelines

### Using Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from autotimm import ImageDataModule


def get_strong_augmentations(image_size: int = 224):
    """Create strong augmentation pipeline."""
    return A.Compose(
        [
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


# Usage with ImageDataModule
data = ImageDataModule(
    data_dir="./data",
    image_size=224,
    batch_size=32,
    train_transforms=get_strong_augmentations(224),
)
```

---

## Lightning Callbacks

### Gradient Monitoring Callback

```python
import pytorch_lightning as pl
import torch


class GradientMonitor(pl.Callback):
    """Monitor gradient statistics during training."""

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norms = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Simplify name for logging
                simple_name = name.replace(".", "/")
                grad_norms[f"grad_norm/{simple_name}"] = grad_norm

        # Log statistics
        if grad_norms:
            values = list(grad_norms.values())
            pl_module.log("grad_norm/mean", sum(values) / len(values))
            pl_module.log("grad_norm/max", max(values))
            pl_module.log("grad_norm/min", min(values))
```

### Learning Rate Warmup Callback

```python
import pytorch_lightning as pl


class LearningRateWarmup(pl.Callback):
    """Linear learning rate warmup."""

    def __init__(self, warmup_steps: int = 1000, start_lr: float = 1e-7):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = None

    def on_train_start(self, trainer, pl_module):
        # Store target LR
        optimizer = trainer.optimizers[0]
        self.target_lr = optimizer.param_groups[0]["lr"]

        # Set initial LR
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.start_lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step >= self.warmup_steps:
            return

        # Linear warmup
        progress = trainer.global_step / self.warmup_steps
        current_lr = self.start_lr + progress * (self.target_lr - self.start_lr)

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        pl_module.log("warmup/lr", current_lr)
```

### Model EMA Callback

```python
import copy

import pytorch_lightning as pl
import torch


class ModelEMA(pl.Callback):
    """Exponential Moving Average of model weights."""

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_model = None

    def on_fit_start(self, trainer, pl_module):
        # Create EMA model
        self.ema_model = copy.deepcopy(pl_module)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update EMA weights
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), pl_module.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        # Swap to EMA weights for validation
        self._swap_weights(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Swap back to training weights
        self._swap_weights(pl_module)

    def _swap_weights(self, pl_module):
        for ema_param, param in zip(
            self.ema_model.parameters(), pl_module.parameters()
        ):
            ema_param.data, param.data = param.data.clone(), ema_param.data.clone()
```

### Usage with AutoTrainer

```python
from autotimm import AutoTrainer

trainer = AutoTrainer(
    max_epochs=50,
    callbacks=[
        GradientMonitor(log_every_n_steps=100),
        LearningRateWarmup(warmup_steps=500),
        ModelEMA(decay=0.999),
    ],
)

trainer.fit(model, datamodule=data)
```

---

## Performance Optimization

### torch.compile (PyTorch 2.0+)

**Enabled by default** in all AutoTimm tasks for automatic optimization:

```python
from autotimm import ImageClassifier, MetricConfig

# Default: torch.compile enabled
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)
```

Disable or customize:

```python
# Disable compilation
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    compile_model=False,  # Disable torch.compile
)

# Custom compile options
model = ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    compile_kwargs={
        "mode": "reduce-overhead",  # "default", "reduce-overhead", "max-autotune"
        "fullgraph": True,           # Attempt full graph compilation
        "dynamic": False,            # Static vs dynamic shapes
    },
)
```

**Compile Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `default` | Balanced optimization | General purpose (recommended) |
| `reduce-overhead` | Lower latency | Small batches, inference |
| `max-autotune` | Maximum speed | Longer compile time, production |

**What Gets Compiled:**

- **ImageClassifier**: Backbone + Head
- **ObjectDetector**: Backbone + FPN + Detection Head
- **SemanticSegmentor**: Backbone + Segmentation Head
- **InstanceSegmentor**: Backbone + FPN + Detection Head + Mask Head
- **YOLOXDetector**: Backbone + Neck + Head

**Notes:**

- First run will be slower due to compilation
- Falls back gracefully on PyTorch < 2.0
- Compatible with all custom heads and modifications

---

## Complete Example

```python
import torch
import torch.nn as nn
import torchmetrics
from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    LoggerConfig,
    MetricConfig,
)


# Custom head
class MLPHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev_dim = in_features
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Custom classifier
class CustomClassifier(ImageClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = MLPHead(
            self.backbone.num_features,
            self.num_classes,
            hidden_dims=[512, 256],
        )


def main():
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
    ]

    data = ImageDataModule(
        data_dir="./data",
        dataset_name="CIFAR10",
        image_size=224,
        batch_size=64,
    )

    model = CustomClassifier(
        backbone="resnet50",
        num_classes=10,
        metrics=metrics,
        lr=1e-4,
    )

    trainer = AutoTrainer(
        max_epochs=50,
        logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
        checkpoint_monitor="val/accuracy",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## See Also

- [Training Guide](training.md) - Training configuration
- [Loss Comparison](loss-comparison.md) - Built-in loss functions
- [Metric Selection](../evaluation/metric-selection.md) - Built-in metrics
- [API Reference: Heads](../../api/heads.md) - Head module API
