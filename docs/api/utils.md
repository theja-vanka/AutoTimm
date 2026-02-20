# Utilities

Utility functions for model inspection, discovery, and reproducibility.

## seed_everything

Set random seeds for reproducibility across all libraries.

### API Reference

::: autotimm.seed_everything
    options:
      show_source: true

### Usage Examples

#### Default Seed

```python
from autotimm import seed_everything

# Seed all RNGs with default value (42)
seed_everything()
```

#### Deterministic Mode

```python
# Full deterministic training (slower but reproducible)
seed_everything(42, deterministic=True)
```

#### Custom Seed

```python
seed_everything(123)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed value |
| `deterministic` | `bool` | `False` | Enable deterministic algorithms (may reduce performance) |

### Returns

| Type | Description |
|------|-------------|
| `int` | The seed value that was set |

### What Gets Seeded

- Python's `random` module
- NumPy's random number generator
- PyTorch (CPU & all CUDA devices)
- `PYTHONHASHSEED` environment variable
- cuDNN backend settings (when `deterministic=True`)
- `torch.use_deterministic_algorithms()` (when `deterministic=True`)

### Notes

- Setting `deterministic=True` may reduce performance due to slower deterministic algorithm implementations.
- When `deterministic=False` (default), cuDNN benchmark mode is enabled for faster training.
- All task classes call `seed_everything()` automatically with `seed=42, deterministic=True` by default.
- If `seed=None` is passed to a task class, `seed_everything()` is not called and a warning is emitted if `deterministic=True`.

---

## count_parameters

Count the number of parameters in a model.

### API Reference

::: autotimm.count_parameters
    options:
      show_source: true

### Usage Examples

#### Trainable Parameters

```python
import autotimm

model = autotimm.ImageClassifier(
    backbone="resnet50",
    num_classes=10,
    metrics=metrics,
)

trainable = autotimm.count_parameters(model)
print(f"Trainable parameters: {trainable:,}")
```

#### All Parameters

```python
total = autotimm.count_parameters(model, trainable_only=False)
print(f"Total parameters: {total:,}")
```

#### Backbone Only

```python
backbone = autotimm.create_backbone("resnet50")
print(f"Backbone parameters: {autotimm.count_parameters(backbone):,}")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | PyTorch model |
| `trainable_only` | `bool` | `True` | Count only trainable params |

### Returns

| Type | Description |
|------|-------------|
| `int` | Number of parameters |

---

## list_optimizers

List available optimizers from torch and timm.

### API Reference

::: autotimm.list_optimizers
    options:
      show_source: true

### Usage Examples

#### All Optimizers

```python
import autotimm

optimizers = autotimm.list_optimizers()
print("Torch optimizers:", optimizers["torch"])
print("Timm optimizers:", optimizers.get("timm", []))
```

#### Torch Only

```python
optimizers = autotimm.list_optimizers(include_timm=False)
print(optimizers["torch"])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_timm` | `bool` | `True` | Include timm optimizers |

### Returns

| Type | Description |
|------|-------------|
| `dict[str, list[str]]` | Dict with "torch" and "timm" keys |

### Available Optimizers

**Torch:**

- `adamw` - AdamW
- `adam` - Adam
- `sgd` - SGD
- `rmsprop` - RMSprop
- `adagrad` - Adagrad

**Timm:**

- `adamp` - AdamP
- `sgdp` - SGDP
- `adabelief` - AdaBelief
- `radam` - RAdam
- `adahessian` - Adahessian
- `lamb` - LAMB
- `lars` - LARS
- `madgrad` - MADGRAD
- `novograd` - NovoGrad

---

## list_schedulers

List available learning rate schedulers from torch and timm.

### API Reference

::: autotimm.list_schedulers
    options:
      show_source: true

### Usage Examples

#### All Schedulers

```python
import autotimm

schedulers = autotimm.list_schedulers()
print("Torch schedulers:", schedulers["torch"])
print("Timm schedulers:", schedulers.get("timm", []))
```

#### Torch Only

```python
schedulers = autotimm.list_schedulers(include_timm=False)
print(schedulers["torch"])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_timm` | `bool` | `True` | Include timm schedulers |

### Returns

| Type | Description |
|------|-------------|
| `dict[str, list[str]]` | Dict with "torch" and "timm" keys |

### Available Schedulers

Schedulers are dynamically discovered from PyTorch and timm. Common schedulers include:

**PyTorch (15 total):**

- `chainedscheduler` - ChainedScheduler
- `constantlr` - ConstantLR
- `cosineannealinglr` - CosineAnnealingLR
- `cosineannealingwarmrestarts` - CosineAnnealingWarmRestarts
- `cycliclr` - CyclicLR
- `exponentiallr` - ExponentialLR
- `lambdalr` - LambdaLR
- `linearlr` - LinearLR
- `multiplicativelr` - MultiplicativeLR
- `multisteplr` - MultiStepLR
- `onecyclelr` - OneCycleLR
- `polynomiallr` - PolynomialLR
- `reducelronplateau` - ReduceLROnPlateau
- `sequentiallr` - SequentialLR
- `steplr` - StepLR

**Timm (6 total):**

- `cosinelrscheduler` - CosineLRScheduler
- `multisteplrscheduler` - MultiStepLRScheduler
- `plateaulrscheduler` - PlateauLRScheduler
- `polylrscheduler` - PolyLRScheduler
- `steplrscheduler` - StepLRScheduler
- `tanhlrscheduler` - TanhLRScheduler

---

## Full Example

```python
import autotimm

# List available options
print("=== Available Optimizers ===")
optimizers = autotimm.list_optimizers()
for source, names in optimizers.items():
    print(f"{source}: {', '.join(names)}")

print("\n=== Available Schedulers ===")
schedulers = autotimm.list_schedulers()
for source, names in schedulers.items():
    print(f"{source}: {', '.join(names)}")

print("\n=== Available Backbones ===")
# Search patterns
patterns = ["*resnet*", "*efficientnet*", "*vit*", "*convnext*"]
for pattern in patterns:
    models = autotimm.list_backbones(pattern, pretrained_only=True)
    print(f"{pattern}: {len(models)} models")

print("\n=== Model Parameters ===")
for backbone_name in ["resnet18", "resnet50", "efficientnet_b0", "vit_base_patch16_224"]:
    backbone = autotimm.create_backbone(backbone_name)
    params = autotimm.count_parameters(backbone, trainable_only=False)
    features = backbone.num_features
    print(f"{backbone_name}: {params:,} params, {features} features")
```

Output:
```
=== Available Optimizers ===
torch: adamw, adam, sgd, rmsprop, adagrad
timm: adamp, sgdp, adabelief, radam, adahessian, lamb, lars, madgrad, novograd

=== Available Schedulers ===
torch: cosine, step, multistep, exponential, onecycle, plateau
timm: cosine_with_restarts

=== Available Backbones ===
*resnet*: 48 models
*efficientnet*: 64 models
*vit*: 98 models
*convnext*: 36 models

=== Model Parameters ===
resnet18: 11,689,512 params, 512 features
resnet50: 23,508,032 params, 2048 features
efficientnet_b0: 4,007,548 params, 1280 features
vit_base_patch16_224: 85,798,656 params, 768 features
```
