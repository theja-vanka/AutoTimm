# CLI Examples

AutoTimm includes a YAML-config-driven command-line interface built on [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

## Quick Start

```bash
# Train a classifier on CIFAR-10
autotimm fit --config examples/cli/classification.yaml

# Quick smoke test (1 batch)
autotimm fit --config examples/cli/classification.yaml --trainer.fast_dev_run true

# Override any parameter from the command line
autotimm fit --config examples/cli/classification.yaml --model.init_args.lr 0.001 --trainer.max_epochs 20

# Validate or test a trained model
autotimm validate --config examples/cli/classification.yaml --ckpt_path best.ckpt
autotimm test --config examples/cli/classification.yaml --ckpt_path best.ckpt
```

## Example Configs

| Config | Task | Description |
|--------|------|-------------|
| `classification.yaml` | Image Classification | ResNet-18 on CIFAR-10 |
| `detection.yaml` | Object Detection | FCOS ResNet-50 on COCO |
| `segmentation.yaml` | Semantic Segmentation | DeepLabV3+ on Cityscapes |

## How It Works

Each config specifies `model`, `data`, and `trainer` sections. The `class_path` tells AutoTimm which task and data module to use:

```yaml
model:
  class_path: autotimm.ImageClassifier
  init_args:
    backbone: resnet18
    num_classes: 10

data:
  class_path: autotimm.ImageDataModule
  init_args:
    dataset_name: CIFAR10
    data_dir: ./data

trainer:
  max_epochs: 10
  accelerator: auto
```

You can also run via `python -m autotimm` instead of the `autotimm` command:

```bash
python -m autotimm fit --config examples/cli/classification.yaml
```
