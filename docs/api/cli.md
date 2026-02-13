# AutoTimmCLI

YAML-config-driven command-line interface built on LightningCLI.

## AutoTimmCLI

A subclass of `LightningCLI` that uses `AutoTrainer` by default and discovers all AutoTimm task and data module classes automatically.

### API Reference

::: autotimm.cli.AutoTimmCLI
    options:
      show_source: true
      members:
        - __init__

### Usage

```bash
# Train with a YAML config
autotimm fit --config config.yaml

# Or via python module
python -m autotimm fit --config config.yaml

# Override parameters
autotimm fit --config config.yaml --trainer.max_epochs 20

# Show help
autotimm --help
autotimm fit --help
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `fit` | Train the model |
| `validate` | Run one validation epoch |
| `test` | Run one test epoch |
| `predict` | Run inference |

---

## main()

Entry point function for the CLI.

::: autotimm.cli.main
    options:
      show_source: true

### Usage

Called automatically when using the `autotimm` command or `python -m autotimm`.

Can also be called programmatically:

```python
from autotimm.cli import main

main()
```

---

## Config File Format

The CLI expects a YAML config with `model`, `data`, and `trainer` sections:

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
    batch_size: 32
    image_size: 224

trainer:
  max_epochs: 10
  accelerator: auto
```

### Supported Model Classes

| Class | Import Path |
|-------|-------------|
| `ImageClassifier` | `autotimm.ImageClassifier` |
| `ObjectDetector` | `autotimm.ObjectDetector` |
| `YOLOXDetector` | `autotimm.YOLOXDetector` |
| `SemanticSegmentor` | `autotimm.SemanticSegmentor` |
| `InstanceSegmentor` | `autotimm.InstanceSegmentor` |

### Supported Data Modules

| Class | Import Path |
|-------|-------------|
| `ImageDataModule` | `autotimm.ImageDataModule` |
| `MultiLabelImageDataModule` | `autotimm.MultiLabelImageDataModule` |
| `DetectionDataModule` | `autotimm.DetectionDataModule` |
| `SegmentationDataModule` | `autotimm.SegmentationDataModule` |
| `InstanceSegmentationDataModule` | `autotimm.InstanceSegmentationDataModule` |

---

See also: [CLI User Guide](../user-guide/training/cli.md) | [AutoTrainer](trainer.md) | [Example Configs](https://github.com/theja-vanka/AutoTimm/tree/main/examples/cli)
