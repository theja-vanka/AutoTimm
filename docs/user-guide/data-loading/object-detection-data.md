# Object Detection Data

The `DetectionDataModule` handles object detection datasets in COCO format, including support for custom annotations, class filtering, and detection-specific augmentations.

## COCO Dataset Format

Expected directory structure:

```
coco/
  train2017/              # Training images
    000000000001.jpg
    000000000002.jpg
    ...
  val2017/                # Validation images
    000000000001.jpg
    ...
  test2017/               # Test images (optional)
    000000000001.jpg
    ...
  annotations/
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json  # Optional
```

The annotation files follow the standard COCO JSON format with image metadata, categories, and bounding box annotations.

---

## Basic Usage

```python
from autotimm import DetectionDataModule

data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,         # Standard COCO size
    batch_size=16,
    num_workers=4,
)
```

---

## Augmentation Presets

DetectionDataModule supports albumentations-based augmentation presets designed for object detection:

```python
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    augmentation_preset="default",  # or "strong"
)
```

Available presets:

| Preset | Description |
|--------|-------------|
| `default` | RandomResizedCrop (80-100%), HorizontalFlip |
| `strong` | Affine, blur, noise, brightness/contrast adjustments |

**Preset Details:**

### Default Preset
- Random resized crop (scale 0.8-1.0)
- Horizontal flip (50%)
- Standard normalization

### Strong Preset
- All default transforms
- Affine transformations (rotation, scale, shear)
- Gaussian blur and noise
- Brightness/contrast adjustments
- Color jittering
- CoarseDropout for regularization

---

## Custom Transforms

Use custom albumentations pipelines with bbox-aware transforms:

```python
import albumentations as A

custom_train = A.Compose(
    [
        A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
)

custom_eval = A.Compose(
    [
        A.Resize(height=640, width=640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
)

data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    train_transforms=custom_train,
    eval_transforms=custom_eval,
)
```

**Important**: Always include `bbox_params` when using custom transforms to ensure bounding boxes are properly transformed along with the images.

### Recommended Detection Transforms

| Transform | Use Case | Parameters |
|-----------|----------|------------|
| `RandomResizedCrop` | Scale variation | `scale=(0.8, 1.0)` |
| `HorizontalFlip` | Mirror augmentation | `p=0.5` |
| `ShiftScaleRotate` | Geometric variation | `shift_limit=0.1, scale_limit=0.1, rotate_limit=15` |
| `RandomBrightnessContrast` | Lighting variation | `brightness_limit=0.2, contrast_limit=0.2` |
| `GaussianBlur` | Robustness to blur | `blur_limit=(3, 7), p=0.3` |
| `HueSaturationValue` | Color variation | `hue_shift_limit=20, sat_shift_limit=30` |
| `CoarseDropout` | Regularization | `max_holes=8, max_height=32, max_width=32` |

**Avoid**: Transforms that can crop out bounding boxes entirely or make them too small.

---

## Filter by Class

Train on a subset of classes:

```python
# COCO class IDs (1-indexed):
# 1: person, 2: bicycle, 3: car, etc.

selected_classes = [1, 2, 3]  # person, bicycle, car

data = DetectionDataModule(
    data_dir="./coco",
    image_size=512,
    batch_size=8,
    class_ids=selected_classes,
)
```

This:
- Filters the dataset to only include images containing these classes
- Removes annotations for other classes
- Remaps class IDs to 0-based indexing (e.g., [1, 2, 3] → [0, 1, 2])
- Updates `num_classes` automatically

**Use Cases:**
- Training specialized detectors (e.g., only vehicles)
- Reducing dataset size for faster experimentation
- Domain-specific applications

---

## Filter Small Boxes

Remove very small bounding boxes that are difficult to detect:

```python
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    min_bbox_area=32.0,  # Minimum box area in pixels
)
```

This filters out bounding boxes smaller than the specified area, which helps:
- Avoid training on boxes too small to detect reliably
- Reduce noise in training data
- Improve model focus on detectable objects

**Guidelines:**
- `min_bbox_area=32.0`: Standard for most applications
- `min_bbox_area=64.0`: For large object detection
- `min_bbox_area=16.0`: When small objects are critical

---

## DataLoader Options

Fine-tune data loading performance:

```python
data = DetectionDataModule(
    data_dir="./coco",
    batch_size=16,
    num_workers=8,            # Parallel data loading
    pin_memory=True,          # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=4,        # Batches to prefetch per worker
)
```

**Performance Tips:**

- **num_workers**: Start with `4`, increase if CPU is underutilized
- **batch_size**: Decrease if you hit OOM errors (detection uses more memory than classification)
- **pin_memory**: Always `True` when using GPU
- **persistent_workers**: `True` reduces worker restart overhead
- **prefetch_factor**: Increase to `4-8` if CPU can keep up

---

## Dataset Summary

Get information about your detection dataset:

```python
data = DetectionDataModule(data_dir="./coco", image_size=640)
data.setup("fit")
print(data.summary())
```

Output (Rich table):

```
┌─────────────────────┬──────────┐
│ Field               │ Value    │
├─────────────────────┼──────────┤
│ Data dir            │ ./coco   │
│ Image size          │ 640      │
│ Batch size          │ 16       │
│ Num classes         │ 80       │
│ Train samples       │ 118287   │
│ Val samples         │ 5000     │
│ Test samples        │ 0        │
└─────────────────────┴──────────┘
```

---

## Full Parameter Reference

```python
DetectionDataModule(
    data_dir="./coco",           # COCO dataset root directory
    image_size=640,              # Target image size (square)
    batch_size=16,               # Batch size
    num_workers=4,               # Data loading workers
    train_transforms=None,       # Custom albumentations transforms for training
    eval_transforms=None,        # Custom albumentations transforms for eval
    augmentation_preset=None,    # "default" or "strong"
    class_ids=None,              # List of class IDs to filter (None = all)
    min_bbox_area=0.0,           # Minimum bounding box area in pixels
    pin_memory=True,             # Pin memory for GPU transfer
    persistent_workers=False,    # Keep workers alive between epochs
    prefetch_factor=None,        # Batches to prefetch per worker
)
```

---

## Complete Example

```python
from autotimm import (
    AutoTrainer,
    DetectionDataModule,
    LoggerConfig,
    MetricConfig,
    ObjectDetector,
)

# Data
data = DetectionDataModule(
    data_dir="./coco",
    image_size=640,
    batch_size=16,
    num_workers=4,
    augmentation_preset="default",
    min_bbox_area=32.0,
)

# Metrics
metric_configs = [
    MetricConfig(
        name="mAP",
        backend="torchmetrics",
        metric_class="MeanAveragePrecision",
        params={"box_format": "xyxy", "iou_type": "bbox"},
        stages=["val", "test"],
        prog_bar=True,
    ),
]

# Model
model = ObjectDetector(
    backbone="resnet50",
    num_classes=80,
    metrics=metric_configs,
    fpn_channels=256,
    head_num_convs=4,
    lr=1e-4,
    scheduler="multistep",
    scheduler_kwargs={"milestones": [8, 11], "gamma": 0.1},
)

# Trainer
trainer = AutoTrainer(
    max_epochs=12,
    logger=[LoggerConfig(backend="tensorboard", params={"save_dir": "logs"})],
    checkpoint_monitor="val/map",
    checkpoint_mode="max",
    gradient_clip_val=1.0,
)

# Train
trainer.fit(model, datamodule=data)
trainer.test(model, datamodule=data)
```

---

## Custom COCO Datasets

To use your own COCO-format dataset:

1. **Organize your data** following the COCO structure
2. **Create annotation files** in COCO JSON format
3. **Ensure class IDs** are consistent and 1-indexed (COCO standard)
4. **Verify images** are in the expected directories

### Minimal COCO Annotation Structure

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "000001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "class_name"
    }
  ]
}
```

---

## See Also

- [Image Classification Data](image-classification-data.md) - For classification datasets
- [Object Detection Examples](../examples/object-detection.md) - More examples and use cases
- [Training Guide](training.md) - How to train detection models
