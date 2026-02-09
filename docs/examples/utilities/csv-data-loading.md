# CSV Data Loading Examples

This page demonstrates loading data from CSV files for all task types.

## CSV Classification

Train a classifier using CSV-based data loading with auto-detected class names.

```python
from autotimm import ImageClassifier, ImageDataModule, AutoTrainer, MetricConfig


def main():
    # CSV format: image_path,label
    data = ImageDataModule(
        train_csv="train.csv",
        val_csv="val.csv",
        image_dir="./images",
        image_size=224,
        batch_size=32,
        balanced_sampling=True,  # Handle imbalanced classes
    )
    data.setup("fit")

    model = ImageClassifier(
        backbone="resnet50",
        num_classes=data.num_classes,
        metrics=[
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass"},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ],
    )

    trainer = AutoTrainer(max_epochs=10)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## CSV Object Detection

Train a detector from CSV with one row per bounding box.

```python
from autotimm import ObjectDetector, DetectionDataModule, AutoTrainer, MetricConfig


def main():
    # CSV format: image_path,x_min,y_min,x_max,y_max,label
    # Multiple rows per image (one per box)
    data = DetectionDataModule(
        train_csv="train_annotations.csv",
        val_csv="val_annotations.csv",
        image_dir="./images",
        image_size=640,
        batch_size=16,
    )
    data.setup("fit")

    model = ObjectDetector(
        backbone="resnet50",
        num_classes=data.num_classes,
        metrics=[
            MetricConfig(
                name="mAP",
                backend="torchmetrics",
                metric_class="MeanAveragePrecision",
                params={"box_format": "xyxy"},
                stages=["val"],
            ),
        ],
    )

    trainer = AutoTrainer(max_epochs=12, gradient_clip_val=1.0)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## CSV Semantic Segmentation

Train a segmentation model from CSV with image-mask path pairs.

```python
from autotimm import SemanticSegmentor, SegmentationDataModule, AutoTrainer, MetricConfig


def main():
    # CSV format: image_path,mask_path
    data = SegmentationDataModule(
        data_dir="./data",
        format="csv",
        train_csv="train_seg.csv",
        val_csv="val_seg.csv",
        image_size=512,
        batch_size=8,
    )

    model = SemanticSegmentor(
        backbone="resnet50",
        num_classes=21,
        head_type="deeplabv3plus",
        loss_type="combined",
        metrics=[
            MetricConfig(
                name="mIoU",
                backend="torchmetrics",
                metric_class="JaccardIndex",
                params={"task": "multiclass", "num_classes": 21},
                stages=["val"],
                prog_bar=True,
            ),
        ],
    )

    trainer = AutoTrainer(max_epochs=50)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## CSV Instance Segmentation

Train an instance segmentation model from CSV with per-instance binary masks.
No pycocotools required.

```python
from autotimm import (
    InstanceSegmentor,
    InstanceSegmentationDataModule,
    AutoTrainer,
)


def main():
    # CSV format: image_path,x_min,y_min,x_max,y_max,label,mask_path
    # Multiple rows per image (one per instance)
    data = InstanceSegmentationDataModule(
        train_csv="train_instances.csv",
        val_csv="val_instances.csv",
        image_dir="./data",
        image_size=640,
        batch_size=4,
    )

    model = InstanceSegmentor(
        backbone="resnet50",
        num_classes=10,
    )

    trainer = AutoTrainer(max_epochs=24)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
```

---

## Running Examples

```bash
python examples/data_training/csv_classification.py
python examples/data_training/csv_detection.py
python examples/data_training/csv_segmentation.py
python examples/data_training/csv_instance_segmentation.py
```

**See Also:**

- [CSV Data Loading Guide](../../user-guide/data-loading/csv-data.md) - Full CSV data documentation
- [Data Handling Examples](data-handling.md) - Folder-based and augmentation examples
- [Data Loading Guide](../../user-guide/data-loading/index.md) - Complete data loading overview
