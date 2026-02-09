"""Train an image classifier using CSV-based data loading.

This example demonstrates:
- Using ImageDataModule with train_csv for single-label classification
- Auto-detection of class names from CSV label values
- Balanced sampling for imbalanced CSV datasets
- Separate train/val CSV files

Usage:
    python examples/data_training/csv_classification.py
"""

from __future__ import annotations

import csv
import os
import tempfile

from PIL import Image

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    ImageDataModule,
    MetricConfig,
)


def create_demo_dataset(root: str, num_images: int = 100) -> tuple[str, str, str]:
    """Create a tiny synthetic CSV dataset for demonstration.

    Returns (train_csv, val_csv, image_dir).
    """
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)

    classes = ["cat", "dog", "bird"]

    for split, n in [("train", num_images), ("val", num_images // 5)]:
        csv_path = os.path.join(root, f"{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            for i in range(n):
                fname = f"{split}_{i:04d}.jpg"
                Image.new("RGB", (64, 64), color=(i * 17 % 256, 100, 150)).save(
                    os.path.join(img_dir, fname)
                )
                writer.writerow([fname, classes[i % len(classes)]])

    return (
        os.path.join(root, "train.csv"),
        os.path.join(root, "val.csv"),
        img_dir,
    )


def main():
    # ------------------------------------------------------------------
    # 1. Prepare data (replace with your own CSV files and image dir)
    # ------------------------------------------------------------------
    # For this demo we create a tiny synthetic dataset.
    # In practice your CSV looks like:
    #
    #   image_path,label
    #   img_001.jpg,cat
    #   img_002.jpg,dog
    #   img_003.jpg,bird
    #   ...
    tmpdir = tempfile.mkdtemp()
    train_csv, val_csv, image_dir = create_demo_dataset(tmpdir)

    data = ImageDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        image_dir=image_dir,
        image_size=224,
        batch_size=16,
        num_workers=0,
        balanced_sampling=True,
    )
    data.setup("fit")

    print(f"Classes: {data.class_names}")
    print(f"Num classes: {data.num_classes}")
    print(f"Train samples: {len(data.train_dataset)}")
    print(f"Val samples: {len(data.val_dataset)}")

    # ------------------------------------------------------------------
    # 2. Define metrics
    # ------------------------------------------------------------------
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val"],
            prog_bar=True,
        ),
    ]

    # ------------------------------------------------------------------
    # 3. Create model
    # ------------------------------------------------------------------
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=data.num_classes,
        metrics=metric_configs,
        lr=1e-3,
        scheduler="cosine",
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    trainer = AutoTrainer(
        max_epochs=3,
        accelerator="auto",
    )

    trainer.fit(model, datamodule=data)

    print("\nTraining complete!")
    print(
        "In production, replace the synthetic dataset with your own CSV "
        "and image directory."
    )


if __name__ == "__main__":
    main()
