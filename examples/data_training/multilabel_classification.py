"""Multi-label image classification with AutoTimm.

This example demonstrates:
- Using MultiLabelImageDataModule to load CSV-based multi-label data
- Training ImageClassifier with multi_label=True (BCEWithLogitsLoss + sigmoid)
- Configuring multilabel metrics (MultilabelAccuracy, MultilabelF1Score)
- Inference with per-label sigmoid probabilities

Usage:
    python examples/data_training/multilabel_classification.py
"""

from __future__ import annotations

import csv
import os
import tempfile

from PIL import Image

from autotimm import (
    AutoTrainer,
    ImageClassifier,
    MetricConfig,
    MultiLabelImageDataModule,
)


def create_demo_dataset(root: str, num_images: int = 100) -> tuple[str, str, str]:
    """Create a tiny synthetic multi-label dataset for demonstration.

    Returns (train_csv, val_csv, image_dir).
    """
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)

    label_names = ["cat", "dog", "outdoor", "indoor"]

    for split, n in [("train", num_images), ("val", num_images // 5)]:
        csv_path = os.path.join(root, f"{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path"] + label_names)
            for i in range(n):
                fname = f"{split}_{i:04d}.jpg"
                Image.new("RGB", (64, 64), color=(i * 17 % 256, 100, 150)).save(
                    os.path.join(img_dir, fname)
                )
                labels = [(i + j) % 2 for j in range(len(label_names))]
                writer.writerow([fname] + labels)

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
    # In practice your CSVs look like:
    #
    #   image_path,cat,dog,outdoor,indoor
    #   img_001.jpg,1,0,1,0
    #   img_002.jpg,0,1,0,1
    #   ...
    tmpdir = tempfile.mkdtemp()
    train_csv, val_csv, image_dir = create_demo_dataset(tmpdir)

    num_labels = 4  # cat, dog, outdoor, indoor

    data = MultiLabelImageDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        image_dir=image_dir,
        image_size=224,
        batch_size=16,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 2. Define multilabel metrics
    # ------------------------------------------------------------------
    # Use torchmetrics.classification.Multilabel* metrics.
    # Note: num_labels is required; num_classes and task are auto-injected
    # and filtered as needed by MetricManager.
    metric_configs = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="MultilabelAccuracy",
            params={"num_labels": num_labels},
            stages=["train", "val"],
            prog_bar=True,
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="MultilabelF1Score",
            params={"num_labels": num_labels, "average": "macro"},
            stages=["val"],
        ),
    ]

    # ------------------------------------------------------------------
    # 3. Create model with multi_label=True
    # ------------------------------------------------------------------
    # This switches:
    #   - Loss:       CrossEntropyLoss  ->  BCEWithLogitsLoss
    #   - Predictions: argmax            ->  sigmoid > threshold
    #   - predict_step: softmax          ->  sigmoid
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=num_labels,
        multi_label=True,
        threshold=0.5,
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
