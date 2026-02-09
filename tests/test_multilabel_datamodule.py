"""Tests for MultiLabelImageDataset and MultiLabelImageDataModule."""

import csv
import os
import tempfile

import pytest
import torch
from PIL import Image
from torchvision import transforms

from autotimm.data.dataset import MultiLabelImageDataset
from autotimm.data.multilabel_datamodule import MultiLabelImageDataModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_csv(tmpdir, num_images=10, label_names=None):
    """Create a temporary CSV and dummy images for testing."""
    if label_names is None:
        label_names = ["cat", "dog", "bird"]

    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(tmpdir, "labels.csv")
    rows = []
    for i in range(num_images):
        fname = f"img_{i:03d}.jpg"
        img_path = os.path.join(img_dir, fname)
        # Create a small dummy image
        Image.new("RGB", (32, 32), color=(i * 25 % 256, 100, 150)).save(img_path)
        labels = [int((i + j) % 2) for j in range(len(label_names))]
        rows.append([fname] + labels)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path"] + label_names)
        writer.writerows(rows)

    return csv_path, img_dir, label_names


# ---------------------------------------------------------------------------
# MultiLabelImageDataset tests
# ---------------------------------------------------------------------------


class TestMultiLabelImageDataset:
    def test_auto_detect_label_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, label_names = _create_test_csv(tmpdir)
            ds = MultiLabelImageDataset(csv_path=csv_path, image_dir=img_dir)

            assert ds.num_labels == len(label_names)
            assert ds.label_names == label_names
            assert len(ds) == 10

    def test_explicit_label_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_test_csv(
                tmpdir, label_names=["cat", "dog", "bird"]
            )
            ds = MultiLabelImageDataset(
                csv_path=csv_path,
                image_dir=img_dir,
                label_columns=["cat", "bird"],
            )

            assert ds.num_labels == 2
            assert ds.label_names == ["cat", "bird"]

    def test_returns_correct_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, label_names = _create_test_csv(tmpdir)
            tfm = transforms.Compose(
                [transforms.Resize((32, 32)), transforms.ToTensor()]
            )
            ds = MultiLabelImageDataset(
                csv_path=csv_path, image_dir=img_dir, transform=tfm
            )

            image, label = ds[0]
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, 32, 32)
            assert isinstance(label, torch.Tensor)
            assert label.shape == (len(label_names),)
            assert label.dtype == torch.float32

    def test_label_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_test_csv(tmpdir)
            ds = MultiLabelImageDataset(csv_path=csv_path, image_dir=img_dir)
            _, label = ds[0]
            # Labels should be 0 or 1
            assert ((label == 0) | (label == 1)).all()

    def test_missing_image_column_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_test_csv(tmpdir)
            with pytest.raises(ValueError, match="Image column"):
                MultiLabelImageDataset(
                    csv_path=csv_path,
                    image_dir=img_dir,
                    image_column="nonexistent",
                )

    def test_missing_label_column_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_test_csv(tmpdir)
            with pytest.raises(ValueError, match="Label column"):
                MultiLabelImageDataset(
                    csv_path=csv_path,
                    image_dir=img_dir,
                    label_columns=["nonexistent"],
                )


# ---------------------------------------------------------------------------
# MultiLabelImageDataModule tests
# ---------------------------------------------------------------------------


class TestMultiLabelImageDataModule:
    def test_setup_with_val_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_csv, img_dir, label_names = _create_test_csv(
                tmpdir, num_images=20
            )
            # Create a separate val CSV
            val_dir = os.path.join(tmpdir, "val_data")
            val_csv, _, _ = _create_test_csv(
                val_dir, num_images=5, label_names=label_names
            )

            dm = MultiLabelImageDataModule(
                train_csv=train_csv,
                image_dir=img_dir,
                val_csv=val_csv,
                image_size=32,
                batch_size=4,
                num_workers=0,
            )
            dm.setup("fit")

            assert dm.num_labels == len(label_names)
            assert dm.label_names == label_names
            assert len(dm.train_dataset) == 20

    def test_setup_with_val_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_test_csv(tmpdir, num_images=20)

            dm = MultiLabelImageDataModule(
                train_csv=csv_path,
                image_dir=img_dir,
                val_split=0.2,
                image_size=32,
                batch_size=4,
                num_workers=0,
            )
            dm.setup("fit")

            total = len(dm.train_dataset) + len(dm.val_dataset)
            assert total == 20

    def test_dataloaders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, label_names = _create_test_csv(
                tmpdir, num_images=10
            )

            dm = MultiLabelImageDataModule(
                train_csv=csv_path,
                image_dir=img_dir,
                val_split=0.2,
                image_size=32,
                batch_size=4,
                num_workers=0,
            )
            dm.setup("fit")

            train_batch = next(iter(dm.train_dataloader()))
            images, labels = train_batch
            assert images.ndim == 4  # (B, C, H, W)
            assert labels.shape[1] == len(label_names)
            assert labels.dtype == torch.float32

    def test_transform_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, img_dir, _ = _create_test_csv(tmpdir, num_images=4)

            dm = MultiLabelImageDataModule(
                train_csv=csv_path,
                image_dir=img_dir,
                image_size=64,
                batch_size=2,
                num_workers=0,
                val_split=0.5,
            )
            dm.setup("fit")

            batch = next(iter(dm.train_dataloader()))
            images, _ = batch
            # Default transforms resize to image_size
            assert images.shape[2] == 64
            assert images.shape[3] == 64
