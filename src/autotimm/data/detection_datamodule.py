"""Detection data module for COCO-format datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader

from autotimm.data.detection_dataset import COCODetectionDataset, detection_collate_fn
from autotimm.data.detection_transforms import (
    detection_eval_transforms,
    get_detection_transforms,
)
from autotimm.data.transform_config import TransformConfig


class DetectionDataModule(pl.LightningDataModule):
    """Lightning data module for object detection with COCO-format annotations.

    Expects COCO-style directory structure::

        data_dir/
          train2017/           # Training images
          val2017/             # Validation images
          annotations/
            instances_train2017.json
            instances_val2017.json

    Or custom paths can be provided via constructor arguments.

    Parameters:
        data_dir: Root directory containing images and annotations.
        train_images_dir: Path to training images. Defaults to data_dir/train2017.
        val_images_dir: Path to validation images. Defaults to data_dir/val2017.
        train_ann_file: Path to train annotations. Defaults to
            data_dir/annotations/instances_train2017.json.
        val_ann_file: Path to val annotations. Defaults to
            data_dir/annotations/instances_val2017.json.
        test_images_dir: Optional path to test images.
        test_ann_file: Optional path to test annotations.
        image_size: Target image size (square).
        batch_size: Batch size for all dataloaders.
        num_workers: Number of data-loading workers.
        train_transforms: Custom training transforms. Must include bbox_params.
        eval_transforms: Custom eval transforms. Must include bbox_params.
        augmentation_preset: Preset name (``"default"``, ``"strong"``).
            Ignored when train_transforms is provided.
        transform_config: Optional :class:`TransformConfig` for unified transform
            configuration. When provided along with ``backbone``, uses model-specific
            normalization from timm. Takes precedence over individual transform args.
        backbone: Optional backbone name or module. Used with ``transform_config``
            to resolve model-specific normalization (mean, std, input_size).
        pin_memory: Pin memory for GPU transfer.
        persistent_workers: Keep worker processes alive between epochs.
        prefetch_factor: Number of batches prefetched per worker.
        min_bbox_area: Minimum bbox area to include in training.
        class_ids: Optional list of class IDs to filter.
    """

    def __init__(
        self,
        data_dir: str | Path = "./coco",
        train_images_dir: str | Path | None = None,
        val_images_dir: str | Path | None = None,
        train_ann_file: str | Path | None = None,
        val_ann_file: str | Path | None = None,
        test_images_dir: str | Path | None = None,
        test_ann_file: str | Path | None = None,
        image_size: int = 640,
        batch_size: int = 16,
        num_workers: int = 4,
        train_transforms: Callable | None = None,
        eval_transforms: Callable | None = None,
        augmentation_preset: str = "default",
        transform_config: TransformConfig | None = None,
        backbone: str | nn.Module | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        min_bbox_area: float = 0.0,
        class_ids: list[int] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.min_bbox_area = min_bbox_area
        self.class_ids = class_ids

        # Set default paths
        self.train_images_dir = (
            Path(train_images_dir) if train_images_dir else self.data_dir / "train2017"
        )
        self.val_images_dir = (
            Path(val_images_dir) if val_images_dir else self.data_dir / "val2017"
        )
        self.train_ann_file = (
            Path(train_ann_file)
            if train_ann_file
            else self.data_dir / "annotations" / "instances_train2017.json"
        )
        self.val_ann_file = (
            Path(val_ann_file)
            if val_ann_file
            else self.data_dir / "annotations" / "instances_val2017.json"
        )
        self.test_images_dir = Path(test_images_dir) if test_images_dir else None
        self.test_ann_file = Path(test_ann_file) if test_ann_file else None
        self.transform_config = transform_config
        self.backbone = backbone

        # Resolve transforms - TransformConfig takes precedence
        if transform_config is not None and backbone is not None:
            from autotimm.data.timm_transforms import get_transforms_from_backbone

            self.train_transforms = get_transforms_from_backbone(
                backbone=backbone,
                transform_config=transform_config,
                is_train=True,
                task="detection",
            )
            self.eval_transforms = get_transforms_from_backbone(
                backbone=backbone,
                transform_config=transform_config,
                is_train=False,
                task="detection",
            )
        elif train_transforms is not None:
            self.train_transforms = train_transforms
        else:
            self.train_transforms = get_detection_transforms(
                preset=augmentation_preset,
                image_size=image_size,
                is_train=True,
            )

        if eval_transforms is not None:
            self.eval_transforms = eval_transforms
        else:
            self.eval_transforms = detection_eval_transforms(image_size=image_size)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes: int | None = None
        self.class_names: list[str] | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = COCODetectionDataset(
                images_dir=self.train_images_dir,
                annotations_file=self.train_ann_file,
                transform=self.train_transforms,
                min_bbox_area=self.min_bbox_area,
                class_ids=self.class_ids,
            )
            self.val_dataset = COCODetectionDataset(
                images_dir=self.val_images_dir,
                annotations_file=self.val_ann_file,
                transform=self.eval_transforms,
                min_bbox_area=0.0,  # Keep all boxes for evaluation
                class_ids=self.class_ids,
            )
            self.num_classes = self.train_dataset.num_classes
            self.class_names = self.train_dataset.class_names

        if stage in ("test", None) and self.test_images_dir and self.test_ann_file:
            self.test_dataset = COCODetectionDataset(
                images_dir=self.test_images_dir,
                annotations_file=self.test_ann_file,
                transform=self.eval_transforms,
                min_bbox_area=0.0,
                class_ids=self.class_ids,
            )
            if self.num_classes is None:
                self.num_classes = self.test_dataset.num_classes
                self.class_names = self.test_dataset.class_names

        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = COCODetectionDataset(
                images_dir=self.val_images_dir,
                annotations_file=self.val_ann_file,
                transform=self.eval_transforms,
                min_bbox_area=0.0,
                class_ids=self.class_ids,
            )
            self.num_classes = self.val_dataset.num_classes
            self.class_names = self.val_dataset.class_names

        # Automatically print data summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print data summary automatically after setup."""
        try:
            from rich.console import Console

            console = Console()
            console.print(self.summary())
        except ImportError:
            # Fallback to basic print if rich is not available
            print(f"\n{'='*50}")
            print("DetectionDataModule Summary")
            print(f"{'='*50}")
            print(f"Data dir: {self.data_dir}")
            print(f"Image size: {self.image_size}")
            print(f"Batch size: {self.batch_size}")
            if self.num_classes is not None:
                print(f"Num classes: {self.num_classes}")
            if self.train_dataset is not None:
                print(f"Train images: {len(self.train_dataset)}")
            if self.val_dataset is not None:
                print(f"Val images: {len(self.val_dataset)}")
            if self.test_dataset is not None:
                print(f"Test images: {len(self.test_dataset)}")
            print(f"{'='*50}\n")
        except Exception:
            # Silently ignore any errors in summary printing
            pass

    def _loader_kwargs(self) -> dict:
        kwargs: dict = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": detection_collate_fn,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "No test dataset configured. Provide test_images_dir and test_ann_file."
            )
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def summary(self):
        """Return a Rich Table summarizing the data module.

        Call after ``setup()`` so that datasets and class info are available.
        """
        from rich.table import Table

        table = Table(title="DetectionDataModule Summary", show_lines=True)
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Data dir", str(self.data_dir))
        table.add_row("Image size", str(self.image_size))
        table.add_row("Batch size", str(self.batch_size))
        table.add_row("Num workers", str(self.num_workers))

        if self.num_classes is not None:
            table.add_row("Num classes", str(self.num_classes))

        if self.train_dataset is not None:
            table.add_row("Train images", str(len(self.train_dataset)))
        if self.val_dataset is not None:
            table.add_row("Val images", str(len(self.val_dataset)))
        if self.test_dataset is not None:
            table.add_row("Test images", str(len(self.test_dataset)))

        if self.class_names is not None and len(self.class_names) <= 20:
            table.add_row("Classes", ", ".join(self.class_names))

        return table
