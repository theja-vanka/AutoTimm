"""Lightning DataModule for semantic segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader

from autotimm.data.segmentation_dataset import (
    SemanticSegmentationDataset,
    collate_segmentation,
)
from autotimm.data.segmentation_transforms import (
    get_segmentation_preset,
    segmentation_eval_transforms,
)
from autotimm.data.transform_config import TransformConfig


class SegmentationDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for semantic segmentation.

    Args:
        data_dir: Root directory of the dataset
        format: Dataset format ('png', 'coco', 'cityscapes', 'voc', 'csv')
        image_size: Target image size (square)
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        augmentation_preset: Augmentation preset ('default', 'strong', 'light')
        custom_train_transforms: Optional custom training transforms
        custom_val_transforms: Optional custom validation transforms
        class_mapping: Optional mapping from dataset class IDs to contiguous IDs
        ignore_index: Index to use for ignored pixels (default: 255)
        transform_config: Optional TransformConfig for unified transform configuration.
            When provided along with backbone, uses model-specific normalization.
        backbone: Optional backbone name or module for model-specific normalization.
        train_csv: Path to training CSV file (used when format='csv').
        val_csv: Path to validation CSV file (used when format='csv').
        test_csv: Path to test CSV file (used when format='csv').
    """

    def __init__(
        self,
        data_dir: str | Path,
        format: str = "png",
        image_size: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
        augmentation_preset: str = "default",
        custom_train_transforms: Any = None,
        custom_val_transforms: Any = None,
        class_mapping: dict[int, int] | None = None,
        ignore_index: int = 255,
        transform_config: TransformConfig | None = None,
        backbone: str | nn.Module | None = None,
        train_csv: str | Path | None = None,
        val_csv: str | Path | None = None,
        test_csv: str | Path | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.format = format
        self.train_csv = Path(train_csv) if train_csv else None
        self.val_csv = Path(val_csv) if val_csv else None
        self.test_csv = Path(test_csv) if test_csv else None
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_preset = augmentation_preset
        self.custom_train_transforms = custom_train_transforms
        self.custom_val_transforms = custom_val_transforms
        self.class_mapping = class_mapping
        self.ignore_index = ignore_index
        self.transform_config = transform_config
        self.backbone = backbone

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        """Setup datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Get transforms - TransformConfig takes precedence
        if self.transform_config is not None and self.backbone is not None:
            from autotimm.data.timm_transforms import get_transforms_from_backbone

            train_transforms = get_transforms_from_backbone(
                backbone=self.backbone,
                transform_config=self.transform_config,
                is_train=True,
                task="segmentation",
            )
            val_transforms = get_transforms_from_backbone(
                backbone=self.backbone,
                transform_config=self.transform_config,
                is_train=False,
                task="segmentation",
            )
        elif self.custom_train_transforms is not None:
            train_transforms = self.custom_train_transforms
            if self.custom_val_transforms is not None:
                val_transforms = self.custom_val_transforms
            else:
                val_transforms = segmentation_eval_transforms(
                    image_size=self.image_size
                )
        else:
            train_transforms = get_segmentation_preset(
                preset=self.augmentation_preset,
                image_size=self.image_size,
                train=True,
            )
            if self.custom_val_transforms is not None:
                val_transforms = self.custom_val_transforms
            else:
                val_transforms = segmentation_eval_transforms(
                    image_size=self.image_size
                )

        # Setup datasets based on stage
        if stage == "fit" or stage is None:
            self.train_dataset = SemanticSegmentationDataset(
                data_dir=self.data_dir,
                split="train",
                format=self.format,
                image_size=self.image_size,
                transforms=train_transforms,
                class_mapping=self.class_mapping,
                ignore_index=self.ignore_index,
                csv_path=self.train_csv,
            )

            # Try to load validation set
            try:
                self.val_dataset = SemanticSegmentationDataset(
                    data_dir=self.data_dir,
                    split="val",
                    format=self.format,
                    image_size=self.image_size,
                    transforms=val_transforms,
                    class_mapping=self.class_mapping,
                    ignore_index=self.ignore_index,
                    csv_path=self.val_csv,
                )
            except (FileNotFoundError, ValueError):
                # No validation set available
                self.val_dataset = None

        if stage == "validate":
            self.val_dataset = SemanticSegmentationDataset(
                data_dir=self.data_dir,
                split="val",
                format=self.format,
                image_size=self.image_size,
                transforms=val_transforms,
                class_mapping=self.class_mapping,
                ignore_index=self.ignore_index,
                csv_path=self.val_csv,
            )

        if stage == "test":
            self.test_dataset = SemanticSegmentationDataset(
                data_dir=self.data_dir,
                split="test",
                format=self.format,
                image_size=self.image_size,
                transforms=val_transforms,
                class_mapping=self.class_mapping,
                ignore_index=self.ignore_index,
                csv_path=self.test_csv,
            )

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
            print(f"\n{'=' * 50}")
            print("SegmentationDataModule Summary")
            print(f"{'=' * 50}")
            print(f"Data dir: {self.data_dir}")
            print(f"Format: {self.format}")
            print(f"Image size: {self.image_size}")
            print(f"Batch size: {self.batch_size}")
            if self.train_dataset is not None:
                print(f"Train samples: {len(self.train_dataset)}")
            if self.val_dataset is not None:
                print(f"Val samples: {len(self.val_dataset)}")
            if self.test_dataset is not None:
                print(f"Test samples: {len(self.test_dataset)}")
            print(f"{'=' * 50}\n")
        except Exception:
            # Silently ignore any errors in summary printing
            pass

    def summary(self):
        """Return a Rich Table summarizing the data module.

        Call after ``setup()`` so that datasets are available.
        """
        from rich.table import Table

        table = Table(title="SegmentationDataModule Summary", show_lines=True)
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Data dir", str(self.data_dir))
        table.add_row("Format", str(self.format))
        table.add_row("Image size", str(self.image_size))
        table.add_row("Batch size", str(self.batch_size))
        table.add_row("Num workers", str(self.num_workers))
        table.add_row("Augmentation", str(self.augmentation_preset))
        table.add_row("Ignore index", str(self.ignore_index))

        if self.train_dataset is not None:
            table.add_row("Train samples", str(len(self.train_dataset)))
        if self.val_dataset is not None:
            table.add_row("Val samples", str(len(self.val_dataset)))
        if self.test_dataset is not None:
            table.add_row("Test samples", str(len(self.test_dataset)))

        return table

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_segmentation,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_segmentation,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_segmentation,
            pin_memory=True,
        )
