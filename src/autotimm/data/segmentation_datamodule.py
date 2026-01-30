"""Lightning DataModule for semantic segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from autotimm.data.segmentation_dataset import (
    SemanticSegmentationDataset,
    collate_segmentation,
)
from autotimm.data.segmentation_transforms import (
    get_segmentation_preset,
    segmentation_eval_transforms,
)


class SegmentationDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for semantic segmentation.

    Args:
        data_dir: Root directory of the dataset
        format: Dataset format ('png', 'coco', 'cityscapes', 'voc')
        image_size: Target image size (square)
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        augmentation_preset: Augmentation preset ('default', 'strong', 'light')
        custom_train_transforms: Optional custom training transforms
        custom_val_transforms: Optional custom validation transforms
        class_mapping: Optional mapping from dataset class IDs to contiguous IDs
        ignore_index: Index to use for ignored pixels (default: 255)
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
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.format = format
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_preset = augmentation_preset
        self.custom_train_transforms = custom_train_transforms
        self.custom_val_transforms = custom_val_transforms
        self.class_mapping = class_mapping
        self.ignore_index = ignore_index

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        """Setup datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Get transforms
        if self.custom_train_transforms is not None:
            train_transforms = self.custom_train_transforms
        else:
            train_transforms = get_segmentation_preset(
                preset=self.augmentation_preset,
                image_size=self.image_size,
                train=True,
            )

        if self.custom_val_transforms is not None:
            val_transforms = self.custom_val_transforms
        else:
            val_transforms = segmentation_eval_transforms(image_size=self.image_size)

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
                )
            except FileNotFoundError:
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
            )

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
