"""Lightning DataModule for instance segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from autotimm.data.instance_dataset import (
    COCOInstanceDataset,
    collate_instance_segmentation,
)
from autotimm.data.segmentation_transforms import instance_segmentation_transforms


class InstanceSegmentationDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for instance segmentation.

    Args:
        data_dir: Root directory of COCO dataset
        image_size: Target image size (square)
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        augmentation_preset: Augmentation strength ('default', 'strong', 'light')
        custom_train_transforms: Optional custom training transforms
        custom_val_transforms: Optional custom validation transforms
        min_keypoints: Minimum number of keypoints for an instance to be valid
        min_area: Minimum area for an instance to be valid
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 640,
        batch_size: int = 4,
        num_workers: int = 4,
        augmentation_preset: str = "default",
        custom_train_transforms: Any = None,
        custom_val_transforms: Any = None,
        min_keypoints: int = 0,
        min_area: float = 0.0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_preset = augmentation_preset
        self.custom_train_transforms = custom_train_transforms
        self.custom_val_transforms = custom_val_transforms
        self.min_keypoints = min_keypoints
        self.min_area = min_area

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
            train_transforms = instance_segmentation_transforms(
                image_size=self.image_size,
                train=True,
            )

        if self.custom_val_transforms is not None:
            val_transforms = self.custom_val_transforms
        else:
            val_transforms = instance_segmentation_transforms(
                image_size=self.image_size,
                train=False,
            )

        # Setup datasets based on stage
        if stage == "fit" or stage is None:
            self.train_dataset = COCOInstanceDataset(
                data_dir=self.data_dir,
                split="train",
                transforms=train_transforms,
                min_keypoints=self.min_keypoints,
                min_area=self.min_area,
            )

            # Try to load validation set
            try:
                self.val_dataset = COCOInstanceDataset(
                    data_dir=self.data_dir,
                    split="val",
                    transforms=val_transforms,
                    min_keypoints=self.min_keypoints,
                    min_area=self.min_area,
                )
            except FileNotFoundError:
                # No validation set available
                self.val_dataset = None

        if stage == "validate":
            self.val_dataset = COCOInstanceDataset(
                data_dir=self.data_dir,
                split="val",
                transforms=val_transforms,
                min_keypoints=self.min_keypoints,
                min_area=self.min_area,
            )

        if stage == "test":
            self.test_dataset = COCOInstanceDataset(
                data_dir=self.data_dir,
                split="test",
                transforms=val_transforms,
                min_keypoints=self.min_keypoints,
                min_area=self.min_area,
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
            collate_fn=collate_instance_segmentation,
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
            collate_fn=collate_instance_segmentation,
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
            collate_fn=collate_instance_segmentation,
            pin_memory=True,
        )
