"""Datasets for semantic segmentation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SemanticSegmentationDataset(Dataset):
    """Generic semantic segmentation dataset.

    Supports multiple formats:
    - PNG format: images_dir + masks_dir with matching filenames
    - COCO format: COCO stuff/panoptic annotations
    - Cityscapes format: images + labelIds masks
    - Pascal VOC format: JPEGImages + SegmentationClass
    - CSV format: CSV file with image_path,mask_path columns

    Args:
        data_dir: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
        format: Dataset format ('png', 'coco', 'cityscapes', 'voc', 'csv')
        image_size: Target image size (resizes if provided)
        transforms: Albumentations transforms to apply
        class_mapping: Optional mapping from dataset class IDs to contiguous IDs
        ignore_index: Index to use for ignored pixels (default: 255)
        csv_path: Path to CSV file (required when format='csv')
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        format: str = "png",
        image_size: int | None = None,
        transforms: Any = None,
        class_mapping: dict[int, int] | None = None,
        ignore_index: int = 255,
        csv_path: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.format = format
        self.image_size = image_size
        self.transforms = transforms
        self.class_mapping = class_mapping
        self.ignore_index = ignore_index
        self.csv_path = Path(csv_path) if csv_path else None

        # Load dataset samples
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict]:
        """Load dataset samples based on format."""
        if self.format == "png":
            return self._load_png_format()
        elif self.format == "coco":
            return self._load_coco_format()
        elif self.format == "cityscapes":
            return self._load_cityscapes_format()
        elif self.format == "voc":
            return self._load_voc_format()
        elif self.format == "csv":
            return self._load_csv_format()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _load_png_format(self) -> list[dict]:
        """Load PNG format: images_dir + masks_dir with matching filenames."""
        images_dir = self.data_dir / self.split / "images"
        masks_dir = self.data_dir / self.split / "masks"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        samples = []
        for image_path in sorted(images_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # Find corresponding mask
            mask_path = masks_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                # Try with same extension
                mask_path = masks_dir / image_path.name
                if not mask_path.exists():
                    continue

            samples.append(
                {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image_id": image_path.stem,
                }
            )

        return samples

    def _load_coco_format(self) -> list[dict]:
        """Load COCO stuff/panoptic format."""
        annotations_file = self.data_dir / "annotations" / f"stuff_{self.split}.json"
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

        with open(annotations_file) as f:
            coco_data = json.load(f)

        images_dir = self.data_dir / self.split

        samples = []
        for img_info in coco_data["images"]:
            image_path = images_dir / img_info["file_name"]
            # Assume masks are in separate directory
            mask_path = self.data_dir / f"{self.split}_masks" / f"{img_info['id']}.png"

            samples.append(
                {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                    "image_id": img_info["id"],
                }
            )

        return samples

    def _load_cityscapes_format(self) -> list[dict]:
        """Load Cityscapes format: leftImg8bit + gtFine."""
        images_dir = self.data_dir / "leftImg8bit" / self.split
        masks_dir = self.data_dir / "gtFine" / self.split

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        samples = []
        for city_dir in sorted(images_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            for image_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                # Find corresponding labelIds mask
                city_name = city_dir.name
                base_name = image_path.stem.replace("_leftImg8bit", "")
                mask_path = masks_dir / city_name / f"{base_name}_gtFine_labelIds.png"

                if mask_path.exists():
                    samples.append(
                        {
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "image_id": base_name,
                        }
                    )

        return samples

    def _load_voc_format(self) -> list[dict]:
        """Load Pascal VOC format."""
        images_dir = self.data_dir / "JPEGImages"
        masks_dir = self.data_dir / "SegmentationClass"
        split_file = self.data_dir / "ImageSets" / "Segmentation" / f"{self.split}.txt"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        # Load image IDs from split file
        if split_file.exists():
            with open(split_file) as f:
                image_ids = [line.strip() for line in f if line.strip()]
        else:
            # If no split file, use all images
            image_ids = [p.stem for p in sorted(images_dir.glob("*.jpg"))]

        samples = []
        for image_id in image_ids:
            image_path = images_dir / f"{image_id}.jpg"
            mask_path = masks_dir / f"{image_id}.png"

            if image_path.exists() and mask_path.exists():
                samples.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "image_id": image_id,
                    }
                )

        return samples

    def _load_csv_format(self) -> list[dict]:
        """Load CSV format: image_path,mask_path columns."""
        if self.csv_path is None:
            raise ValueError("csv_path is required when format='csv'")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        samples = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

            # Determine column names
            img_col = "image_path" if "image_path" in fieldnames else fieldnames[0]
            mask_col = "mask_path" if "mask_path" in fieldnames else fieldnames[1]

            for row in reader:
                image_path = self.data_dir / row[img_col]
                mask_path = self.data_dir / row[mask_col]

                samples.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "image_id": Path(row[img_col]).stem,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dict with keys:
            - image: [C, H, W] tensor
            - mask: [H, W] tensor with class indices
            - image_id: str or int
            - orig_size: [2] tensor (H, W)
        """
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        if sample.get("mask_path"):
            mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        else:
            # Create empty mask if not available
            mask = np.full(image.shape[:2], self.ignore_index, dtype=np.uint8)

        # Apply class mapping if provided
        if self.class_mapping is not None:
            mask_mapped = np.full_like(mask, self.ignore_index)
            for src_class, tgt_class in self.class_mapping.items():
                mask_mapped[mask == src_class] = tgt_class
            mask = mask_mapped

        # Store original size
        orig_size = torch.tensor(image.shape[:2], dtype=torch.long)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # Ensure mask is long tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).long()
        elif mask.dtype != torch.long:
            mask = mask.long()

        return {
            "image": image,
            "mask": mask,
            "image_id": sample["image_id"],
            "orig_size": orig_size,
        }


def collate_segmentation(batch: list[dict]) -> dict[str, Any]:
    """Collate function for segmentation datasets.

    Stacks images and masks (assumes they are the same size after transforms).

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dict with stacked tensors
    """
    images = torch.stack([sample["image"] for sample in batch])
    masks = torch.stack([sample["mask"] for sample in batch])
    image_ids = [sample["image_id"] for sample in batch]
    orig_sizes = torch.stack([sample["orig_size"] for sample in batch])

    return {
        "image": images,
        "mask": masks,
        "image_id": image_ids,
        "orig_size": orig_sizes,
    }
