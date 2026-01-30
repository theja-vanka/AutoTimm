"""Datasets for instance segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from pycocotools import mask as mask_utils

    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    mask_utils = None


class COCOInstanceDataset(Dataset):
    """COCO instance segmentation dataset.

    Loads COCO-format annotations with instance segmentation masks.

    Args:
        data_dir: Root directory of COCO dataset
        split: Dataset split ('train', 'val', 'test')
        transforms: Albumentations transforms to apply
        min_keypoints: Minimum number of keypoints for an instance to be valid
        min_area: Minimum area for an instance to be valid
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transforms: Any = None,
        min_keypoints: int = 0,
        min_area: float = 0.0,
    ):
        if not HAS_PYCOCOTOOLS:
            raise ImportError(
                "pycocotools is required for instance segmentation. "
                "Install it with: pip install pycocotools"
            )

        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms
        self.min_keypoints = min_keypoints
        self.min_area = min_area

        # Load COCO annotations
        ann_file = self.data_dir / "annotations" / f"instances_{split}2017.json"
        if not ann_file.exists():
            # Try without year suffix
            ann_file = self.data_dir / "annotations" / f"instances_{split}.json"

        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_file}")

        with open(ann_file) as f:
            self.coco_data = json.load(f)

        # Build index
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}

        # Create category ID to contiguous ID mapping
        category_ids = sorted(self.categories.keys())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(category_ids)}
        self.label_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_label.items()}

        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Filter images with valid annotations
        self.image_ids = []
        for img_id in self.images.keys():
            if img_id in self.img_to_anns:
                self.image_ids.append(img_id)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _decode_mask(
        self, segmentation: list | dict, height: int, width: int
    ) -> np.ndarray:
        """Decode COCO segmentation to binary mask.

        Args:
            segmentation: COCO segmentation (polygon or RLE)
            height: Image height
            width: Image width

        Returns:
            Binary mask [H, W]
        """
        if isinstance(segmentation, list):
            # Polygon format
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            # RLE format
            if isinstance(segmentation["counts"], list):
                # Uncompressed RLE
                rle = mask_utils.frPyObjects(segmentation, height, width)
            else:
                # Compressed RLE
                rle = segmentation
        else:
            raise ValueError(f"Unknown segmentation format: {type(segmentation)}")

        mask = mask_utils.decode(rle)
        return mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dict with keys:
            - image: [C, H, W] tensor
            - boxes: [N, 4] tensor in xyxy format
            - labels: [N] tensor with class indices
            - masks: [N, H, W] binary masks
            - image_id: int
            - orig_size: [2] tensor (H, W)
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.data_dir / f"{self.split}2017" / img_info["file_name"]
        if not img_path.exists():
            # Try without year suffix
            img_path = self.data_dir / self.split / img_info["file_name"]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])

        # Filter valid annotations
        boxes = []
        labels = []
        masks = []

        for ann in anns:
            # Skip if no segmentation
            if "segmentation" not in ann:
                continue

            # Skip if too small
            if ann.get("area", 0) < self.min_area:
                continue

            # Skip if not enough keypoints
            if ann.get("num_keypoints", 0) < self.min_keypoints:
                continue

            # Skip if iscrowd
            if ann.get("iscrowd", 0):
                continue

            # Get bbox in xyxy format
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            # Get label (convert to contiguous)
            cat_id = ann["category_id"]
            label = self.cat_id_to_label[cat_id]
            labels.append(label)

            # Decode mask
            mask = self._decode_mask(ann["segmentation"], height, width)
            masks.append(mask)

        # Convert to arrays
        if len(boxes) == 0:
            # No valid annotations - create dummy data
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            masks = np.zeros((0, height, width), dtype=np.uint8)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            masks = np.stack(masks, axis=0).astype(np.uint8)

        # Store original size
        orig_size = torch.tensor([height, width], dtype=torch.long)

        # Apply transforms
        if self.transforms:
            # Albumentations expects masks as list for instance segmentation
            mask_list = [masks[i] for i in range(len(masks))]

            transformed = self.transforms(
                image=image,
                masks=mask_list,
                bboxes=boxes,
                labels=labels,
            )

            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
            masks = transformed["masks"]

            # Convert back to array
            if len(masks) > 0:
                masks = np.stack(masks, axis=0)
            else:
                masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)

            # Convert to tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            masks = torch.from_numpy(masks).float()
        else:
            # Convert to tensors manually
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.from_numpy(boxes).float()
            labels = torch.from_numpy(labels).long()
            masks = torch.from_numpy(masks).float()

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": img_id,
            "orig_size": orig_size,
        }


def collate_instance_segmentation(batch: list[dict]) -> dict[str, Any]:
    """Collate function for instance segmentation datasets.

    Handles variable number of instances per image.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dict with lists for variable-length tensors
    """
    images = torch.stack([sample["image"] for sample in batch])
    boxes = [sample["boxes"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    masks = [sample["masks"] for sample in batch]
    image_ids = [sample["image_id"] for sample in batch]
    orig_sizes = torch.stack([sample["orig_size"] for sample in batch])

    return {
        "image": images,
        "boxes": boxes,
        "labels": labels,
        "masks": masks,
        "image_id": image_ids,
        "orig_size": orig_sizes,
    }
