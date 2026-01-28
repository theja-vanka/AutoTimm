"""OpenCV-backed image datasets for use with albumentations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from torch.utils.data import Dataset


class ImageFolderCV2(Dataset):
    """ImageFolder-style dataset that loads images with OpenCV.

    Expects the same directory layout as ``torchvision.datasets.ImageFolder``::

        root/
          class_a/
            img1.jpg
          class_b/
            img2.jpg

    Images are loaded as RGB numpy arrays (H, W, C) with dtype ``uint8``,
    which is the format expected by albumentations transforms.

    Parameters:
        root: Root directory containing class sub-folders.
        transform: Callable applied to the image array (e.g. an
            albumentations ``Compose``). Must return a dict with
            an ``"image"`` key containing a torch Tensor.
        extensions: Tuple of valid file extensions.
    """

    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        extensions: tuple[str, ...] | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions or self.VALID_EXTENSIONS

        self.classes: list[str] = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.samples: list[tuple[str, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), cls_idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in '{self.root}'. Expected sub-folders "
                f"with image files ({', '.join(self.extensions)})."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        import cv2

        path, target = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, target
