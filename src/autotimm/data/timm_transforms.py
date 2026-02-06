"""Utilities for creating transforms using timm's model-specific data config."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import transforms

from autotimm.data.transform_config import TransformConfig
from autotimm.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def resolve_backbone_data_config(
    backbone: str | nn.Module,
    override_mean: tuple[float, float, float] | None = None,
    override_std: tuple[float, float, float] | None = None,
    override_input_size: tuple[int, int, int] | None = None,
    override_interpolation: str | None = None,
    override_crop_pct: float | None = None,
) -> dict[str, Any]:
    """Get model-specific preprocessing config from a timm backbone.

    Uses timm's ``resolve_model_data_config()`` to extract the normalization
    statistics and input specifications that the model was pretrained with.

    Args:
        backbone: Either a model name string or an instantiated nn.Module.
            If a string, the config is resolved from timm's model registry.
            If a module, attempts to extract config from the model itself.
        override_mean: Override the normalization mean.
        override_std: Override the normalization std.
        override_input_size: Override the input size as (C, H, W).
        override_interpolation: Override the interpolation mode.
        override_crop_pct: Override the center crop percentage.

    Returns:
        Dictionary with keys:
            - ``mean``: Tuple of 3 floats for channel normalization mean.
            - ``std``: Tuple of 3 floats for channel normalization std.
            - ``input_size``: Tuple (C, H, W) for expected input dimensions.
            - ``interpolation``: String interpolation mode (e.g., "bicubic").
            - ``crop_pct``: Float center crop percentage for eval.

    Example:
        >>> config = resolve_backbone_data_config("efficientnet_b0")
        >>> print(config["mean"])  # (0.485, 0.456, 0.406)
        >>> print(config["input_size"])  # (3, 224, 224)
    """
    import timm.data

    # Get base config from timm
    if isinstance(backbone, str):
        # Resolve from model name
        try:
            data_config = timm.data.resolve_model_data_config(model=backbone)
        except Exception:
            # Fallback to ImageNet defaults if model not found
            data_config = {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "input_size": (3, 224, 224),
                "interpolation": "bicubic",
                "crop_pct": 0.875,
            }
    else:
        # Extract from model instance
        try:
            data_config = timm.data.resolve_model_data_config(model=backbone)
        except Exception:
            # Fallback to ImageNet defaults
            data_config = {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "input_size": (3, 224, 224),
                "interpolation": "bicubic",
                "crop_pct": 0.875,
            }

    # Apply overrides
    result = {
        "mean": override_mean
        if override_mean is not None
        else tuple(data_config.get("mean", IMAGENET_MEAN)),
        "std": override_std
        if override_std is not None
        else tuple(data_config.get("std", IMAGENET_STD)),
        "input_size": override_input_size
        if override_input_size is not None
        else tuple(data_config.get("input_size", (3, 224, 224))),
        "interpolation": override_interpolation
        if override_interpolation is not None
        else data_config.get("interpolation", "bicubic"),
        "crop_pct": override_crop_pct
        if override_crop_pct is not None
        else data_config.get("crop_pct", 0.875),
    }

    return result


def _get_interpolation_mode(interpolation: str) -> transforms.InterpolationMode:
    """Convert string interpolation mode to torchvision InterpolationMode."""
    mode_map = {
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "lanczos": transforms.InterpolationMode.LANCZOS,
        "nearest": transforms.InterpolationMode.NEAREST,
    }
    return mode_map.get(interpolation.lower(), transforms.InterpolationMode.BICUBIC)


def get_transforms_from_backbone(
    backbone: str | nn.Module,
    transform_config: TransformConfig,
    is_train: bool = False,
    task: str = "classification",
) -> Any:
    """Create transforms using model-specific normalization.

    This function combines the user's TransformConfig preferences with
    model-specific data configuration from timm to create appropriate
    transforms for training or evaluation.

    Args:
        backbone: Model name or nn.Module to get data config from.
        transform_config: TransformConfig specifying transform preferences.
        is_train: Whether to create training transforms (with augmentation)
            or evaluation transforms (minimal transforms).
        task: Task type - "classification", "detection", or "segmentation".

    Returns:
        A callable transform pipeline. The type depends on the backend:
            - torchvision: ``transforms.Compose``
            - albumentations: ``albumentations.Compose``

    Example:
        >>> config = TransformConfig(preset="randaugment", image_size=384)
        >>> train_transforms = get_transforms_from_backbone(
        ...     "efficientnet_b4", config, is_train=True
        ... )
    """
    # Resolve data config from backbone
    data_config = resolve_backbone_data_config(
        backbone,
        override_mean=transform_config.mean
        if not transform_config.use_timm_config
        else None,
        override_std=transform_config.std
        if not transform_config.use_timm_config
        else None,
        override_interpolation=transform_config.interpolation,
        override_crop_pct=transform_config.crop_pct,
    )

    # Use config overrides if use_timm_config is False
    mean = transform_config.mean if transform_config.mean else data_config["mean"]
    std = transform_config.std if transform_config.std else data_config["std"]
    image_size = transform_config.image_size
    interpolation = transform_config.interpolation
    crop_pct = transform_config.crop_pct

    if transform_config.backend == "torchvision":
        return _create_torchvision_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            interpolation=interpolation,
            crop_pct=crop_pct,
            preset=transform_config.preset,
            is_train=is_train,
            randaugment_num_ops=transform_config.randaugment_num_ops,
            randaugment_magnitude=transform_config.randaugment_magnitude,
        )
    else:
        return _create_albumentations_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            preset=transform_config.preset,
            is_train=is_train,
            task=task,
            ignore_index=transform_config.ignore_index,
            min_bbox_area=transform_config.min_bbox_area,
            min_visibility=transform_config.min_visibility,
            bbox_format=transform_config.bbox_format,
        )


def _create_torchvision_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    interpolation: str,
    crop_pct: float,
    preset: str,
    is_train: bool,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
) -> transforms.Compose:
    """Create torchvision transforms with specified normalization."""
    interp_mode = _get_interpolation_mode(interpolation)

    if is_train:
        # Training transforms
        base_transforms = [
            transforms.RandomResizedCrop(
                image_size,
                interpolation=interp_mode,
            ),
            transforms.RandomHorizontalFlip(),
        ]

        # Add augmentation based on preset
        if preset == "autoaugment":
            base_transforms.append(
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
            )
        elif preset == "randaugment":
            base_transforms.append(
                transforms.RandAugment(
                    num_ops=randaugment_num_ops,
                    magnitude=randaugment_magnitude,
                )
            )
        elif preset == "trivialaugment":
            base_transforms.append(transforms.TrivialAugmentWide())
        elif preset == "default":
            base_transforms.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            )
        elif preset == "light":
            pass  # No additional augmentation
        # For unknown presets, skip additional augmentation

        base_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Evaluation transforms
        resize_size = int(image_size / crop_pct)
        base_transforms = [
            transforms.Resize(resize_size, interpolation=interp_mode),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(base_transforms)


def _create_albumentations_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    preset: str,
    is_train: bool,
    task: str = "classification",
    ignore_index: int = 255,
    min_bbox_area: float = 0.0,
    min_visibility: float = 0.0,
    bbox_format: str = "coco",
) -> Any:
    """Create albumentations transforms with specified normalization."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError(
            "Albumentations is required for this transform backend. "
            "Install with: pip install autotimm[albumentations]"
        )

    # Configure bbox params for detection tasks
    bbox_params = None
    if task == "detection":
        bbox_params = A.BboxParams(
            format=bbox_format,
            min_area=min_bbox_area,
            min_visibility=min_visibility,
            label_fields=["labels"],
        )

    if is_train:
        if preset == "strong":
            transform_list = [
                A.RandomResizedCrop(size=(image_size, image_size)),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent=(-0.1, 0.1),
                    scale=(0.8, 1.2),
                    rotate=(-15, 15),
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=(3, 5)),
                        A.GaussNoise(std_range=(0.02, 0.05)),
                    ],
                    p=0.3,
                ),
                A.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 3),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    p=0.3,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        elif preset == "light":
            transform_list = [
                A.RandomResizedCrop(size=(image_size, image_size)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        else:  # default
            transform_list = [
                A.RandomResizedCrop(size=(image_size, image_size)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.8),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
    else:
        # Evaluation transforms
        resize_size = int(image_size * 256 / 224)  # Standard resize ratio
        transform_list = [
            A.Resize(height=resize_size, width=resize_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]

    return A.Compose(transform_list, bbox_params=bbox_params)


def create_inference_transform(
    backbone: str | nn.Module,
    transform_config: TransformConfig | None = None,
) -> transforms.Compose:
    """Create a simple inference transform using model-specific normalization.

    This is a convenience function for creating transforms suitable for
    single-image inference. It creates evaluation-mode transforms with
    the correct normalization for the given backbone.

    Args:
        backbone: Model name or nn.Module to get data config from.
        transform_config: Optional TransformConfig. If None, uses defaults
            with ``use_timm_config=True``.

    Returns:
        A torchvision transforms.Compose suitable for inference.

    Example:
        >>> transform = create_inference_transform("resnet50")
        >>> tensor = transform(pil_image)
    """
    if transform_config is None:
        transform_config = TransformConfig(use_timm_config=True)

    # Always use torchvision for simple inference
    config = transform_config.with_overrides(backend="torchvision")

    return get_transforms_from_backbone(
        backbone=backbone,
        transform_config=config,
        is_train=False,
        task="classification",
    )
