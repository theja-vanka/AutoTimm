"""Backbone creation and discovery via timm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import timm
import torch.nn as nn


@dataclass
class BackboneConfig:
    """Configuration for a timm backbone.

    Attributes:
        model_name: Name recognized by ``timm.create_model`` (e.g. ``"resnet50"``).
        pretrained: Whether to load pretrained weights.
        num_classes: Set to 0 to get the feature extractor without the
            classification head (the head is provided separately by the task).
        drop_rate: Dropout rate applied inside the model.
        drop_path_rate: Stochastic depth rate.
        extra_kwargs: Forwarded verbatim to ``timm.create_model``.
    """

    model_name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def create_backbone(cfg: BackboneConfig | str) -> nn.Module:
    """Create a timm backbone from a config or model name string.

    When *cfg* is a plain string it is treated as a model name with
    ``pretrained=True`` and ``num_classes=0`` (headless).

    Raises ``ValueError`` if the model name is not found in timm.
    """
    if isinstance(cfg, str):
        cfg = BackboneConfig(model_name=cfg)

    available = timm.list_models()
    if cfg.model_name not in available:
        raise ValueError(
            f"Model '{cfg.model_name}' not found in timm. "
            f"Use autotimm.list_backbones('*pattern*') to search."
        )

    return timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        **cfg.extra_kwargs,
    )


def get_backbone_out_features(backbone: nn.Module) -> int:
    """Return the number of output features from a headless timm backbone."""
    return backbone.num_features


def list_backbones(pattern: str = "", pretrained_only: bool = False) -> list[str]:
    """List available timm backbones, optionally filtered by glob *pattern*.

    Examples::

        list_backbones("*resnet*")
        list_backbones("*efficientnet*", pretrained_only=True)
    """
    return timm.list_models(pattern or "*", pretrained=pretrained_only)


@dataclass
class FeatureBackboneConfig:
    """Configuration for a timm backbone with multi-scale feature extraction.

    Used for tasks like object detection that require feature maps at multiple
    scales (e.g., C2, C3, C4, C5 from ResNet).

    Attributes:
        model_name: Name recognized by ``timm.create_model`` (e.g. ``"resnet50"``).
        pretrained: Whether to load pretrained weights.
        out_indices: Tuple of stage indices to extract features from.
            Default (1, 2, 3, 4) extracts C2, C3, C4, C5 for most architectures.
        drop_rate: Dropout rate applied inside the model.
        drop_path_rate: Stochastic depth rate.
        extra_kwargs: Forwarded verbatim to ``timm.create_model``.
    """

    model_name: str = "resnet50"
    pretrained: bool = True
    out_indices: tuple[int, ...] = (1, 2, 3, 4)
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def create_feature_backbone(cfg: FeatureBackboneConfig | str) -> nn.Module:
    """Create a timm backbone for multi-scale feature extraction.

    Uses timm's ``features_only=True`` mode to extract intermediate feature maps,
    typically used for object detection, segmentation, etc.

    When *cfg* is a plain string it is treated as a model name with
    ``pretrained=True`` and default ``out_indices=(1, 2, 3, 4)``.

    Returns a model that outputs a list of feature tensors, one per stage.

    Raises ``ValueError`` if the model name is not found in timm.
    """
    if isinstance(cfg, str):
        cfg = FeatureBackboneConfig(model_name=cfg)

    available = timm.list_models()
    if cfg.model_name not in available:
        raise ValueError(
            f"Model '{cfg.model_name}' not found in timm. "
            f"Use autotimm.list_backbones('*pattern*') to search."
        )

    return timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        features_only=True,
        out_indices=cfg.out_indices,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        **cfg.extra_kwargs,
    )


def get_feature_info(backbone: nn.Module) -> list[dict[str, Any]]:
    """Return feature info for a backbone created with features_only=True.

    Each dict contains:
        - ``num_chs``: Number of output channels
        - ``reduction``: Spatial reduction factor (stride) relative to input
        - ``module``: Name of the module producing this feature

    Raises ``AttributeError`` if the backbone was not created with features_only=True.
    """
    if not hasattr(backbone, "feature_info"):
        raise AttributeError(
            "Backbone does not have 'feature_info'. "
            "Ensure it was created with features_only=True."
        )
    return [info for info in backbone.feature_info]


def get_feature_channels(backbone: nn.Module) -> list[int]:
    """Return the number of channels for each feature level.

    Convenience function that extracts just the channel counts from feature_info.
    """
    info = get_feature_info(backbone)
    return [f["num_chs"] for f in info]


def get_feature_strides(backbone: nn.Module) -> list[int]:
    """Return the stride (spatial reduction) for each feature level.

    Convenience function that extracts just the strides from feature_info.
    """
    info = get_feature_info(backbone)
    return [f["reduction"] for f in info]
