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
