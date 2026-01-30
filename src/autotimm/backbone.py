"""Backbone creation and discovery via timm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import timm
import torch.nn as nn

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


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


def _is_hf_hub_model(model_name: str) -> bool:
    """Check if a model name refers to a Hugging Face Hub model.

    HF Hub models start with 'hf-hub:', 'hf_hub:', or 'timm/' prefix.
    """
    return (
        model_name.startswith("hf-hub:")
        or model_name.startswith("hf_hub:")
        or model_name.startswith("timm/")
    )


def create_backbone(cfg: BackboneConfig | str) -> nn.Module:
    """Create a timm backbone from a config or model name string.

    When *cfg* is a plain string it is treated as a model name with
    ``pretrained=True`` and ``num_classes=0`` (headless).

    Supports both timm models and Hugging Face Hub models:
        - Timm models: Use the model name directly (e.g., ``"resnet50"``).
        - HF Hub models: Use the ``hf-hub:`` prefix (e.g., ``"hf-hub:timm/resnet50.a1_in1k"``).

    Examples::

        create_backbone("resnet50")  # Standard timm model
        create_backbone("hf-hub:timm/resnet50.a1_in1k")  # HF Hub model

    Raises ``ValueError`` if the model name is not found.
    """
    if isinstance(cfg, str):
        cfg = BackboneConfig(model_name=cfg)

    # Skip validation for HF Hub models as they're not in timm.list_models()
    if not _is_hf_hub_model(cfg.model_name):
        available = timm.list_models()
        if cfg.model_name not in available:
            raise ValueError(
                f"Model '{cfg.model_name}' not found in timm. "
                f"Use autotimm.list_backbones('*pattern*') to search, "
                f"or use 'hf-hub:' prefix for Hugging Face Hub models."
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

    This lists models available in the timm library. To search Hugging Face Hub
    models, use ``list_hf_hub_backbones()``.

    Examples::

        list_backbones("*resnet*")
        list_backbones("*efficientnet*", pretrained_only=True)
    """
    return timm.list_models(pattern or "*", pretrained=pretrained_only)


def list_hf_hub_backbones(
    author: str = "timm",
    model_name: str | None = None,
    limit: int = 50,
) -> list[str]:
    """List timm-compatible models available on Hugging Face Hub.

    Parameters:
        author: Filter by author/organization (default: ``"timm"`` for official timm models).
        model_name: Optional search query for model names.
        limit: Maximum number of results to return (default: 50).

    Returns:
        List of model identifiers in ``hf-hub:`` format that can be used directly
        with ``create_backbone()`` or ``create_feature_backbone()``.

    Examples::

        # List official timm models on HF Hub
        list_hf_hub_backbones()

        # Search for ResNet models
        list_hf_hub_backbones(model_name="resnet")

        # Search models from a specific author
        list_hf_hub_backbones(author="facebook", model_name="convnext")

    Note:
        Requires ``huggingface_hub`` to be installed.
    """
    if HfApi is None:
        raise ImportError(
            "huggingface_hub is required to list HF Hub models. "
            "Install it with: pip install huggingface_hub"
        )

    api = HfApi()

    # Build search query - combine author and model name
    search_query = None
    if author or model_name:
        parts = []
        if author:
            parts.append(author)
        if model_name:
            parts.append(model_name)
        search_query = " ".join(parts)

    try:
        # Search for models with timm filter
        models = api.list_models(
            filter="timm",
            author=author if author else None,
            search=search_query,
            limit=limit,
            sort="downloads",
        )

        # Format as hf-hub: strings
        result = []
        for model in models:
            model_id = model.id
            result.append(f"hf-hub:{model_id}")

        return result

    except Exception as e:
        raise RuntimeError(f"Failed to fetch models from Hugging Face Hub: {e}")


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

    Supports both timm models and Hugging Face Hub models:
        - Timm models: Use the model name directly (e.g., ``"resnet50"``).
        - HF Hub models: Use the ``hf-hub:`` prefix (e.g., ``"hf-hub:timm/resnet50.a1_in1k"``).

    Returns a model that outputs a list of feature tensors, one per stage.

    Examples::

        create_feature_backbone("resnet50")  # Standard timm model
        create_feature_backbone("hf-hub:timm/convnext_base.fb_in22k_ft_in1k")  # HF Hub model

    Raises ``ValueError`` if the model name is not found.
    """
    if isinstance(cfg, str):
        cfg = FeatureBackboneConfig(model_name=cfg)

    # Skip validation for HF Hub models as they're not in timm.list_models()
    if not _is_hf_hub_model(cfg.model_name):
        available = timm.list_models()
        if cfg.model_name not in available:
            raise ValueError(
                f"Model '{cfg.model_name}' not found in timm. "
                f"Use autotimm.list_backbones('*pattern*') to search, "
                f"or use 'hf-hub:' prefix for Hugging Face Hub models."
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
