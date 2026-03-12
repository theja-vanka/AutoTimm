"""Backbone creation and discovery via timm.

This module has moved to ``autotimm.core.backbone``.
This file is a backward-compatible re-export stub.
"""

from autotimm.core.backbone import *  # noqa: F401,F403
from autotimm.core.backbone import (  # explicit re-exports for type checkers
    BackboneConfig,
    FeatureBackboneConfig,
    ModelSource,
    create_backbone,
    create_feature_backbone,
    get_backbone_out_features,
    get_feature_channels,
    get_feature_info,
    get_feature_strides,
    get_model_source,
    list_backbones,
    list_hf_hub_backbones,
    _is_hf_hub_model,
)
