"""Tests for backbone creation and discovery."""

import pytest
import torch.nn as nn

from autotimm.backbone import (
    BackboneConfig,
    create_backbone,
    get_backbone_out_features,
    list_backbones,
)


def test_create_backbone_from_string():
    model = create_backbone("resnet18")
    assert isinstance(model, nn.Module)
    assert get_backbone_out_features(model) == 512


def test_create_backbone_from_config():
    cfg = BackboneConfig(model_name="resnet18", pretrained=False, num_classes=0)
    model = create_backbone(cfg)
    assert isinstance(model, nn.Module)
    assert get_backbone_out_features(model) == 512


def test_create_backbone_invalid_name():
    with pytest.raises(ValueError, match="not found in timm"):
        create_backbone("totally_fake_model_xyz")


def test_list_backbones_returns_list():
    models = list_backbones("resnet*")
    assert isinstance(models, list)
    assert len(models) > 0
    assert any("resnet" in m for m in models)


def test_list_backbones_empty_pattern():
    models = list_backbones()
    assert len(models) > 0
