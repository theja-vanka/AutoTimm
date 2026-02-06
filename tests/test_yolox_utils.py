"""Tests for YOLOX utility functions."""

import pytest

from autotimm import (
    get_yolox_model_info,
    list_yolox_backbones,
    list_yolox_heads,
    list_yolox_models,
    list_yolox_necks,
)


def test_list_yolox_models():
    """Test listing YOLOX models."""
    models = list_yolox_models()

    # Check all expected models are present
    expected = ["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"]
    assert models == expected

    # Check verbose output doesn't crash
    list_yolox_models(verbose=True)


def test_list_yolox_backbones():
    """Test listing YOLOX backbones."""
    backbones = list_yolox_backbones()

    # Check all expected backbones are present
    expected = [
        "csp_darknet_nano",
        "csp_darknet_tiny",
        "csp_darknet_s",
        "csp_darknet_m",
        "csp_darknet_l",
        "csp_darknet_x",
    ]
    assert backbones == expected

    # Check verbose output doesn't crash
    list_yolox_backbones(verbose=True)


def test_list_yolox_necks():
    """Test listing YOLOX necks."""
    necks = list_yolox_necks()

    # Check all expected necks are present
    expected = [
        "yolox_pafpn_nano",
        "yolox_pafpn_tiny",
        "yolox_pafpn_s",
        "yolox_pafpn_m",
        "yolox_pafpn_l",
        "yolox_pafpn_x",
    ]
    assert necks == expected

    # Check verbose output doesn't crash
    list_yolox_necks(verbose=True)


def test_list_yolox_heads():
    """Test listing YOLOX heads."""
    heads = list_yolox_heads()

    # Check expected head is present
    assert heads == ["yolox_head"]

    # Check verbose output doesn't crash
    list_yolox_heads(verbose=True)


@pytest.mark.parametrize(
    "model_name",
    ["yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x"],
)
def test_get_yolox_model_info(model_name):
    """Test getting YOLOX model info."""
    info = get_yolox_model_info(model_name)

    # Check all required fields are present
    assert "depth" in info
    assert "width" in info
    assert "backbone" in info
    assert "neck" in info
    assert "head" in info
    assert "backbone_channels" in info
    assert "neck_channels" in info
    assert "params" in info
    assert "flops" in info
    assert "mAP" in info
    assert "input_size" in info
    assert "description" in info

    # Check types
    assert isinstance(info["depth"], float)
    assert isinstance(info["width"], float)
    assert isinstance(info["backbone"], str)
    assert isinstance(info["neck"], str)
    assert isinstance(info["head"], str)
    assert isinstance(info["backbone_channels"], tuple)
    assert isinstance(info["neck_channels"], int)
    assert isinstance(info["params"], str)
    assert isinstance(info["flops"], str)
    assert isinstance(info["mAP"], float)
    assert isinstance(info["input_size"], int)
    assert isinstance(info["description"], str)


def test_get_yolox_model_info_invalid():
    """Test getting info for invalid model."""
    with pytest.raises(ValueError, match="Unknown YOLOX model"):
        get_yolox_model_info("yolox-invalid")


def test_yolox_model_consistency():
    """Test consistency between model info and components."""
    models = list_yolox_models()
    backbones = list_yolox_backbones()
    necks = list_yolox_necks()

    for model_name in models:
        info = get_yolox_model_info(model_name)

        # Check backbone exists
        assert info["backbone"] in backbones

        # Check neck exists
        assert info["neck"] in necks

        # Check head exists
        assert info["head"] == "yolox_head"


def test_yolox_s_model_info():
    """Test specific info for YOLOX-S."""
    info = get_yolox_model_info("yolox-s")

    assert info["depth"] == 0.33
    assert info["width"] == 0.50
    assert info["backbone"] == "csp_darknet_s"
    assert info["neck"] == "yolox_pafpn_s"
    assert info["backbone_channels"] == (128, 256, 512)
    assert info["neck_channels"] == 128
    assert info["params"] == "9.0M"
    assert info["flops"] == "26.8G"
    assert info["mAP"] == 40.5
    assert info["input_size"] == 640


def test_yolox_nano_model_info():
    """Test specific info for YOLOX-Nano."""
    info = get_yolox_model_info("yolox-nano")

    assert info["depth"] == 0.33
    assert info["width"] == 0.25
    assert info["backbone"] == "csp_darknet_nano"
    assert info["neck"] == "yolox_pafpn_nano"
    assert info["backbone_channels"] == (64, 128, 256)
    assert info["neck_channels"] == 64
    assert info["params"] == "0.9M"
    assert info["input_size"] == 416


def test_yolox_x_model_info():
    """Test specific info for YOLOX-X."""
    info = get_yolox_model_info("yolox-x")

    assert info["depth"] == 1.33
    assert info["width"] == 1.25
    assert info["backbone"] == "csp_darknet_x"
    assert info["neck"] == "yolox_pafpn_x"
    assert info["backbone_channels"] == (320, 640, 1280)
    assert info["neck_channels"] == 320
    assert info["params"] == "99.1M"
    assert info["mAP"] == 51.5
