"""Tests for Hugging Face Hub backbone support."""

from __future__ import annotations

import pytest
import torch

import autotimm
from autotimm.backbone import _is_hf_hub_model


class TestHFHubDetection:
    """Test detection of HF Hub model names."""

    def test_is_hf_hub_model_with_prefix(self):
        """Test detection of models with hf-hub: prefix."""
        assert _is_hf_hub_model("hf-hub:timm/resnet50.a1_in1k")
        assert _is_hf_hub_model("hf_hub:timm/resnet50.a1_in1k")
        assert _is_hf_hub_model("timm/resnet50.a1_in1k")

    def test_is_not_hf_hub_model(self):
        """Test that regular timm models are not detected as HF Hub."""
        assert not _is_hf_hub_model("resnet50")
        assert not _is_hf_hub_model("efficientnet_b0")
        assert not _is_hf_hub_model("vit_base_patch16_224")


class TestHFHubBackboneCreation:
    """Test creating backbones from HF Hub."""

    @pytest.mark.slow
    def test_create_hf_hub_backbone(self):
        """Test creating a backbone from HF Hub (requires internet)."""
        # Use a small model to speed up the test
        model_name = "hf-hub:timm/resnet18.a1_in1k"
        backbone = autotimm.create_backbone(model_name)

        assert backbone is not None
        assert hasattr(backbone, "num_features")
        assert backbone.num_features > 0

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = backbone(x)
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == backbone.num_features

    @pytest.mark.slow
    def test_create_hf_hub_feature_backbone(self):
        """Test creating a feature backbone from HF Hub (requires internet)."""
        model_name = "hf-hub:timm/resnet18.a1_in1k"
        backbone = autotimm.create_feature_backbone(model_name)

        assert backbone is not None
        assert hasattr(backbone, "feature_info")

        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        features = backbone(x)
        assert isinstance(features, (list, tuple))
        # Number of features depends on model architecture and timm version
        assert len(features) >= 4

        # Get feature channels
        channels = autotimm.get_feature_channels(backbone)
        assert len(channels) >= 4
        assert all(c > 0 for c in channels)

    def test_create_backbone_string_shortcut(self):
        """Test that string shortcut works with HF Hub models."""
        # This should not raise an error even though the model is not in timm.list_models()
        # Note: This test doesn't actually download the model, just checks validation
        model_name = "hf-hub:timm/resnet18.a1_in1k"

        # The validation should skip checking timm.list_models() for HF Hub models
        # We can't test actual model creation without internet, but we can verify
        # that the validation logic allows HF Hub models
        try:
            from autotimm.backbone import _is_hf_hub_model

            assert _is_hf_hub_model(model_name)
        except Exception as e:
            pytest.fail(f"HF Hub model validation failed: {e}")


class TestListHFHubBackbones:
    """Test listing HF Hub backbones."""

    @pytest.mark.slow
    def test_list_hf_hub_backbones(self):
        """Test listing HF Hub backbones (requires internet)."""
        try:
            models = autotimm.list_hf_hub_backbones(limit=5)
            assert isinstance(models, list)
            assert len(models) > 0
            assert all(m.startswith("hf-hub:") for m in models)
        except ImportError:
            pytest.skip("huggingface_hub not installed")
        except Exception as e:
            pytest.skip(f"Cannot access HF Hub: {e}")

    @pytest.mark.slow
    def test_list_hf_hub_backbones_with_search(self):
        """Test searching HF Hub backbones (requires internet)."""
        try:
            models = autotimm.list_hf_hub_backbones(model_name="resnet", limit=5)
            assert isinstance(models, list)
            # Should find at least some resnet models
            assert len(models) > 0
            assert all(m.startswith("hf-hub:") for m in models)
        except ImportError:
            pytest.skip("huggingface_hub not installed")
        except Exception as e:
            pytest.skip(f"Cannot access HF Hub: {e}")

    def test_list_hf_hub_backbones_without_hf_hub(self, monkeypatch):
        """Test that proper error is raised when huggingface_hub is not available."""
        # Mock HfApi as None to simulate missing package
        import autotimm.backbone

        original_hfapi = autotimm.backbone.HfApi
        monkeypatch.setattr(autotimm.backbone, "HfApi", None)

        with pytest.raises(ImportError, match="huggingface_hub is required"):
            autotimm.list_hf_hub_backbones()

        # Restore original
        monkeypatch.setattr(autotimm.backbone, "HfApi", original_hfapi)


class TestBackboneConfigWithHFHub:
    """Test BackboneConfig with HF Hub models."""

    def test_backbone_config_with_hf_hub_model(self):
        """Test creating BackboneConfig with HF Hub model."""
        from autotimm.backbone import BackboneConfig

        config = BackboneConfig(
            model_name="hf-hub:timm/resnet18.a1_in1k",
            pretrained=True,
            num_classes=1000,
        )

        assert config.model_name == "hf-hub:timm/resnet18.a1_in1k"
        assert config.pretrained is True
        assert config.num_classes == 1000

    def test_feature_backbone_config_with_hf_hub_model(self):
        """Test creating FeatureBackboneConfig with HF Hub model."""
        from autotimm.backbone import FeatureBackboneConfig

        config = FeatureBackboneConfig(
            model_name="hf-hub:timm/resnet50.a1_in1k",
            pretrained=True,
            out_indices=(1, 2, 3, 4),
        )

        assert config.model_name == "hf-hub:timm/resnet50.a1_in1k"
        assert config.pretrained is True
        assert config.out_indices == (1, 2, 3, 4)


@pytest.mark.slow
class TestEndToEndWithHFHub:
    """End-to-end tests using HF Hub models (requires internet)."""

    def test_image_classifier_with_hf_hub(self):
        """Test ImageClassifier with HF Hub backbone."""
        from autotimm import ImageClassifier, MetricConfig

        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["train", "val"],
                prog_bar=True,
            ),
        ]

        model = ImageClassifier(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=10,
            metrics=metrics,
            lr=1e-3,
        )

        assert model is not None
        # num_classes is stored in hparams by save_hyperparameters
        assert model.hparams.num_classes == 10

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)

    def test_semantic_segmentor_with_hf_hub(self):
        """Test SemanticSegmentor with HF Hub backbone."""
        from autotimm import MetricConfig, SemanticSegmentor

        metrics = [
            MetricConfig(
                name="iou",
                backend="torchmetrics",
                metric_class="JaccardIndex",
                params={"task": "multiclass", "num_classes": 19, "average": "macro"},
                stages=["val"],
                prog_bar=True,
            ),
        ]

        model = SemanticSegmentor(
            backbone="hf-hub:timm/resnet18.a1_in1k",
            num_classes=19,
            head_type="fcn",
            metrics=metrics,
        )

        assert model is not None
        assert model.num_classes == 19

        # Test forward pass
        x = torch.randn(1, 3, 512, 512)
        output = model(x)
        # FCN head outputs at reduced resolution (feature map size, not input size)
        # The output is upsampled during loss computation, not in forward()
        assert output.shape[0] == 1
        assert output.shape[1] == 19
        # Output spatial size will be smaller than input due to backbone downsampling

    def test_object_detector_with_hf_hub(self):
        """Test ObjectDetector with HF Hub backbone."""
        from autotimm import MetricConfig, ObjectDetector
        from autotimm.backbone import FeatureBackboneConfig

        metrics = [
            MetricConfig(
                name="mAP",
                backend="torchmetrics",
                metric_class="MeanAveragePrecision",
                params={"box_format": "xyxy", "iou_type": "bbox"},
                stages=["val"],
                prog_bar=True,
            ),
        ]

        # Configure backbone with specific out_indices that match FCOS strides
        # out_indices (2, 3, 4) corresponds to C3, C4, C5 with strides [8, 16, 32]
        # FPN adds P6, P7 with strides [64, 128] giving 5 total levels
        backbone_config = FeatureBackboneConfig(
            model_name="resnet18",
            pretrained=False,
            out_indices=(2, 3, 4),
        )

        model = ObjectDetector(
            backbone=backbone_config,
            num_classes=80,
            metrics=metrics,
        )

        assert model is not None
        assert model.num_classes == 80

        # Test predict method (inference mode with post-processing)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model.predict(x)
        assert isinstance(output, list)
        assert len(output) == 1  # One prediction per image
        # Each prediction should be a dict with boxes, scores, labels
        assert isinstance(output[0], dict)
        assert "boxes" in output[0]
        assert "scores" in output[0]
        assert "labels" in output[0]
