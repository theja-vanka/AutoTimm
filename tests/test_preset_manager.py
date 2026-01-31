"""Tests for preset manager functionality."""

import pytest

from autotimm.data.preset_manager import (
    BackendRecommendation,
    compare_backends,
    recommend_backend,
)
from autotimm.data.transform_config import TransformConfig


class TestBackendRecommendation:
    """Tests for BackendRecommendation dataclass."""

    def test_to_config(self):
        """Test converting recommendation to TransformConfig."""
        rec = BackendRecommendation(
            backend="torchvision",
            preset="randaugment",
            reasoning="Test",
            available_presets=["default", "randaugment"],
        )
        config = rec.to_config(image_size=384)
        assert isinstance(config, TransformConfig)
        assert config.backend == "torchvision"
        assert config.preset == "randaugment"
        assert config.image_size == 384

    def test_str_representation(self):
        """Test string representation."""
        rec = BackendRecommendation(
            backend="albumentations",
            preset="strong",
            reasoning="Advanced augmentation needed",
            available_presets=["default", "strong"],
            alternative="Torchvision is simpler",
        )
        str_repr = str(rec)
        assert "albumentations" in str_repr
        assert "strong" in str_repr
        assert "Advanced augmentation needed" in str_repr
        assert "Alternative" in str_repr


class TestRecommendBackend:
    """Tests for recommend_backend function."""

    def test_classification_default(self):
        """Test default recommendation for classification."""
        rec = recommend_backend(task="classification")
        assert rec.backend in ("torchvision", "albumentations")
        assert rec.preset in rec.available_presets
        assert len(rec.reasoning) > 0

    def test_classification_advanced(self):
        """Test classification with advanced augmentation."""
        rec = recommend_backend(task="classification", needs_advanced_augmentation=True)
        assert rec.backend == "albumentations"
        assert rec.preset == "strong"

    def test_detection(self):
        """Test recommendation for object detection."""
        rec = recommend_backend(task="detection")
        assert rec.backend == "albumentations"
        assert "bbox" in rec.reasoning.lower() or "detection" in rec.reasoning.lower()

    def test_segmentation(self):
        """Test recommendation for semantic segmentation."""
        rec = recommend_backend(task="segmentation")
        assert rec.backend == "albumentations"
        assert "mask" in rec.reasoning.lower() or "segmentation" in rec.reasoning.lower()

    def test_instance_segmentation(self):
        """Test recommendation for instance segmentation."""
        rec = recommend_backend(task="instance_segmentation")
        assert rec.backend == "albumentations"

    def test_spatial_transforms(self):
        """Test recommendation with spatial transform requirement."""
        rec = recommend_backend(needs_spatial_transforms=True)
        assert rec.backend == "albumentations"
        assert "spatial" in rec.reasoning.lower() or "rotation" in rec.reasoning.lower()

    def test_bbox_or_masks(self):
        """Test recommendation with bbox/mask requirement."""
        rec = recommend_backend(has_bbox_or_masks=True)
        assert rec.backend == "albumentations"

    def test_prioritize_speed(self):
        """Test recommendation when prioritizing speed."""
        rec = recommend_backend(prioritize_speed=True)
        # Should recommend torchvision for speed when no bbox/masks
        assert rec.backend == "torchvision"
        assert rec.preset == "light"

    def test_combined_requirements(self):
        """Test recommendation with multiple requirements."""
        rec = recommend_backend(
            task="detection",
            needs_advanced_augmentation=True,
            needs_spatial_transforms=True,
        )
        assert rec.backend == "albumentations"
        assert rec.preset in rec.available_presets

    def test_available_presets_populated(self):
        """Test that available presets are populated."""
        rec = recommend_backend(task="classification")
        assert len(rec.available_presets) > 0
        assert all(isinstance(p, str) for p in rec.available_presets)

    def test_config_creation(self):
        """Test creating config from recommendation."""
        rec = recommend_backend(task="classification")
        config = rec.to_config(image_size=512)
        assert config.backend == rec.backend
        assert config.preset == rec.preset
        assert config.image_size == 512

    def test_no_task_provided(self):
        """Test recommendation without specifying task."""
        rec = recommend_backend()
        assert rec.backend in ("torchvision", "albumentations")
        assert rec.preset in rec.available_presets


class TestCompareBackends:
    """Tests for compare_backends function."""

    def test_returns_comparison_dict(self):
        """Test that comparison returns dictionary."""
        comparison = compare_backends(verbose=False)
        assert isinstance(comparison, dict)
        assert "torchvision" in comparison
        assert "albumentations" in comparison

    def test_comparison_structure(self):
        """Test structure of comparison dictionary."""
        comparison = compare_backends(verbose=False)

        for backend in ["torchvision", "albumentations"]:
            assert "backend" in comparison[backend]
            assert "speed" in comparison[backend]
            assert "augmentations" in comparison[backend]
            assert "bbox_mask_support" in comparison[backend]
            assert "best_for" in comparison[backend]
            assert "presets" in comparison[backend]
            assert "pros" in comparison[backend]
            assert "cons" in comparison[backend]

    def test_presets_are_lists(self):
        """Test that presets are returned as lists."""
        comparison = compare_backends(verbose=False)
        assert isinstance(comparison["torchvision"]["presets"], list)
        assert isinstance(comparison["albumentations"]["presets"], list)
        assert len(comparison["torchvision"]["presets"]) > 0
        assert len(comparison["albumentations"]["presets"]) > 0

    def test_verbose_mode_no_error(self):
        """Test that verbose mode doesn't raise errors."""
        # Should not raise even if rich is not available
        try:
            comparison = compare_backends(verbose=True)
            assert isinstance(comparison, dict)
        except Exception as e:
            pytest.fail(f"compare_backends(verbose=True) raised {e}")


class TestIntegration:
    """Integration tests for preset manager with other components."""

    def test_recommendation_with_image_classifier(self):
        """Test using recommendation with ImageClassifier."""
        from autotimm import ImageClassifier, MetricConfig

        rec = recommend_backend(task="classification")
        config = rec.to_config(image_size=224)

        metrics = [
            MetricConfig(
                name="accuracy",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 10},
                stages=["val"],
            )
        ]

        model = ImageClassifier(
            backbone="resnet18",
            num_classes=10,
            transform_config=config,
            metrics=metrics,
        )

        assert model is not None

    def test_recommendation_with_semantic_segmentor(self):
        """Test using recommendation with SemanticSegmentor."""
        from autotimm import MetricConfig, SemanticSegmentor

        rec = recommend_backend(task="segmentation")
        config = rec.to_config(image_size=512)

        metrics = [
            MetricConfig(
                name="iou",
                backend="torchmetrics",
                metric_class="JaccardIndex",
                params={"task": "multiclass", "num_classes": 19},
                stages=["val"],
            )
        ]

        model = SemanticSegmentor(
            backbone="resnet50",
            num_classes=19,
            metrics=metrics,
        )

        assert model is not None

    def test_all_task_types(self):
        """Test recommendation for all task types."""
        tasks = ["classification", "detection", "segmentation", "instance_segmentation"]

        for task in tasks:
            rec = recommend_backend(task=task)
            assert rec.backend in ("torchvision", "albumentations")
            assert rec.preset in rec.available_presets
            config = rec.to_config()
            assert isinstance(config, TransformConfig)
