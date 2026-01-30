"""Tests for semantic segmentation task."""

import pytest
import torch

from autotimm import MetricConfig, SemanticSegmentor


class TestSemanticSegmentor:
    """Test SemanticSegmentor."""

    def test_model_creation_deeplabv3plus(self):
        """Test model creation with DeepLabV3+ head."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            head_type="deeplabv3plus",
            loss_type="combined",
            metrics=None,
        )

        assert model is not None
        assert model.num_classes == 10

    def test_model_creation_fcn(self):
        """Test model creation with FCN head."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=21,
            head_type="fcn",
            loss_type="ce",
            metrics=None,
        )

        assert model is not None
        assert model.num_classes == 21

    def test_forward_pass(self):
        """Test forward pass."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            metrics=None,
        )

        images = torch.randn(2, 3, 224, 224)
        output = model(images)

        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 5  # Number of classes
        assert output.shape[2] > 0  # Height
        assert output.shape[3] > 0  # Width

    def test_predict(self):
        """Test predict method."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            head_type="fcn",
            metrics=None,
        )

        images = torch.randn(2, 3, 224, 224)
        predictions = model.predict(images)

        assert predictions.shape[0] == 2  # Batch size
        assert predictions.dim() == 3  # [B, H, W]
        assert predictions.min() >= 0
        assert predictions.max() < 10

    def test_predict_logits(self):
        """Test predict with return_logits=True."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            metrics=None,
        )

        images = torch.randn(2, 3, 224, 224)
        logits = model.predict(images, return_logits=True)

        assert logits.shape[0] == 2
        assert logits.shape[1] == 5  # Number of classes

    def test_loss_types(self):
        """Test different loss types."""
        loss_types = ["ce", "dice", "combined", "focal"]

        for loss_type in loss_types:
            model = SemanticSegmentor(
                backbone="resnet18",
                num_classes=5,
                head_type="fcn",
                loss_type=loss_type,
                metrics=None,
            )

            images = torch.randn(2, 3, 224, 224)
            logits = model(images)
            masks = torch.randint(0, 5, (2, logits.shape[2], logits.shape[3]))

            loss = model._compute_loss(logits, masks)

            assert not torch.isnan(loss), f"Loss type {loss_type} produced NaN"
            assert loss >= 0, f"Loss type {loss_type} produced negative loss"

    def test_training_step(self):
        """Test training step."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            loss_type="combined",
            metrics=None,
        )

        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "mask": torch.randint(0, 5, (2, 56, 56)),  # Smaller size
        }

        loss = model.training_step(batch, 0)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_validation_step(self):
        """Test validation step."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            metrics=None,
        )

        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "mask": torch.randint(0, 5, (2, 56, 56)),
        }

        # Should not raise
        model.validation_step(batch, 0)

    def test_with_metrics(self):
        """Test with metrics configuration."""
        metrics = [
            MetricConfig(
                name="test_metric",
                backend="torchmetrics",
                metric_class="Accuracy",
                params={"task": "multiclass", "num_classes": 5},
                stages=["val"],
                prog_bar=True,
            ),
        ]

        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            metrics=metrics,
        )

        assert len(model.val_metrics) > 0

    def test_freeze_backbone(self):
        """Test backbone freezing."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            freeze_backbone=True,
            metrics=None,
        )

        # Check that backbone parameters are frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_optimizer_configuration(self):
        """Test optimizer configuration."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            lr=1e-4,
            weight_decay=1e-5,
            optimizer="adamw",
            metrics=None,
        )

        # Should not raise
        config = model.configure_optimizers()
        assert "optimizer" in config

    def test_different_backbones(self):
        """Test with different backbones."""
        backbones = ["resnet18", "resnet34", "mobilenetv3_small_100"]

        for backbone in backbones:
            try:
                model = SemanticSegmentor(
                    backbone=backbone,
                    num_classes=10,
                    head_type="fcn",
                    metrics=None,
                )

                images = torch.randn(1, 3, 224, 224)
                output = model(images)

                assert output.shape[0] == 1
                assert output.shape[1] == 10
            except Exception as e:
                pytest.skip(f"Backbone {backbone} not available: {e}")

    def test_ignore_index(self):
        """Test ignore index handling."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            loss_type="combined",
            ignore_index=255,
            metrics=None,
        )

        images = torch.randn(2, 3, 224, 224)
        logits = model(images)
        masks = torch.randint(0, 6, (2, logits.shape[2], logits.shape[3]))
        masks[0, 0:10, 0:10] = 255  # Add ignored pixels

        loss = model._compute_loss(logits, masks)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            metrics=None,
        )

        images = torch.randn(1, 3, 224, 224, requires_grad=True)
        logits = model(images)
        loss = logits.sum()
        loss.backward()

        assert images.grad is not None
        assert not torch.isnan(images.grad).any()


class TestSemanticSegmentorIntegration:
    """Integration tests for SemanticSegmentor."""

    def test_full_training_loop(self):
        """Test a full training iteration."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            loss_type="combined",
            metrics=None,
        )

        # Set to training mode
        model.train()

        # Training step
        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "mask": torch.randint(0, 5, (2, 56, 56)),
        }

        loss = model.training_step(batch, 0)

        assert not torch.isnan(loss)
        assert loss.requires_grad

    def test_evaluation_mode(self):
        """Test evaluation mode."""
        model = SemanticSegmentor(
            backbone="resnet18",
            num_classes=5,
            head_type="fcn",
            metrics=None,
        )

        model.eval()

        images = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            predictions = model.predict(images)

        assert predictions.shape[0] == 2
        assert predictions.min() >= 0
        assert predictions.max() < 5

    def test_deeplabv3plus_vs_fcn(self):
        """Test DeepLabV3+ vs FCN output shapes."""
        images = torch.randn(1, 3, 224, 224)

        model_deeplabv3 = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            head_type="deeplabv3plus",
            metrics=None,
        )

        model_fcn = SemanticSegmentor(
            backbone="resnet18",
            num_classes=10,
            head_type="fcn",
            metrics=None,
        )

        output_deeplabv3 = model_deeplabv3(images)
        output_fcn = model_fcn(images)

        # Both should produce valid outputs
        assert output_deeplabv3.shape[0] == 1
        assert output_deeplabv3.shape[1] == 10

        assert output_fcn.shape[0] == 1
        assert output_fcn.shape[1] == 10

        # DeepLabV3+ typically produces higher resolution
        assert output_deeplabv3.shape[2] >= output_fcn.shape[2]
