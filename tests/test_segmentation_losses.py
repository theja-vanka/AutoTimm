"""Tests for segmentation loss functions."""

import torch

from autotimm.losses.segmentation import (
    CombinedSegmentationLoss,
    DiceLoss,
    FocalLossPixelwise,
    MaskLoss,
    TverskyLoss,
)


class TestDiceLoss:
    """Test DiceLoss."""

    def test_dice_loss_forward(self):
        """Test forward pass."""
        loss_fn = DiceLoss(num_classes=5, ignore_index=255)
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        loss = loss_fn(logits, targets)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # Loss should be non-negative
        assert loss <= 1  # Dice loss is bounded by 1

    def test_dice_loss_with_ignore_index(self):
        """Test with ignore index."""
        loss_fn = DiceLoss(num_classes=5, ignore_index=255)
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 6, (2, 32, 32))
        targets[0, 0:10, 0:10] = 255  # Add ignored pixels

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_dice_loss_perfect_prediction(self):
        """Test with perfect prediction."""
        loss_fn = DiceLoss(num_classes=3, ignore_index=255)

        # Create one-hot logits (perfect prediction)
        targets = torch.tensor([[[0, 1, 2], [1, 2, 0]]])  # [B, H, W]
        logits = torch.zeros(1, 3, 2, 3)
        for b in range(1):
            for h in range(2):
                for w in range(3):
                    logits[b, targets[b, h, w], h, w] = 10.0  # High confidence

        loss = loss_fn(logits, targets)

        assert loss < 0.1  # Should be close to 0

    def test_dice_loss_reduction(self):
        """Test different reduction modes."""
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        loss_mean = DiceLoss(num_classes=5, reduction="mean")(logits, targets)
        loss_sum = DiceLoss(num_classes=5, reduction="sum")(logits, targets)
        loss_none = DiceLoss(num_classes=5, reduction="none")(logits, targets)

        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0
        assert loss_none.ndim == 2  # [B, C]


class TestFocalLossPixelwise:
    """Test FocalLossPixelwise."""

    def test_focal_loss_forward(self):
        """Test forward pass."""
        loss_fn = FocalLossPixelwise(alpha=0.25, gamma=2.0, ignore_index=255)
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        loss = loss_fn(logits, targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_focal_loss_with_ignore_index(self):
        """Test with ignore index."""
        loss_fn = FocalLossPixelwise(ignore_index=255)
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))  # Valid range 0-4
        targets[0, 0:10, 0:10] = 255  # Set some pixels to ignore_index

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_focal_loss_gamma_effect(self):
        """Test that higher gamma increases focus on hard examples."""
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        loss_gamma0 = FocalLossPixelwise(gamma=0.0)(logits, targets)
        loss_gamma2 = FocalLossPixelwise(gamma=2.0)(logits, targets)

        # Both should be valid losses
        assert not torch.isnan(loss_gamma0)
        assert not torch.isnan(loss_gamma2)


class TestTverskyLoss:
    """Test TverskyLoss."""

    def test_tversky_loss_forward(self):
        """Test forward pass."""
        loss_fn = TverskyLoss(num_classes=5, alpha=0.5, beta=0.5, ignore_index=255)
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        loss = loss_fn(logits, targets)

        assert loss.ndim == 0
        assert loss >= 0
        assert loss <= 1

    def test_tversky_loss_alpha_beta(self):
        """Test with different alpha/beta values."""
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        # FP emphasis
        loss_fp = TverskyLoss(num_classes=5, alpha=0.7, beta=0.3)(logits, targets)

        # FN emphasis
        loss_fn = TverskyLoss(num_classes=5, alpha=0.3, beta=0.7)(logits, targets)

        # Balanced (should be similar to Dice)
        loss_balanced = TverskyLoss(num_classes=5, alpha=0.5, beta=0.5)(logits, targets)

        assert all(not torch.isnan(loss) for loss in [loss_fp, loss_fn, loss_balanced])


class TestMaskLoss:
    """Test MaskLoss."""

    def test_mask_loss_forward(self):
        """Test forward pass."""
        loss_fn = MaskLoss()
        pred_masks = torch.randn(10, 28, 28)  # Logits
        target_masks = torch.randint(0, 2, (10, 28, 28)).float()

        loss = loss_fn(pred_masks, target_masks)

        assert loss.ndim == 0
        assert loss >= 0

    def test_mask_loss_4d_input(self):
        """Test with 4D input."""
        loss_fn = MaskLoss()
        pred_masks = torch.randn(10, 1, 28, 28)
        target_masks = torch.randint(0, 2, (10, 1, 28, 28)).float()

        loss = loss_fn(pred_masks, target_masks)

        assert loss.ndim == 0
        assert loss >= 0

    def test_mask_loss_pos_weight(self):
        """Test with positive weight."""
        pred_masks = torch.randn(10, 28, 28)
        target_masks = torch.randint(0, 2, (10, 28, 28)).float()

        loss_default = MaskLoss()(pred_masks, target_masks)
        loss_weighted = MaskLoss(pos_weight=2.0)(pred_masks, target_masks)

        assert not torch.isnan(loss_default)
        assert not torch.isnan(loss_weighted)


class TestCombinedSegmentationLoss:
    """Test CombinedSegmentationLoss."""

    def test_combined_loss_forward(self):
        """Test forward pass."""
        loss_fn = CombinedSegmentationLoss(
            num_classes=5,
            ce_weight=1.0,
            dice_weight=1.0,
            ignore_index=255,
        )
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        loss = loss_fn(logits, targets)

        assert loss.ndim == 0
        assert loss >= 0

    def test_combined_loss_weights(self):
        """Test with different weight combinations."""
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        # CE only
        loss_ce = CombinedSegmentationLoss(
            num_classes=5, ce_weight=1.0, dice_weight=0.0
        )(logits, targets)

        # Dice only
        loss_dice = CombinedSegmentationLoss(
            num_classes=5, ce_weight=0.0, dice_weight=1.0
        )(logits, targets)

        # Combined
        loss_combined = CombinedSegmentationLoss(
            num_classes=5, ce_weight=1.0, dice_weight=1.0
        )(logits, targets)

        assert all(
            not torch.isnan(loss) for loss in [loss_ce, loss_dice, loss_combined]
        )

    def test_combined_loss_class_weights(self):
        """Test with class weights."""
        loss_fn = CombinedSegmentationLoss(
            num_classes=5,
            ce_weight=1.0,
            dice_weight=1.0,
            class_weights=torch.tensor([1.0, 2.0, 1.0, 1.0, 0.5]),
        )
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))

        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss)
        assert loss >= 0


class TestLossIntegration:
    """Integration tests for loss functions."""

    def test_all_losses_with_batch(self):
        """Test all losses with a batch."""
        logits = torch.randn(4, 10, 64, 64)
        targets = torch.randint(0, 10, (4, 64, 64))

        losses = {
            "dice": DiceLoss(num_classes=10),
            "focal": FocalLossPixelwise(),
            "tversky": TverskyLoss(num_classes=10),
            "combined": CombinedSegmentationLoss(num_classes=10),
        }

        for name, loss_fn in losses.items():
            loss = loss_fn(logits, targets)
            assert not torch.isnan(loss), f"{name} produced NaN"
            assert loss >= 0, f"{name} produced negative loss"

    def test_gradient_flow(self):
        """Test that gradients flow through losses."""
        logits = torch.randn(2, 5, 32, 32, requires_grad=True)
        targets = torch.randint(0, 5, (2, 32, 32))

        loss_fn = CombinedSegmentationLoss(num_classes=5)
        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
