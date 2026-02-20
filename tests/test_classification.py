"""Tests for the ImageClassifier LightningModule."""

import pytest
import torch

from autotimm.metrics import MetricConfig
from autotimm.tasks.classification import ImageClassifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_metrics():
    return [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]


NUM_LABELS = 4


@pytest.fixture
def multilabel_metrics():
    return [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="MultilabelAccuracy",
            params={"num_labels": NUM_LABELS},
            stages=["train", "val", "test"],
            prog_bar=True,
        ),
    ]


def test_forward_shape(basic_metrics):
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=basic_metrics,
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_training_step(basic_metrics):
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=basic_metrics,
    )
    model.train()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 5, (4,))
    loss = model.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0  # scalar
    assert loss.requires_grad


def test_freeze_backbone(basic_metrics):
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=basic_metrics,
        freeze_backbone=True,
    )
    for param in model.backbone.parameters():
        assert not param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad


def test_predict_step(basic_metrics):
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=basic_metrics,
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        probs = model.predict_step((x,), batch_idx=0)
    assert probs.shape == (2, 5)
    # softmax probabilities should sum to ~1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_multiple_metrics():
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["train", "val", "test"],
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="F1Score",
            params={"task": "multiclass", "average": "macro"},
            stages=["val", "test"],
        ),
    ]
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=metrics,
    )

    assert "accuracy" in model.train_metrics
    assert "accuracy" in model.val_metrics
    assert "f1" in model.val_metrics
    assert "f1" not in model.train_metrics


def test_validation_step():
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["val"],
        ),
    ]
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=metrics,
    )
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 5, (4,))

    # Should not raise
    model.validation_step((x, y), batch_idx=0)


def test_test_step():
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="Accuracy",
            params={"task": "multiclass"},
            stages=["test"],
        ),
    ]
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=metrics,
    )
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 5, (4,))

    # Should not raise
    model.test_step((x, y), batch_idx=0)


def test_inference_without_metrics():
    """Test that ImageClassifier works for inference without metrics."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=None,  # No metrics for inference
    )
    model.eval()

    # Forward pass should work
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)

    # Predict step should work
    with torch.no_grad():
        probs = model.predict_step((x,), batch_idx=0)
    assert probs.shape == (2, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_training_without_metrics():
    """Test that training steps work without metrics (only loss is logged)."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=5,
        metrics=None,  # No metrics
    )
    model.train()

    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 5, (4,))

    # Training step should work (logs loss only)
    loss = model.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0
    assert loss.requires_grad

    # Validation step should work
    model.eval()
    model.validation_step((x, y), batch_idx=0)

    # Test step should work
    model.test_step((x, y), batch_idx=0)


# ---------------------------------------------------------------------------
# Multi-label tests
# ---------------------------------------------------------------------------


def test_multi_label_forward_shape():
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=NUM_LABELS,
        multi_label=True,
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_LABELS)


def test_multi_label_training_step(multilabel_metrics):
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=NUM_LABELS,
        multi_label=True,
        metrics=multilabel_metrics,
    )
    model.train()
    x = torch.randn(4, 3, 224, 224)
    # Multi-hot targets
    y = torch.zeros(4, NUM_LABELS)
    y[0, [0, 2]] = 1
    y[1, [1]] = 1
    y[2, [0, 1, 3]] = 1
    y[3, [3]] = 1

    loss = model.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0  # scalar
    assert loss.requires_grad


def test_multi_label_predict_step():
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=NUM_LABELS,
        multi_label=True,
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        probs = model.predict_step((x,), batch_idx=0)
    assert probs.shape == (2, NUM_LABELS)
    # Sigmoid outputs are each in [0, 1] but do NOT sum to 1
    assert (probs >= 0).all() and (probs <= 1).all()
    # Very unlikely that sigmoid outputs sum exactly to 1 for all rows
    # (unless num_labels == 1), so just check they are valid probabilities


def test_multi_label_validation_step(multilabel_metrics):
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=NUM_LABELS,
        multi_label=True,
        metrics=multilabel_metrics,
    )
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    y = torch.zeros(4, NUM_LABELS)
    y[0, [0, 2]] = 1
    y[1, [1]] = 1

    # Should not raise
    model.validation_step((x, y), batch_idx=0)


def test_multi_label_with_metrics():
    metrics = [
        MetricConfig(
            name="accuracy",
            backend="torchmetrics",
            metric_class="MultilabelAccuracy",
            params={"num_labels": NUM_LABELS},
            stages=["train", "val"],
            prog_bar=True,
        ),
        MetricConfig(
            name="f1",
            backend="torchmetrics",
            metric_class="MultilabelF1Score",
            params={"num_labels": NUM_LABELS, "average": "macro"},
            stages=["val"],
        ),
    ]
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=NUM_LABELS,
        multi_label=True,
        metrics=metrics,
    )

    assert "accuracy" in model.train_metrics
    assert "accuracy" in model.val_metrics
    assert "f1" in model.val_metrics
    assert "f1" not in model.train_metrics


def test_multi_label_label_smoothing_error():
    with pytest.raises(ValueError, match="label_smoothing.*not supported.*multi_label"):
        ImageClassifier(
            backbone="resnet18",
            num_classes=NUM_LABELS,
            multi_label=True,
            label_smoothing=0.1,
        )


def test_multi_label_inference_without_metrics():
    """Test multi-label works for inference without metrics."""
    model = ImageClassifier(
        backbone="resnet18",
        num_classes=NUM_LABELS,
        multi_label=True,
        metrics=None,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_LABELS)

    with torch.no_grad():
        probs = model.predict_step((x,), batch_idx=0)
    assert probs.shape == (2, NUM_LABELS)
    assert (probs >= 0).all() and (probs <= 1).all()
