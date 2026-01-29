"""Tests for the ImageClassifier LightningModule."""

import pytest
import torch

from autotimm.metrics import MetricConfig
from autotimm.tasks.classification import ImageClassifier


# Helper fixture for common metric configs
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
