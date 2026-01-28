"""Tests for the ImageClassifier LightningModule."""

import torch

from autotimm.tasks.classification import ImageClassifier


def test_forward_shape():
    model = ImageClassifier(backbone="resnet18", num_classes=5)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)


def test_training_step():
    model = ImageClassifier(backbone="resnet18", num_classes=5)
    model.train()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 5, (4,))
    loss = model.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0  # scalar
    assert loss.requires_grad


def test_freeze_backbone():
    model = ImageClassifier(backbone="resnet18", num_classes=5, freeze_backbone=True)
    for param in model.backbone.parameters():
        assert not param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad


def test_predict_step():
    model = ImageClassifier(backbone="resnet18", num_classes=5)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        probs = model.predict_step((x,), batch_idx=0)
    assert probs.shape == (2, 5)
    # softmax probabilities should sum to ~1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
