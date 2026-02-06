"""Tests for YOLOX learning rate scheduler."""

import torch
from torch import nn

from autotimm.models.yolox_scheduler import YOLOXLRScheduler, YOLOXWarmupLR


def test_yolox_scheduler_warmup():
    """Test YOLOX scheduler warmup phase."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
        scheduler_type="cosine",
    )

    # Initial LR should be 0 (warmup start)
    assert optimizer.param_groups[0]["lr"] == 0.0

    # After 1 epoch, LR should be 1/5 of base_lr
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.01 / 5) < 1e-6

    # After 5 epochs (end of warmup), LR should be base_lr
    for _ in range(4):
        scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-6


def test_yolox_scheduler_cosine_decay():
    """Test YOLOX scheduler cosine decay phase."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
        scheduler_type="cosine",
    )

    # Skip warmup
    for _ in range(5):
        scheduler.step()

    # During cosine decay, LR should decrease smoothly
    lr_after_warmup = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr_after_first_decay = optimizer.param_groups[0]["lr"]

    assert lr_after_first_decay < lr_after_warmup
    assert lr_after_first_decay > 0.01 * 0.05  # Above minimum


def test_yolox_scheduler_no_aug_phase():
    """Test YOLOX scheduler no augmentation phase."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
        scheduler_type="cosine",
    )

    # Go to no-aug phase (last 15 epochs)
    for _ in range(300 - 15):
        scheduler.step()

    # LR should be at minimum during no-aug phase
    assert abs(optimizer.param_groups[0]["lr"] - 0.01 * 0.05) < 1e-6

    # Should stay at minimum for remaining epochs
    for _ in range(14):
        scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - 0.01 * 0.05) < 1e-6


def test_yolox_scheduler_linear_decay():
    """Test YOLOX scheduler with linear decay."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
        scheduler_type="linear",
    )

    # Skip warmup
    for _ in range(5):
        scheduler.step()

    # During linear decay, LR should decrease linearly
    lr_values = []
    for _ in range(10):
        scheduler.step()
        lr_values.append(optimizer.param_groups[0]["lr"])

    # Check that LR is decreasing
    for i in range(len(lr_values) - 1):
        assert lr_values[i] > lr_values[i + 1]


def test_yolox_warmup_lr():
    """Test simple YOLOX warmup scheduler."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = YOLOXWarmupLR(
        optimizer,
        warmup_epochs=5,
        warmup_lr_start=0.0,
    )

    # Initial LR should be 0
    assert optimizer.param_groups[0]["lr"] == 0.0

    # After warmup, should be at base LR
    for _ in range(5):
        scheduler.step()

    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-6


def test_yolox_scheduler_multiple_param_groups():
    """Test YOLOX scheduler with multiple parameter groups."""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(
        [
            {"params": model.weight, "lr": 0.01},
            {"params": model.bias, "lr": 0.001},
        ]
    )
    scheduler = YOLOXLRScheduler(
        optimizer,
        total_epochs=300,
        warmup_epochs=5,
    )

    # Both groups should start at warmup_lr_start
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 0.0

    # After warmup, should be at respective base LRs
    for _ in range(5):
        scheduler.step()

    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-6
    assert abs(optimizer.param_groups[1]["lr"] - 0.001) < 1e-6
