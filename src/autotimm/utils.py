"""Utility helpers for autotimm."""

from __future__ import annotations


def count_parameters(model, trainable_only: bool = True) -> int:
    """Return the number of parameters in a model.

    Parameters:
        model: A ``torch.nn.Module``.
        trainable_only: If ``True``, count only parameters with
            ``requires_grad=True``.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
