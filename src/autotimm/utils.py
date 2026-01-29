"""Utility helpers for autotimm."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of parameters in a model.

    Parameters:
        model: A ``torch.nn.Module``.
        trainable_only: If ``True``, count only parameters with
            ``requires_grad=True``.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def list_optimizers(include_timm: bool = True) -> dict[str, list[str]]:
    """List available optimizers from torch and optionally timm.

    Parameters:
        include_timm: If ``True``, include timm optimizers (requires timm).

    Returns:
        Dictionary with keys ``"torch"`` and optionally ``"timm"``, each containing
        a list of optimizer names.

    Example:
        >>> optimizers = list_optimizers()
        >>> print(optimizers["torch"])
        ['adamw', 'adam', 'sgd', 'rmsprop', 'adagrad']
        >>> print(optimizers.get("timm", []))
        ['adamp', 'sgdp', 'adabelief', 'radam', ...]
    """
    result = {
        "torch": [
            "adamw",
            "adam",
            "sgd",
            "rmsprop",
            "adagrad",
        ]
    }

    if include_timm:
        try:
            import timm.optim  # noqa: F401

            result["timm"] = [
                "adamp",
                "sgdp",
                "adabelief",
                "radam",
                "adahessian",
                "lamb",
                "lars",
                "madgrad",
                "novograd",
            ]
        except ImportError:
            result["timm"] = []

    return result


def list_schedulers(include_timm: bool = True) -> dict[str, list[str]]:
    """List available learning rate schedulers from torch and optionally timm.

    Parameters:
        include_timm: If ``True``, include timm schedulers (requires timm).

    Returns:
        Dictionary with keys ``"torch"`` and optionally ``"timm"``, each containing
        a list of scheduler names.

    Example:
        >>> schedulers = list_schedulers()
        >>> print(schedulers["torch"])
        ['cosine', 'step', 'multistep', 'exponential', 'onecycle', 'plateau']
        >>> print(schedulers.get("timm", []))
        ['cosine_with_restarts']
    """
    result = {
        "torch": [
            "cosine",
            "step",
            "multistep",
            "exponential",
            "onecycle",
            "plateau",
        ]
    }

    if include_timm:
        try:
            import timm.scheduler  # noqa: F401

            result["timm"] = [
                "cosine_with_restarts",
            ]
        except ImportError:
            result["timm"] = []

    return result
