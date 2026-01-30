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
        ['adadelta', 'adagrad', 'adam', 'adamax', 'adamw', 'asgd', ...]
        >>> print(optimizers.get("timm", []))
        ['adabelief', 'adafactor', 'adahessian', 'adamp', ...]
    """
    import inspect
    import torch.optim as torch_optim

    # Dynamically discover PyTorch optimizers
    torch_optimizers = []
    for name, obj in inspect.getmembers(torch_optim):
        if (
            inspect.isclass(obj)
            and issubclass(obj, torch_optim.Optimizer)
            and obj is not torch_optim.Optimizer
        ):
            torch_optimizers.append(name.lower())

    torch_optimizers.sort()
    result = {"torch": torch_optimizers}

    if include_timm:
        try:
            import timm.optim as timm_optim

            # Dynamically discover timm optimizers
            timm_optimizers = []
            for name, obj in inspect.getmembers(timm_optim):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, torch_optim.Optimizer)
                    and obj.__module__.startswith("timm.optim")
                ):
                    # Use lowercase name without 'Optimizer' suffix
                    clean_name = name.replace("Optimizer", "").lower()
                    if clean_name and clean_name not in timm_optimizers:
                        timm_optimizers.append(clean_name)

            timm_optimizers.sort()
            result["timm"] = timm_optimizers
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
        ['chainedscheduler', 'constantlr', 'cosineannealinglr', ...]
        >>> print(schedulers.get("timm", []))
        ['cosinelrscheduler', 'multisteplrscheduler', ...]
    """
    import inspect
    import torch.optim.lr_scheduler as torch_scheduler

    # Classes to exclude (not actual schedulers)
    exclude_classes = {
        "LRScheduler",
        "_LRScheduler",
        "Optimizer",
        "Counter",
        "Tensor",
        "Any",
        "SupportsFloat",
        "partial",
        "ref",
    }

    # Dynamically discover PyTorch schedulers
    torch_schedulers = []
    for name, obj in inspect.getmembers(torch_scheduler):
        if (
            inspect.isclass(obj)
            and not name.startswith("_")
            and name not in exclude_classes
            and obj.__module__ == "torch.optim.lr_scheduler"
            and (
                # Match common scheduler patterns
                "LR" in name
                or "Scheduler" in name
                or "Cyclic" in name
                or "Annealing" in name
                or "Warm" in name
            )
        ):
            # Convert to lowercase for consistency
            torch_schedulers.append(name.lower())

    torch_schedulers.sort()
    result = {"torch": torch_schedulers}

    if include_timm:
        try:
            import timm.scheduler as timm_scheduler

            # Dynamically discover timm schedulers
            timm_schedulers = []
            for name, obj in inspect.getmembers(timm_scheduler):
                if (
                    inspect.isclass(obj)
                    and hasattr(obj, "step")
                    and obj.__module__.startswith("timm.scheduler")
                    and not name.startswith("_")
                ):
                    # Use lowercase name
                    clean_name = name.lower()
                    if clean_name and clean_name not in timm_schedulers:
                        timm_schedulers.append(clean_name)

            timm_schedulers.sort()
            result["timm"] = timm_schedulers
        except ImportError:
            result["timm"] = []

    return result
