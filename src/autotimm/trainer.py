"""Convenience wrapper around pl.Trainer for autotimm tasks."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from autotimm.loggers import create_logger


def create_trainer(
    max_epochs: int = 10,
    accelerator: str = "auto",
    devices: int | str = "auto",
    precision: str | int = 32,
    logger: str | pl.loggers.Logger = "tensorboard",
    logger_kwargs: dict[str, Any] | None = None,
    callbacks: list[pl.Callback] | None = None,
    default_root_dir: str = ".",
    gradient_clip_val: float | None = None,
    accumulate_grad_batches: int = 1,
    val_check_interval: float | int = 1.0,
    enable_checkpointing: bool = True,
    **trainer_kwargs: Any,
) -> pl.Trainer:
    """Create a configured ``pl.Trainer``.

    This is a convenience function that wires up the logger and
    sensible defaults.  All ``**trainer_kwargs`` are forwarded to
    ``pl.Trainer``, so any Lightning Trainer argument works.

    Parameters:
        max_epochs: Number of training epochs.
        accelerator: ``"auto"``, ``"gpu"``, ``"cpu"``, ``"tpu"``, etc.
        devices: Number of devices or ``"auto"``.
        precision: Training precision (``32``, ``16``, ``"bf16-mixed"``, etc.).
        logger: Logger backend name (str) or a pre-built Logger instance.
        logger_kwargs: Extra kwargs for the logger factory.
        callbacks: List of Lightning callbacks.
        default_root_dir: Root directory for logs and checkpoints.
        gradient_clip_val: Gradient clipping value.
        accumulate_grad_batches: Gradient accumulation steps.
        val_check_interval: How often to run validation (1.0 = every epoch).
        enable_checkpointing: Whether to save model checkpoints.
        **trainer_kwargs: Any other ``pl.Trainer`` argument.

    Returns:
        A configured ``pl.Trainer`` instance.
    """
    if isinstance(logger, str):
        resolved_logger = create_logger(logger, **(logger_kwargs or {}))
    else:
        resolved_logger = logger

    if callbacks is None:
        callbacks = []

    if enable_checkpointing:
        has_checkpoint_cb = any(
            isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in callbacks
        )
        if not has_checkpoint_cb:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    monitor="val/acc",
                    mode="max",
                    save_top_k=1,
                    filename="best-{epoch}-{val/acc:.4f}",
                )
            )

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=resolved_logger,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        enable_checkpointing=enable_checkpointing,
        **trainer_kwargs,
    )
