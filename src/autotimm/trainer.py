"""Convenience wrapper around pl.Trainer for autotimm tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner

from autotimm.loggers import LoggerConfig, LoggerManager


@dataclass
class TunerConfig:
    """Configuration for automatic hyperparameter tuning.

    Parameters:
        auto_lr: Whether to use the learning rate finder before training.
            If ``True``, the optimal learning rate will be found and applied.
            If ``False``, the user-specified learning rate is used.
        auto_batch_size: Whether to use the batch size finder before training.
            If ``True``, the optimal batch size will be found and applied.
            If ``False``, the user-specified batch size is used.
        lr_find_kwargs: Additional kwargs passed to ``Tuner.lr_find()``.
            Common options: ``min_lr``, ``max_lr``, ``num_training``,
            ``mode`` ("exponential" or "linear"), ``early_stop_threshold``.
        batch_size_kwargs: Additional kwargs passed to ``Tuner.scale_batch_size()``.
            Common options: ``mode`` ("power" or "binsearch"), ``steps_per_trial``,
            ``init_val``, ``max_trials``.

    Example:
        >>> config = TunerConfig(
        ...     auto_lr=True,
        ...     auto_batch_size=True,
        ...     lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1e-1, "num_training": 100},
        ...     batch_size_kwargs={"mode": "power", "init_val": 16},
        ... )
    """

    auto_lr: bool
    auto_batch_size: bool
    lr_find_kwargs: dict[str, Any] | None = None
    batch_size_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.lr_find_kwargs is None:
            self.lr_find_kwargs = {}
        if self.batch_size_kwargs is None:
            self.batch_size_kwargs = {}


class AutoTrainer(pl.Trainer):
    """A configured ``pl.Trainer`` with sensible defaults for autotimm.

    This is a convenience class that wires up the logger, checkpointing,
    and optional hyperparameter tuning. All ``**trainer_kwargs`` are
    forwarded to ``pl.Trainer``, so any Lightning Trainer argument works.

    Parameters:
        max_epochs: Number of training epochs.
        accelerator: ``"auto"``, ``"gpu"``, ``"cpu"``, ``"tpu"``, etc.
        devices: Number of devices or ``"auto"``.
        precision: Training precision (``32``, ``16``, ``"bf16-mixed"``, etc.).
        logger: A ``LoggerManager`` instance, a list of ``LoggerConfig``
            objects, a pre-built Logger instance/list, or ``False`` to
            disable logging.
        tuner_config: A ``TunerConfig`` instance to enable automatic learning
            rate and/or batch size finding. If ``None``, no tuning is performed
            and user-specified values are used directly.
        checkpoint_monitor: Metric to monitor for checkpointing (e.g.,
            ``"val/accuracy"``). If ``None``, no automatic checkpoint
            callback is added.
        checkpoint_mode: One of ``"min"`` or ``"max"`` for checkpoint
            monitoring.
        callbacks: List of Lightning callbacks.
        default_root_dir: Root directory for logs and checkpoints.
        gradient_clip_val: Gradient clipping value.
        accumulate_grad_batches: Gradient accumulation steps.
        val_check_interval: How often to run validation (1.0 = every epoch).
        enable_checkpointing: Whether to save model checkpoints.
        fast_dev_run: Runs a single batch through train, val, and test to
            find bugs quickly. Can be ``True`` (1 batch), ``False`` (disabled),
            or an integer (number of batches). Useful for debugging.
        **trainer_kwargs: Any other ``pl.Trainer`` argument.

    Example:
        >>> # Manual hyperparameters (no tuning)
        >>> trainer = AutoTrainer(max_epochs=10)
        >>> trainer.fit(model, datamodule=data)

        >>> # With automatic LR and batch size finding
        >>> trainer = AutoTrainer(
        ...     max_epochs=10,
        ...     tuner_config=TunerConfig(
        ...         auto_lr=True,
        ...         auto_batch_size=True,
        ...         lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0},
        ...     ),
        ... )
        >>> trainer.fit(model, datamodule=data)

        >>> # Only LR finding, manual batch size
        >>> trainer = AutoTrainer(
        ...     max_epochs=10,
        ...     tuner_config=TunerConfig(auto_lr=True, auto_batch_size=False),
        ... )

        >>> # Quick debugging with fast_dev_run
        >>> trainer = AutoTrainer(fast_dev_run=True)
        >>> trainer.fit(model, datamodule=data)  # Runs 1 batch only
    """

    def __init__(
        self,
        max_epochs: int = 10,
        accelerator: str = "auto",
        devices: int | str = "auto",
        precision: str | int = 32,
        logger: (
            LoggerManager
            | list[LoggerConfig]
            | pl.loggers.Logger
            | list[pl.loggers.Logger]
            | bool
        ) = False,
        tuner_config: TunerConfig | None = None,
        checkpoint_monitor: str | None = None,
        checkpoint_mode: str = "max",
        callbacks: list[pl.Callback] | None = None,
        default_root_dir: str = ".",
        gradient_clip_val: float | None = None,
        accumulate_grad_batches: int = 1,
        val_check_interval: float | int = 1.0,
        enable_checkpointing: bool = True,
        fast_dev_run: bool | int = False,
        **trainer_kwargs: Any,
    ) -> None:
        if isinstance(logger, LoggerManager):
            resolved_logger = logger.loggers
        elif (
            isinstance(logger, list) and logger and isinstance(logger[0], LoggerConfig)
        ):
            manager = LoggerManager(configs=logger)
            resolved_logger = manager.loggers
        else:
            resolved_logger = logger

        if callbacks is None:
            callbacks = []

        if enable_checkpointing and checkpoint_monitor:
            has_checkpoint_cb = any(
                isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in callbacks
            )
            if not has_checkpoint_cb:
                callbacks.append(
                    pl.callbacks.ModelCheckpoint(
                        monitor=checkpoint_monitor,
                        mode=checkpoint_mode,
                        save_top_k=1,
                        filename=f"best-{{epoch}}-{{{checkpoint_monitor}:.4f}}",
                    )
                )

        self._tuner_config = tuner_config

        super().__init__(
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
            fast_dev_run=fast_dev_run,
            **trainer_kwargs,
        )

    def fit(
        self,
        model: pl.LightningModule,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        """Fit the model, optionally running LR/batch size tuning first.

        If ``tuner_config`` was provided with ``auto_lr=True`` or
        ``auto_batch_size=True``, the respective tuning will run before
        training begins.

        Parameters:
            model: The LightningModule to train.
            train_dataloaders: Train dataloaders (if not using datamodule).
            val_dataloaders: Validation dataloaders (if not using datamodule).
            datamodule: A LightningDataModule instance.
            ckpt_path: Path to checkpoint for resuming training.
        """
        if self._tuner_config is not None:
            self._run_tuning(model, train_dataloaders, val_dataloaders, datamodule)

        super().fit(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    def _run_tuning(
        self,
        model: pl.LightningModule,
        train_dataloaders: Any,
        val_dataloaders: Any,
        datamodule: pl.LightningDataModule | None,
    ) -> None:
        """Run automatic hyperparameter tuning."""
        tuner = Tuner(self)
        config = self._tuner_config

        # Run batch size finder first (if enabled)
        # This should run before LR finder since LR depends on batch size
        if config.auto_batch_size:
            print("Running batch size finder...")
            try:
                result = tuner.scale_batch_size(
                    model,
                    train_dataloaders=train_dataloaders,
                    val_dataloaders=val_dataloaders,
                    datamodule=datamodule,
                    **config.batch_size_kwargs,
                )
                print(f"Optimal batch size found: {result}")
            except Exception as e:
                print(f"Batch size finder failed: {e}")
                print("Continuing with user-specified batch size.")

        # Run LR finder (if enabled)
        if config.auto_lr:
            print("Running learning rate finder...")
            try:
                lr_finder = tuner.lr_find(
                    model,
                    train_dataloaders=train_dataloaders,
                    val_dataloaders=val_dataloaders,
                    datamodule=datamodule,
                    **config.lr_find_kwargs,
                )
                if lr_finder is not None:
                    suggested_lr = lr_finder.suggestion()
                    if suggested_lr is not None:
                        print(f"Suggested learning rate: {suggested_lr:.2e}")
                        # Update model's learning rate
                        if hasattr(model, "_lr"):
                            model._lr = suggested_lr
                        elif hasattr(model, "lr"):
                            model.lr = suggested_lr
                        elif hasattr(model, "hparams") and "lr" in model.hparams:
                            model.hparams.lr = suggested_lr
                    else:
                        print("LR finder could not suggest a learning rate.")
            except Exception as e:
                print(f"LR finder failed: {e}")
                print("Continuing with user-specified learning rate.")

    @property
    def tuner_config(self) -> TunerConfig | None:
        """Return the tuner configuration."""
        return self._tuner_config
