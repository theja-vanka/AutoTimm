"""Convenience wrapper around pl.Trainer for autotimm tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner

from autotimm.loggers import LoggerConfig, LoggerManager
from autotimm.utils import seed_everything

# Module-level flag to ensure watermark is printed only once per session
_WATERMARK_PRINTED = False

import builtins
try:
    from rich.console import Console

    console = Console()
except Exception:
    class _ConsoleFallback:
        def print(self, *args, **kwargs):
            builtins.print(*args, **kwargs)

    console = _ConsoleFallback()


def _print_watermark() -> None:
    """Print environment watermark with package versions."""
    global _WATERMARK_PRINTED
    if _WATERMARK_PRINTED:
        return
    _WATERMARK_PRINTED = True

    try:
        from watermark import watermark

        console.print(watermark(packages="torch,lightning,timm,transformers", python=True))
    except ImportError:
        # Fallback if watermark is not available
        import sys

        import torch

        console.print(f"Python: {sys.version}")
        console.print(f"PyTorch: {torch.__version__}")
        try:
            import lightning

            console.print(f"Lightning: {lightning.__version__}")
        except ImportError:
            try:
                import pytorch_lightning as pl_fallback

                console.print(f"PyTorch Lightning: {pl_fallback.__version__}")
            except Exception:
                pass
        try:
            import timm

            console.print(f"timm: {timm.__version__}")
        except ImportError:
            pass
        try:
            import transformers

            console.print(f"transformers: {transformers.__version__}")
        except ImportError:
            pass
    except Exception:
        # Silently ignore any errors in watermark printing
        pass


@dataclass
class TunerConfig:
    """Configuration for automatic hyperparameter tuning.

    Parameters:
        auto_lr: Whether to use the learning rate finder before training.
            If ``True``, the optimal learning rate will be found and applied.
            If ``False``, the user-specified learning rate is used.
            Default: ``True`` (enabled by default).
        auto_batch_size: Whether to use the batch size finder before training.
            If ``True``, the optimal batch size will be found and applied.
            If ``False``, the user-specified batch size is used.
            Default: ``True`` (enabled by default).
        lr_find_kwargs: Additional kwargs passed to ``Tuner.lr_find()``.
            Common options: ``min_lr``, ``max_lr``, ``num_training``,
            ``mode`` ("exponential" or "linear"), ``early_stop_threshold``.
            Default values are set if not provided.
        batch_size_kwargs: Additional kwargs passed to ``Tuner.scale_batch_size()``.
            Common options: ``mode`` ("power" or "binsearch"), ``steps_per_trial``,
            ``init_val``, ``max_trials``.
            Default values are set if not provided.

    Example:
        >>> # Use defaults (both auto_lr and auto_batch_size disabled)
        >>> config = TunerConfig()

        >>> # Disable auto-tuning
        >>> config = TunerConfig(auto_lr=False, auto_batch_size=False)

        >>> # Custom configuration
        >>> config = TunerConfig(
        ...     auto_lr=True,
        ...     auto_batch_size=True,
        ...     lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1e-1, "num_training": 100},
        ...     batch_size_kwargs={"mode": "power", "init_val": 16},
        ... )
    """

    auto_lr: bool = False
    auto_batch_size: bool = False
    lr_find_kwargs: dict[str, Any] | None = None
    batch_size_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Set sensible defaults for LR finder if not provided
        if self.lr_find_kwargs is None:
            self.lr_find_kwargs = {
                "min_lr": 1e-7,
                "max_lr": 1.0,
                "num_training": 100,
                "mode": "exponential",
                "early_stop_threshold": 4.0,
            }

        # Set sensible defaults for batch size finder if not provided
        if self.batch_size_kwargs is None:
            self.batch_size_kwargs = {
                "mode": "power",
                "steps_per_trial": 3,
                "init_val": 16,
                "max_trials": 25,
            }


class AutoTrainer(pl.Trainer):
    """A configured ``pl.Trainer`` with sensible defaults for autotimm.

    This is a convenience class that wires up the logger, checkpointing,
    and automatic hyperparameter tuning. All ``**trainer_kwargs`` are
    forwarded to ``pl.Trainer``, so any Lightning Trainer argument works.

    **Auto-tuning is disabled by default** - both learning rate and batch size
    finding are disabled unless explicitly enabled via ``tuner_config``.

    Parameters:
        max_epochs: Number of training epochs.
        accelerator: ``"auto"``, ``"gpu"``, ``"cpu"``, ``"tpu"``, etc.
        devices: Number of devices or ``"auto"``.
        precision: Training precision (``32``, ``16``, ``"bf16-mixed"``, etc.).
        logger: A ``LoggerManager`` instance, a list of ``LoggerConfig``
            objects, a pre-built Logger instance/list, or ``False`` to
            disable logging.
        tuner_config: A ``TunerConfig`` instance to configure automatic learning
            rate and/or batch size finding. If ``None``, a default ``TunerConfig()``
            is created with both auto_lr and auto_batch_size disabled.
            To disable auto-tuning completely, pass ``False``.
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
        seed: Random seed for reproducibility. If ``None``, no seeding is performed.
            Default is ``42`` for reproducible results.
        deterministic: If ``True`` (default), enables deterministic algorithms in PyTorch
            for full reproducibility (may impact performance). Set to ``False`` for faster training.
        use_autotimm_seeding: If ``True``, uses AutoTimm's custom ``seed_everything()``
            function which provides comprehensive seeding with deterministic mode support.
            If ``False`` (default), uses PyTorch Lightning's built-in seeding.
        **trainer_kwargs: Any other ``pl.Trainer`` argument.

    Example:
        >>> # Default: auto-tuning disabled (both LR and batch size)
        >>> trainer = AutoTrainer(max_epochs=10)
        >>> trainer.fit(model, datamodule=data)

        >>> # Disable all auto-tuning
        >>> trainer = AutoTrainer(max_epochs=10, tuner_config=False)
        >>> trainer.fit(model, datamodule=data)

        >>> # Custom tuning configuration
        >>> trainer = AutoTrainer(
        ...     max_epochs=10,
        ...     tuner_config=TunerConfig(
        ...         auto_lr=True,
        ...         auto_batch_size=False,
        ...         lr_find_kwargs={"min_lr": 1e-6, "max_lr": 1.0},
        ...     ),
        ... )
        >>> trainer.fit(model, datamodule=data)

        >>> # Quick debugging with fast_dev_run (auto-tuning disabled automatically)
        >>> trainer = AutoTrainer(fast_dev_run=True)
        >>> trainer.fit(model, datamodule=data)  # Runs 1 batch only

        >>> # Custom seeding for reproducibility
        >>> trainer = AutoTrainer(max_epochs=10, seed=123, deterministic=True)
        >>> trainer.fit(model, datamodule=data)

        >>> # Use Lightning's built-in seeding instead of AutoTimm's
        >>> trainer = AutoTrainer(max_epochs=10, seed=42, use_autotimm_seeding=False)
        >>> trainer.fit(model, datamodule=data)

        >>> # Disable seeding completely
        >>> trainer = AutoTrainer(max_epochs=10, seed=None)
        >>> trainer.fit(model, datamodule=data)
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
        tuner_config: TunerConfig | None | bool = None,
        checkpoint_monitor: str | None = None,
        checkpoint_mode: str = "max",
        callbacks: list[pl.Callback] | None = None,
        default_root_dir: str = ".",
        gradient_clip_val: float | None = None,
        accumulate_grad_batches: int = 1,
        val_check_interval: float | int = 1.0,
        enable_checkpointing: bool = True,
        fast_dev_run: bool | int = False,
        seed: int | None = 42,
        deterministic: bool = False,
        use_autotimm_seeding: bool = False,
        **trainer_kwargs: Any,
    ) -> None:
        # Print environment watermark on first AutoTrainer instantiation
        _print_watermark()

        # Handle seeding
        if seed is not None and use_autotimm_seeding:
            # Use AutoTimm's custom seed_everything with deterministic support
            seed_everything(seed, deterministic=deterministic)
        elif seed is not None and not use_autotimm_seeding:
            # Use PyTorch Lightning's built-in seeding (if not already in trainer_kwargs)
            if "seed_everything" not in trainer_kwargs:
                # Pass seed to Lightning's Trainer
                import pytorch_lightning as pl_seed

                pl_seed.seed_everything(seed, workers=True)
                # Note: Lightning doesn't have built-in deterministic parameter,
                # so we still apply our deterministic settings
                if deterministic:
                    import torch

                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

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

        # Configure auto-tuning behavior
        # Disable auto-tuning during fast_dev_run or if explicitly set to False
        if fast_dev_run or tuner_config is False:
            self._tuner_config = None
        elif tuner_config is None:
            # Enable auto-tuning by default with sensible defaults
            self._tuner_config = TunerConfig()
        else:
            self._tuner_config = tuner_config

        # Guard flag: Lightning's Tuner internally calls trainer.fit(), which would
        # re-enter our fit() override and trigger a recursive tuning loop. This flag
        # breaks the cycle so that the inner fit() call skips _run_tuning().
        self._is_tuning = False

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
        if self._tuner_config is not None and not self._is_tuning:
            self._is_tuning = True
            try:
                self._run_tuning(model, train_dataloaders, val_dataloaders, datamodule)
            finally:
                self._is_tuning = False

        super().fit(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    def _remove_tuner_callbacks(self) -> None:
        """Remove BatchSizeFinder/LearningRateFinder callbacks injected by Tuner.

        In Lightning >= 2.x, Tuner methods permanently attach their callbacks
        to the trainer. Calling both scale_batch_size and lr_find sequentially
        (or calling fit() more than once) triggers a conflict. This method
        cleans them up after each tuning step.
        """
        try:
            from pytorch_lightning.callbacks import BatchSizeFinder, LearningRateFinder

            self.callbacks[:] = [
                cb
                for cb in self.callbacks
                if not isinstance(cb, (BatchSizeFinder, LearningRateFinder))
            ]
        except ImportError:
            pass

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
            console.print("Running batch size finder...")
            try:
                result = tuner.scale_batch_size(
                    model,
                    train_dataloaders=train_dataloaders,
                    val_dataloaders=val_dataloaders,
                    datamodule=datamodule,
                    **config.batch_size_kwargs,
                )
                console.print(f"Optimal batch size found: {result}")
            except Exception as e:
                console.print(f"Batch size finder failed: {e}")
                console.print("Continuing with user-specified batch size.")
            finally:
                # Remove the BatchSizeFinder callback Tuner injected so the
                # subsequent lr_find call (and any future fit() call) won't
                # see a duplicate and raise an error.
                self._remove_tuner_callbacks()

        # Run LR finder (if enabled)
        if config.auto_lr:
            console.print("Running learning rate finder...")
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
                        console.print(f"Suggested learning rate: {suggested_lr:.2e}")
                        # Update model's learning rate
                        if hasattr(model, "_lr"):
                            model._lr = suggested_lr
                        elif hasattr(model, "lr"):
                            model.lr = suggested_lr
                        elif hasattr(model, "hparams") and "lr" in model.hparams:
                            model.hparams.lr = suggested_lr
                    else:
                        console.print("LR finder could not suggest a learning rate.")
            except Exception as e:
                console.print(f"LR finder failed: {e}")
                console.print("Continuing with user-specified learning rate.")
            finally:
                # Remove the LearningRateFinder callback so fit() starts clean.
                self._remove_tuner_callbacks()

    @property
    def tuner_config(self) -> TunerConfig | None:
        """Return the tuner configuration."""
        return self._tuner_config
