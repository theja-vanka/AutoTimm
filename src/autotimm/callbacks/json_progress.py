"""JSON progress callback for structured training events on stdout."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch


def _emit(
    event: dict[str, Any],
    log_file: Path | None = None,
) -> None:
    """Write a JSON event line to stdout (and optionally a log file) and flush.

    Stdout writes are wrapped in a try/except so a broken pipe (e.g. Tauri
    crash) does not kill the training process.
    """
    event["timestamp"] = time.time()
    line = json.dumps(event, default=str) + "\n"

    # Always try stdout — swallow BrokenPipeError so training survives
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except BrokenPipeError:
        pass

    # Durable file log — Tauri can tail / replay this on reconnect
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line)


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert tensor values to Python scalars."""
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else v.tolist()
        else:
            out[k] = v
    return out


class JsonProgressCallback(pl.Callback):
    """Lightning Callback that emits NDJSON progress events to stdout.

    Designed for consumption by a Tauri/Preact frontend running the
    training process as a subprocess.

    Parameters:
        emit_every_n_steps: How often to emit ``batch_end`` events.
            Set to 1 to emit on every batch.  Default ``10``.
        log_file: Optional path to an NDJSON file where every event is
            also appended.  Tauri can read this file on reconnect to
            replay missed events.  If ``None`` (default), events are
            only written to stdout.
    """

    def __init__(
        self,
        emit_every_n_steps: int = 10,
        log_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.emit_every_n_steps = emit_every_n_steps
        self._log_file: Path | None = Path(log_file) if log_file is not None else None

    def _emit(self, event: dict[str, Any]) -> None:
        """Emit an event via the module-level helper with our log_file."""
        _emit(event, log_file=self._log_file)

    # ------------------------------------------------------------------
    # Fit lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._emit(
            {
                "event": "training_started",
                "max_epochs": trainer.max_epochs,
                "total_steps": trainer.estimated_stepping_batches,
                "fast_dev_run": bool(trainer.fast_dev_run),
            }
        )

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        metrics = _sanitize_metrics(trainer.callback_metrics)
        self._emit({"event": "training_complete", "final_metrics": metrics})

    # ------------------------------------------------------------------
    # Training epoch / batch
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._emit(
            {
                "event": "epoch_started",
                "epoch": trainer.current_epoch,
                "max_epochs": trainer.max_epochs,
            }
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (batch_idx + 1) % self.emit_every_n_steps != 0:
            return

        loss = None
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            loss = outputs

        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        self._emit(
            {
                "event": "batch_end",
                "epoch": trainer.current_epoch,
                "step": trainer.global_step,
                "total_steps": trainer.estimated_stepping_batches,
                "loss": loss,
            }
        )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        metrics = _sanitize_metrics(trainer.callback_metrics)
        self._emit(
            {
                "event": "epoch_end",
                "epoch": trainer.current_epoch,
                "metrics": metrics,
            }
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        metrics = _sanitize_metrics(trainer.callback_metrics)
        self._emit(
            {
                "event": "validation_end",
                "epoch": trainer.current_epoch,
                "metrics": metrics,
            }
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._emit(
            {
                "event": "training_error",
                "error": str(exception),
            }
        )
