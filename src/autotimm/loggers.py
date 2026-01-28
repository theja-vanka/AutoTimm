"""Factory for experiment loggers (TensorBoard, MLflow, W&B)."""

from __future__ import annotations

from typing import Any

from pytorch_lightning.loggers import Logger


def create_logger(
    backend: str = "tensorboard",
    **kwargs: Any,
) -> Logger | bool:
    """Create a PyTorch Lightning logger.

    Parameters:
        backend: One of ``"tensorboard"``, ``"mlflow"``, ``"wandb"``,
            ``"csv"``, or ``"none"``.
        **kwargs: Forwarded to the logger constructor.

    Returns:
        A ``pytorch_lightning.loggers.Logger`` instance, or ``False``
        when *backend* is ``"none"`` (Lightning interprets ``False``
        as "no logger").

    Raises:
        ImportError: If the requested backend's package is not installed.
        ValueError: If the backend name is unrecognized.
    """
    backend = backend.lower()

    if backend == "none":
        return False

    if backend == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger

        kwargs.setdefault("save_dir", "lightning_logs")
        return TensorBoardLogger(**kwargs)

    if backend == "mlflow":
        try:
            from pytorch_lightning.loggers import MLFlowLogger
        except ImportError:
            raise ImportError(
                "MLflow logger requires mlflow. Install with: "
                "pip install autotimm[mlflow]"
            ) from None
        kwargs.setdefault("experiment_name", "autotimm")
        return MLFlowLogger(**kwargs)

    if backend == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:
            raise ImportError(
                "W&B logger requires wandb. Install with: "
                "pip install autotimm[wandb]"
            ) from None
        kwargs.setdefault("project", "autotimm")
        return WandbLogger(**kwargs)

    if backend == "csv":
        from pytorch_lightning.loggers import CSVLogger

        kwargs.setdefault("save_dir", "lightning_logs")
        return CSVLogger(**kwargs)

    raise ValueError(
        f"Unknown logger backend '{backend}'. "
        f"Choose from: tensorboard, mlflow, wandb, csv, none."
    )
