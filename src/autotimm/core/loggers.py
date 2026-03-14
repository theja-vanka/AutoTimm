"""Factory for experiment loggers (TensorBoard, MLflow, W&B, CSV)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pytorch_lightning.loggers import Logger


@dataclass
class LoggerConfig:
    """Configuration for a single logger backend.

    All parameters are required - no defaults are provided to ensure
    explicit configuration.

    Parameters:
        backend: Logger backend type. One of ``"tensorboard"``, ``"mlflow"``,
            ``"wandb"``, or ``"csv"``.
        params: Parameters passed to the logger constructor. Required keys
            depend on the backend.

    Example:
        >>> config = LoggerConfig(
        ...     backend="tensorboard",
        ...     params={"save_dir": "logs", "name": "experiment_1"},
        ... )
    """

    backend: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend is required")
        self.backend = self.backend.lower()
        valid_backends = {"tensorboard", "mlflow", "wandb", "csv"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                f"Valid backends: {', '.join(sorted(valid_backends))}"
            )


class LoggerManager:
    """Manages multiple PyTorch Lightning loggers.

    This class creates and manages multiple logger instances from explicit
    configurations. No default values are provided - all configuration
    must be specified by the user.

    Parameters:
        configs: List of ``LoggerConfig`` objects defining each logger.

    Attributes:
        loggers: List of instantiated PyTorch Lightning logger objects.

    Example:
        >>> manager = LoggerManager(
        ...     configs=[
        ...         LoggerConfig(
        ...             backend="tensorboard",
        ...             params={"save_dir": "logs/tb", "name": "run_1"},
        ...         ),
        ...         LoggerConfig(
        ...             backend="wandb",
        ...             params={"project": "my_project", "name": "run_1"},
        ...         ),
        ...     ]
        ... )
        >>> trainer = pl.Trainer(logger=manager.loggers)
    """

    def __init__(self, configs: list[LoggerConfig]) -> None:
        if not configs:
            raise ValueError("At least one LoggerConfig is required")

        self._configs = configs
        self._loggers: list[Logger] = []
        self._initialize_loggers()

    def _initialize_loggers(self) -> None:
        """Initialize all logger instances from configs."""
        for config in self._configs:
            logger = self._create_logger(config)
            self._loggers.append(logger)

    def _create_logger(self, config: LoggerConfig) -> Logger:
        """Create a single logger instance from config."""
        backend = config.backend
        params = config.params

        if backend == "tensorboard":
            from pytorch_lightning.loggers import TensorBoardLogger

            self._validate_required_params(params, ["save_dir"], "tensorboard")
            return TensorBoardLogger(**params)

        if backend == "mlflow":
            try:
                from pytorch_lightning.loggers import MLFlowLogger
            except ImportError:
                raise ImportError(
                    "MLflow logger requires mlflow. Install with: pip install mlflow"
                ) from None
            self._validate_required_params(params, ["experiment_name"], "mlflow")
            return MLFlowLogger(**params)

        if backend == "wandb":
            try:
                from pytorch_lightning.loggers import WandbLogger
            except ImportError:
                raise ImportError(
                    "W&B logger requires wandb. Install with: pip install wandb"
                ) from None
            self._validate_required_params(params, ["project"], "wandb")
            return WandbLogger(**params)

        if backend == "csv":
            from pytorch_lightning.loggers import CSVLogger

            self._validate_required_params(params, ["save_dir"], "csv")
            return CSVLogger(**params)

        raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def _validate_required_params(
        params: dict[str, Any],
        required: list[str],
        backend: str,
    ) -> None:
        """Validate that required parameters are present."""
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(
                f"Missing required parameters for {backend} logger: {', '.join(missing)}"
            )

    @property
    def loggers(self) -> list[Logger]:
        """Return list of instantiated loggers for use with pl.Trainer."""
        return self._loggers

    @property
    def configs(self) -> list[LoggerConfig]:
        """Return the configurations used to create the loggers."""
        return self._configs

    def __len__(self) -> int:
        """Return the number of loggers."""
        return len(self._loggers)

    def __iter__(self):
        """Iterate over the loggers."""
        return iter(self._loggers)

    def __getitem__(self, index: int) -> Logger:
        """Get a logger by index."""
        return self._loggers[index]

    def get_logger_by_backend(self, backend: str) -> Logger | None:
        """Get the first logger matching the given backend type.

        Parameters:
            backend: Backend name to search for.

        Returns:
            The first matching logger, or None if not found.
        """
        backend = backend.lower()
        for config, logger in zip(self._configs, self._loggers):
            if config.backend == backend:
                return logger
        return None
