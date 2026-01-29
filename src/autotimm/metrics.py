"""Factory for training metrics (torchmetrics and timm)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torchmetrics


@dataclass
class MetricConfig:
    """Configuration for a single metric.

    All parameters are required - no defaults are provided to ensure
    explicit configuration.

    Parameters:
        name: Unique identifier for this metric (used in logging).
        backend: Metric backend type. One of ``"torchmetrics"``, ``"timm"``,
            or ``"custom"``.
        metric_class: The metric class name (for torchmetrics) or function
            name (for timm).
        params: Parameters passed to the metric constructor/function.
        stages: List of stages where this metric applies: ``"train"``,
            ``"val"``, ``"test"``.
        log_on_step: Whether to log on each step.
        log_on_epoch: Whether to log on epoch end.
        prog_bar: Whether to show in progress bar.

    Example:
        >>> config = MetricConfig(
        ...     name="accuracy",
        ...     backend="torchmetrics",
        ...     metric_class="Accuracy",
        ...     params={"task": "multiclass"},
        ...     stages=["train", "val", "test"],
        ...     prog_bar=True,
        ... )
    """

    name: str
    backend: str
    metric_class: str
    params: dict[str, Any]
    stages: list[str]
    log_on_step: bool = False
    log_on_epoch: bool = True
    prog_bar: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name is required")
        if not self.backend:
            raise ValueError("backend is required")
        if not self.metric_class:
            raise ValueError("metric_class is required")
        if not self.stages:
            raise ValueError("stages is required (e.g., ['train', 'val', 'test'])")

        self.backend = self.backend.lower()
        valid_backends = {"torchmetrics", "timm", "custom"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                f"Valid backends: {', '.join(sorted(valid_backends))}"
            )

        valid_stages = {"train", "val", "test"}
        for stage in self.stages:
            if stage.lower() not in valid_stages:
                raise ValueError(
                    f"Unknown stage '{stage}'. "
                    f"Valid stages: {', '.join(sorted(valid_stages))}"
                )
        self.stages = [s.lower() for s in self.stages]


@dataclass
class LoggingConfig:
    """Configuration for enhanced logging during training.

    Parameters:
        log_learning_rate: Whether to log the current learning rate.
        log_gradient_norm: Whether to log gradient norms.
        log_weight_norm: Whether to log weight norms.
        log_confusion_matrix: Whether to log confusion matrix at epoch end.
        log_predictions: Whether to log sample predictions/images.
        predictions_per_epoch: Number of sample predictions to log per epoch.
        verbosity: Logging verbosity level (0=minimal, 1=normal, 2=verbose).

    Example:
        >>> config = LoggingConfig(
        ...     log_learning_rate=True,
        ...     log_gradient_norm=True,
        ...     log_confusion_matrix=True,
        ... )
    """

    log_learning_rate: bool
    log_gradient_norm: bool
    log_weight_norm: bool = False
    log_confusion_matrix: bool = False
    log_predictions: bool = False
    predictions_per_epoch: int = 8
    verbosity: int = 1

    def __post_init__(self) -> None:
        if self.verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1, or 2")
        if self.predictions_per_epoch < 0:
            raise ValueError("predictions_per_epoch must be non-negative")


class TimmMetricWrapper(torch.nn.Module):
    """Wrapper to make timm metric functions compatible with Lightning.

    timm metrics are functional (not stateful like torchmetrics), so this
    wrapper accumulates predictions and targets across steps.

    Parameters:
        fn: The timm metric function to wrap.
        params: Parameters to pass to the function.

    Example:
        >>> from timm.utils.metrics import accuracy
        >>> wrapper = TimmMetricWrapper(fn=accuracy, params={"topk": (1, 5)})
        >>> wrapper.update(logits, targets)
        >>> result = wrapper.compute()
    """

    def __init__(self, fn: Callable, params: dict[str, Any]) -> None:
        super().__init__()
        self._fn = fn
        self._params = params
        self._outputs: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions and targets."""
        self._outputs.append(preds.detach())
        self._targets.append(targets.detach())

    def compute(self) -> torch.Tensor:
        """Compute the metric over all accumulated data."""
        if not self._outputs:
            return torch.tensor(0.0)

        outputs = torch.cat(self._outputs, dim=0)
        targets = torch.cat(self._targets, dim=0)

        topk = self._params.get("topk", (1,))
        result = self._fn(outputs, targets, topk=topk)

        # Return first top-k accuracy (usually top-1), normalized to 0-1
        return result[0] / 100.0

    def reset(self) -> None:
        """Clear accumulated data."""
        self._outputs.clear()
        self._targets.clear()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Update and return current computed value."""
        self.update(preds, targets)
        return self.compute()


class MetricManager:
    """Manages multiple metrics for training/validation/testing.

    This class creates and manages metric instances from explicit configurations.
    No default values are provided - all configuration must be specified.

    Parameters:
        configs: List of ``MetricConfig`` objects defining each metric.
        num_classes: Number of classes (required for classification metrics).

    Attributes:
        train_metrics: Dict of metrics for training stage.
        val_metrics: Dict of metrics for validation stage.
        test_metrics: Dict of metrics for test stage.

    Example:
        >>> manager = MetricManager(
        ...     configs=[
        ...         MetricConfig(
        ...             name="accuracy",
        ...             backend="torchmetrics",
        ...             metric_class="Accuracy",
        ...             params={"task": "multiclass"},
        ...             stages=["train", "val", "test"],
        ...             prog_bar=True,
        ...         ),
        ...         MetricConfig(
        ...             name="f1",
        ...             backend="torchmetrics",
        ...             metric_class="F1Score",
        ...             params={"task": "multiclass", "average": "macro"},
        ...             stages=["val"],
        ...         ),
        ...     ],
        ...     num_classes=10,
        ... )
        >>> train_metrics = manager.get_train_metrics()
    """

    def __init__(self, configs: list[MetricConfig], num_classes: int) -> None:
        if not configs:
            raise ValueError("At least one MetricConfig is required")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self._configs = configs
        self._num_classes = num_classes

        # Store metrics with their configs by stage
        self._train_metrics: dict[str, tuple[torch.nn.Module, MetricConfig]] = {}
        self._val_metrics: dict[str, tuple[torch.nn.Module, MetricConfig]] = {}
        self._test_metrics: dict[str, tuple[torch.nn.Module, MetricConfig]] = {}

        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize all metric instances from configs."""
        for config in self._configs:
            if "train" in config.stages:
                metric = self._create_metric(config)
                self._train_metrics[config.name] = (metric, config)
            if "val" in config.stages:
                metric = self._create_metric(config)
                self._val_metrics[config.name] = (metric, config)
            if "test" in config.stages:
                metric = self._create_metric(config)
                self._test_metrics[config.name] = (metric, config)

    def _create_metric(self, config: MetricConfig) -> torch.nn.Module:
        """Create a single metric instance from config."""
        backend = config.backend
        params = config.params.copy()

        # Inject num_classes if not specified
        if "num_classes" not in params:
            params["num_classes"] = self._num_classes

        if backend == "torchmetrics":
            return self._create_torchmetrics_metric(config.metric_class, params)
        elif backend == "timm":
            return self._create_timm_metric(config.metric_class, params)
        elif backend == "custom":
            return self._create_custom_metric(config.metric_class, params)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _create_torchmetrics_metric(
        self, metric_class: str, params: dict[str, Any]
    ) -> torchmetrics.Metric:
        """Create a torchmetrics Metric instance."""
        if not hasattr(torchmetrics, metric_class):
            raise ValueError(
                f"Unknown torchmetrics metric: {metric_class}. "
                f"Check torchmetrics documentation for available metrics."
            )
        metric_cls = getattr(torchmetrics, metric_class)
        return metric_cls(**params)

    def _create_timm_metric(
        self, metric_class: str, params: dict[str, Any]
    ) -> TimmMetricWrapper:
        """Create a wrapper around timm metric functions."""
        try:
            from timm.utils.metrics import accuracy as timm_accuracy
        except ImportError:
            raise ImportError(
                "timm metrics require timm to be installed. "
                "Install with: pip install timm"
            ) from None

        timm_metrics = {
            "accuracy": timm_accuracy,
        }

        if metric_class.lower() not in timm_metrics:
            raise ValueError(
                f"Unknown timm metric: {metric_class}. "
                f"Available: {', '.join(timm_metrics.keys())}"
            )

        # Remove num_classes as timm accuracy doesn't use it
        params_copy = {k: v for k, v in params.items() if k != "num_classes"}

        return TimmMetricWrapper(
            fn=timm_metrics[metric_class.lower()],
            params=params_copy,
        )

    def _create_custom_metric(
        self, metric_class: str, params: dict[str, Any]
    ) -> torch.nn.Module:
        """Create a custom metric from a fully qualified class path."""
        import importlib

        if "." not in metric_class:
            raise ValueError(
                f"custom metric_class must be a fully qualified path "
                f"(e.g., 'mypackage.metrics.CustomMetric'), got: {metric_class}"
            )

        module_path, class_name = metric_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        metric_cls = getattr(module, class_name)
        return metric_cls(**params)

    def get_train_metrics(self) -> torch.nn.ModuleDict:
        """Return ModuleDict of train metrics for Lightning module."""
        return torch.nn.ModuleDict(
            {name: metric for name, (metric, _) in self._train_metrics.items()}
        )

    def get_val_metrics(self) -> torch.nn.ModuleDict:
        """Return ModuleDict of validation metrics for Lightning module."""
        return torch.nn.ModuleDict(
            {name: metric for name, (metric, _) in self._val_metrics.items()}
        )

    def get_test_metrics(self) -> torch.nn.ModuleDict:
        """Return ModuleDict of test metrics for Lightning module."""
        return torch.nn.ModuleDict(
            {name: metric for name, (metric, _) in self._test_metrics.items()}
        )

    def get_metric_config(self, stage: str, name: str) -> MetricConfig | None:
        """Get the config for a specific metric.

        Parameters:
            stage: One of "train", "val", "test".
            name: The metric name.

        Returns:
            The MetricConfig if found, None otherwise.
        """
        metrics_dict = {
            "train": self._train_metrics,
            "val": self._val_metrics,
            "test": self._test_metrics,
        }.get(stage, {})

        if name in metrics_dict:
            return metrics_dict[name][1]
        return None

    @property
    def configs(self) -> list[MetricConfig]:
        """Return the configurations used to create the metrics."""
        return self._configs

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self._num_classes

    def __len__(self) -> int:
        """Return total number of unique metrics across all stages."""
        return len(self._configs)
