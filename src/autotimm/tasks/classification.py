"""Image classification task as a PyTorch Lightning module."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from autotimm.backbone import BackboneConfig, create_backbone, get_backbone_out_features
from autotimm.data.transform_config import TransformConfig
from autotimm.heads import ClassificationHead
from autotimm.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.tasks.preprocessing_mixin import PreprocessingMixin
from autotimm.utils import seed_everything


class ImageClassifier(PreprocessingMixin, pl.LightningModule):
    """End-to-end image classifier backed by a timm backbone.

    Parameters:
        backbone: A timm model name (str) or a :class:`BackboneConfig`.
        num_classes: Number of target classes.
        metrics: A :class:`MetricManager` instance or list of :class:`MetricConfig`
            objects. Optional - if ``None``, no metrics will be computed during training.
            This is useful for inference-only scenarios.
        logging_config: Optional :class:`LoggingConfig` for enhanced logging.
        transform_config: Optional :class:`TransformConfig` for unified transform
            configuration. When provided, enables the ``preprocess()`` method
            for inference-time preprocessing using model-specific normalization.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        optimizer: Optimizer name (``"adamw"``, ``"adam"``, ``"sgd"``, etc.) or dict
            with ``"class"`` (fully qualified class path) and ``"params"`` keys.
            Supports both torch.optim and timm optimizers.
        optimizer_kwargs: Additional kwargs for the optimizer (merged with lr and weight_decay).
        scheduler: Scheduler name (``"cosine"``, ``"step"``, ``"onecycle"``, ``"cosine_with_restarts"``, etc.),
            dict with ``"class"`` and ``"params"`` keys, or ``None`` for no scheduler.
            Supports both torch.optim.lr_scheduler and timm schedulers.
        scheduler_kwargs: Extra kwargs forwarded to the LR scheduler.
        head_dropout: Dropout before the classification linear layer.
        label_smoothing: Label smoothing factor for cross-entropy.
        freeze_backbone: If ``True``, backbone parameters are frozen
            (useful for linear probing).
        mixup_alpha: If > 0, apply Mixup augmentation with this alpha.
        compile_model: If ``True`` (default), apply ``torch.compile()`` to the backbone and head
            for faster inference and training. Requires PyTorch 2.0+.
        compile_kwargs: Optional dict of kwargs to pass to ``torch.compile()``.
            Common options: ``mode`` (``"default"``, ``"reduce-overhead"``, ``"max-autotune"``),
            ``fullgraph`` (``True``/``False``), ``dynamic`` (``True``/``False``).
        seed: Random seed for reproducibility. If ``None``, no seeding is performed.
            Default is ``42`` for reproducible results.
        deterministic: If ``True`` (default), enables deterministic algorithms in PyTorch for full
            reproducibility (may impact performance). Set to ``False`` for faster training.

    Example:
        >>> # For training with metrics
        >>> model = ImageClassifier(
        ...     backbone="resnet50",
        ...     num_classes=10,
        ...     metrics=[
        ...         MetricConfig(
        ...             name="accuracy",
        ...             backend="torchmetrics",
        ...             metric_class="Accuracy",
        ...             params={"task": "multiclass"},
        ...             stages=["train", "val", "test"],
        ...             prog_bar=True,
        ...         ),
        ...     ],
        ...     logging_config=LoggingConfig(
        ...         log_learning_rate=True,
        ...         log_gradient_norm=False,
        ...     ),
        ...     transform_config=TransformConfig(use_timm_config=True),
        ... )
        >>>
        >>> # For inference only (no metrics needed)
        >>> model = ImageClassifier(
        ...     backbone="resnet50",
        ...     num_classes=10,
        ...     transform_config=TransformConfig(use_timm_config=True),
        ... )
        >>> # With transform_config, you can preprocess raw images
        >>> from PIL import Image
        >>> img = Image.open("test.jpg")
        >>> tensor = model.preprocess(img)  # Returns (1, 3, 224, 224)
        >>> output = model(tensor)
    """

    def __init__(
        self,
        backbone: str | BackboneConfig,
        num_classes: int,
        metrics: MetricManager | list[MetricConfig] | None = None,
        logging_config: LoggingConfig | None = None,
        transform_config: TransformConfig | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str | dict[str, Any] = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: str | dict[str, Any] | None = "cosine",
        scheduler_kwargs: dict[str, Any] | None = None,
        head_dropout: float = 0.0,
        label_smoothing: float = 0.0,
        freeze_backbone: bool = False,
        mixup_alpha: float = 0.0,
        compile_model: bool = True,
        compile_kwargs: dict[str, Any] | None = None,
        seed: int | None = 42,
        deterministic: bool = True,
    ):
        # Seed for reproducibility
        if seed is not None:
            seed_everything(seed, deterministic=deterministic)

        super().__init__()
        self.save_hyperparameters(
            ignore=["metrics", "logging_config", "transform_config"]
        )

        # Backbone and head
        self.backbone = create_backbone(backbone)
        in_features = get_backbone_out_features(self.backbone)
        self.head = ClassificationHead(in_features, num_classes, dropout=head_dropout)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Initialize metrics from config
        if metrics is not None:
            if isinstance(metrics, list):
                metrics = MetricManager(configs=metrics, num_classes=num_classes)
            self._metric_manager = metrics
            # Register metrics as ModuleDicts for proper device handling
            self.train_metrics = metrics.get_train_metrics()
            self.val_metrics = metrics.get_val_metrics()
            self.test_metrics = metrics.get_test_metrics()
        else:
            self._metric_manager = None
            # Create empty ModuleDicts when no metrics are provided
            self.train_metrics = nn.ModuleDict()
            self.val_metrics = nn.ModuleDict()
            self.test_metrics = nn.ModuleDict()

        # Logging configuration
        self._logging_config = logging_config or LoggingConfig(
            log_learning_rate=False,
            log_gradient_norm=False,
        )

        # For confusion matrix logging
        if self._logging_config.log_confusion_matrix:
            self._val_confusion = torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=num_classes
            )

        # Store hyperparameters
        self._lr = lr
        self._weight_decay = weight_decay
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler = scheduler
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._mixup_alpha = mixup_alpha
        self._num_classes = num_classes

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Apply torch.compile for optimization (PyTorch 2.0+)
        if compile_model:
            try:
                compile_opts = compile_kwargs or {}
                self.backbone = torch.compile(self.backbone, **compile_opts)
                self.head = torch.compile(self.head, **compile_opts)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"torch.compile failed: {e}. Continuing without compilation. "
                    f"Ensure you have PyTorch 2.0+ for compile support.",
                    stacklevel=2,
                )

        # Setup transforms from config (PreprocessingMixin)
        self._setup_transforms(transform_config, task="classification")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        if self._mixup_alpha > 0 and self.training:
            lam = (
                torch.distributions.Beta(self._mixup_alpha, self._mixup_alpha)
                .sample()
                .to(x.device)
            )
            idx = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1 - lam) * x[idx]
            y_a, y_b = y, y[idx]
            logits = self(x)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(
                logits, y_b
            )
        else:
            logits = self(x)
            loss = self.criterion(logits, y)

        preds = logits.argmax(dim=-1)

        # Log loss
        self.log("train/loss", loss, prog_bar=True)

        # Update and log all train metrics
        if self._metric_manager is not None:
            for name, metric in self.train_metrics.items():
                config = self._metric_manager.get_metric_config("train", name)
                metric.update(preds, y)
                if config:
                    self.log(
                        f"train/{name}",
                        metric,
                        on_step=config.log_on_step,
                        on_epoch=config.log_on_epoch,
                        prog_bar=config.prog_bar,
                    )

        # Enhanced logging: learning rate
        if self._logging_config.log_learning_rate:
            opt = self.optimizers()
            if opt is not None and hasattr(opt, "param_groups"):
                lr = opt.param_groups[0]["lr"]
                self.log("train/lr", lr, on_step=True, on_epoch=False)

        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Hook for gradient norm logging."""
        if self._logging_config.log_gradient_norm:
            grad_norm = self._compute_gradient_norm()
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

        if self._logging_config.log_weight_norm:
            weight_norm = self._compute_weight_norm()
            self.log("train/weight_norm", weight_norm, on_step=True, on_epoch=False)

    def _compute_gradient_norm(self) -> torch.Tensor:
        """Compute the total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return torch.tensor(total_norm**0.5, device=self.device)

    def _compute_weight_norm(self) -> torch.Tensor:
        """Compute the total weight norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return torch.tensor(total_norm**0.5, device=self.device)

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)

        # Log loss
        self.log("val/loss", loss, prog_bar=True)

        # Update and log all val metrics
        if self._metric_manager is not None:
            for name, metric in self.val_metrics.items():
                config = self._metric_manager.get_metric_config("val", name)
                metric.update(preds, y)
                if config:
                    self.log(
                        f"val/{name}",
                        metric,
                        on_step=config.log_on_step,
                        on_epoch=config.log_on_epoch,
                        prog_bar=config.prog_bar,
                    )

        # Update confusion matrix if enabled
        if self._logging_config.log_confusion_matrix:
            self._val_confusion.update(preds, y)

    def on_validation_epoch_end(self) -> None:
        """Log confusion matrix at the end of validation epoch."""
        if not self._logging_config.log_confusion_matrix:
            return

        if self.logger is None:
            return

        cm = self._val_confusion.compute()

        # Log as image if matplotlib is available
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = self._create_confusion_matrix_figure(cm.cpu().numpy())

            # Try to log to tensorboard or other loggers
            if hasattr(self.logger, "experiment") and hasattr(
                self.logger.experiment, "add_figure"
            ):
                self.logger.experiment.add_figure(
                    "val/confusion_matrix", fig, self.current_epoch
                )

            plt.close(fig)
        except ImportError:
            # matplotlib not available, skip confusion matrix visualization
            pass

        self._val_confusion.reset()

    def _create_confusion_matrix_figure(self, cm_np: Any) -> Any:
        """Create a professional confusion matrix visualization.

        Parameters:
            cm_np: Confusion matrix as numpy array.

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Calculate normalized confusion matrix for percentages
        cm_normalized = cm_np.astype("float") / (
            cm_np.sum(axis=1)[:, np.newaxis] + 1e-10
        )

        # Create figure with appropriate size
        fig_size = max(10, min(20, self._num_classes * 0.8))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Use a professional color scheme
        cmap = plt.cm.Blues
        im = ax.imshow(
            cm_normalized, interpolation="nearest", cmap=cmap, vmin=0, vmax=1
        )

        # Add colorbar with label
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            "Normalized Count (Recall)", rotation=270, labelpad=25, fontsize=12
        )

        # Set ticks and labels
        num_classes = cm_np.shape[0]
        tick_marks = np.arange(num_classes)

        # Create class labels (use class indices if no names available)
        class_labels = [str(i) for i in range(num_classes)]

        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_labels, fontsize=10, rotation=45, ha="right")
        ax.set_yticklabels(class_labels, fontsize=10)

        # Add text annotations to each cell
        thresh = cm_normalized.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                count = cm_np[i, j]
                percentage = cm_normalized[i, j] * 100

                # Choose text color based on background
                color = "white" if cm_normalized[i, j] > thresh else "black"

                # Display count and percentage
                text_str = f"{int(count)}\n({percentage:.1f}%)"
                ax.text(
                    j,
                    i,
                    text_str,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=max(8, min(12, 100 // num_classes)),
                    weight="bold" if i == j else "normal",
                )

        # Calculate overall accuracy
        accuracy = np.trace(cm_np) / (np.sum(cm_np) + 1e-10) * 100

        # Set labels and title with metrics
        ax.set_xlabel("Predicted Label", fontsize=14, weight="bold")
        ax.set_ylabel("True Label", fontsize=14, weight="bold")
        ax.set_title(
            f"Confusion Matrix - Epoch {self.current_epoch}\n"
            f"Validation Accuracy: {accuracy:.2f}%",
            fontsize=16,
            weight="bold",
            pad=20,
        )

        # Add grid for better readability
        ax.set_xticks(tick_marks - 0.5, minor=True)
        ax.set_yticks(tick_marks - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        return fig

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)

        # Log loss
        self.log("test/loss", loss)

        # Update and log all test metrics
        if self._metric_manager is not None:
            for name, metric in self.test_metrics.items():
                config = self._metric_manager.get_metric_config("test", name)
                metric.update(preds, y)
                if config:
                    self.log(
                        f"test/{name}",
                        metric,
                        on_step=config.log_on_step,
                        on_epoch=config.log_on_epoch,
                        prog_bar=config.prog_bar,
                    )

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self(x).softmax(dim=-1)

    def to_torchscript(
        self,
        save_path: str | None = None,
        example_input: torch.Tensor | None = None,
        method: str = "trace",
        **kwargs: Any,
    ) -> torch.jit.ScriptModule:
        """Export model to TorchScript format.

        Args:
            save_path: Optional path to save the TorchScript model. If None, returns compiled model without saving.
            example_input: Example input tensor for tracing. If None, uses default shape (1, 3, 224, 224).
            method: Export method ("trace" or "script"). Default is "trace".
            **kwargs: Additional arguments passed to export_to_torchscript.

        Returns:
            Compiled TorchScript module.

        Example:
            >>> model = ImageClassifier(backbone="resnet50", num_classes=10)
            >>> scripted = model.to_torchscript("model.pt")
        """
        from autotimm.export import export_to_torchscript

        if example_input is None:
            example_input = torch.randn(1, 3, 224, 224)

        if save_path is None:
            # Return scripted model without saving
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                scripted = export_to_torchscript(
                    self, tmp.name, example_input, method, **kwargs
                )
                import os

                os.unlink(tmp.name)
                return scripted
        else:
            return export_to_torchscript(
                self, save_path, example_input, method, **kwargs
            )

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler.

        Supports torch.optim, timm optimizers, and custom optimizers/schedulers.
        """
        params = filter(lambda p: p.requires_grad, self.parameters())

        # Create optimizer
        optimizer = self._create_optimizer(params)

        # Return early if no scheduler
        if self._scheduler is None or self._scheduler == "none":
            return {"optimizer": optimizer}

        # Create scheduler
        scheduler_config = self._create_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _create_optimizer(self, params) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        # Prepare base kwargs
        opt_kwargs = {"lr": self._lr, "weight_decay": self._weight_decay}
        opt_kwargs.update(self._optimizer_kwargs)

        # Dict config: {"class": "path.to.Optimizer", "params": {...}}
        if isinstance(self._optimizer, dict):
            opt_class_path = self._optimizer["class"]
            opt_params = self._optimizer.get("params", {})
            opt_kwargs.update(opt_params)

            # Import and instantiate
            optimizer_cls = self._import_class(opt_class_path)
            return optimizer_cls(params, **opt_kwargs)

        # String name: try torch.optim first, then timm
        optimizer_name = self._optimizer.lower()

        # Torch optimizers
        torch_optimizers = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }

        if optimizer_name in torch_optimizers:
            return torch_optimizers[optimizer_name](params, **opt_kwargs)

        # Try timm optimizers
        try:
            import timm.optim as timm_optim

            timm_optimizers = {
                "adamp": timm_optim.AdamP,
                "sgdp": timm_optim.SGDP,
                "adabelief": timm_optim.AdaBelief,
                "radam": timm_optim.RAdam,
                "adahessian": timm_optim.Adahessian,
                "lamb": timm_optim.Lamb,
                "lars": timm_optim.Lars,
                "madgrad": timm_optim.MADGRAD,
                "novograd": timm_optim.NovGrad,
            }

            if optimizer_name in timm_optimizers:
                return timm_optimizers[optimizer_name](params, **opt_kwargs)
        except ImportError:
            pass

        raise ValueError(
            f"Unknown optimizer: {self._optimizer}. "
            f"Use a torch.optim optimizer name, timm optimizer name, "
            f"or provide a dict with 'class' and 'params' keys."
        )

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        """Create scheduler config from optimizer."""
        sched_kwargs = self._scheduler_kwargs.copy()
        interval = sched_kwargs.pop("interval", "step")
        frequency = sched_kwargs.pop("frequency", 1)

        # Dict config: {"class": "path.to.Scheduler", "params": {...}}
        if isinstance(self._scheduler, dict):
            sched_class_path = self._scheduler["class"]
            sched_params = self._scheduler.get("params", {})
            sched_kwargs.update(sched_params)

            scheduler_cls = self._import_class(sched_class_path)
            scheduler = scheduler_cls(optimizer, **sched_kwargs)

            return {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": frequency,
            }

        # String name: try torch schedulers first, then timm
        scheduler_name = self._scheduler.lower()

        # Torch schedulers
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_kwargs.pop(
                    "T_max", self.trainer.estimated_stepping_batches
                ),
                **sched_kwargs,
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_kwargs.pop("step_size", 30),
                gamma=sched_kwargs.pop("gamma", 0.1),
                **sched_kwargs,
            )
        elif scheduler_name == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_kwargs.pop("milestones", [30, 60, 90]),
                gamma=sched_kwargs.pop("gamma", 0.1),
                **sched_kwargs,
            )
        elif scheduler_name == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sched_kwargs.pop("gamma", 0.95),
                **sched_kwargs,
            )
        elif scheduler_name == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=sched_kwargs.pop("max_lr", self._lr * 10),
                total_steps=sched_kwargs.pop(
                    "total_steps", self.trainer.estimated_stepping_batches
                ),
                **sched_kwargs,
            )
        else:
            # Try timm schedulers
            try:
                import timm.scheduler as timm_scheduler

                if scheduler_name == "cosine_with_restarts":
                    scheduler = timm_scheduler.CosineLRScheduler(
                        optimizer,
                        t_initial=sched_kwargs.pop(
                            "t_initial", self.trainer.max_epochs
                        ),
                        cycle_limit=sched_kwargs.pop("cycle_limit", 1),
                        **sched_kwargs,
                    )
                    interval = "epoch"
                elif scheduler_name == "plateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=sched_kwargs.pop("mode", "min"),
                        factor=sched_kwargs.pop("factor", 0.1),
                        patience=sched_kwargs.pop("patience", 10),
                        **sched_kwargs,
                    )
                    interval = "epoch"
                    return {
                        "scheduler": scheduler,
                        "monitor": sched_kwargs.pop("monitor", "val/loss"),
                        "interval": interval,
                        "frequency": frequency,
                    }
                else:
                    raise ValueError(f"Unknown scheduler: {self._scheduler}")
            except (ImportError, ValueError):
                raise ValueError(
                    f"Unknown scheduler: {self._scheduler}. "
                    f"Use a torch.optim.lr_scheduler name, timm scheduler name, "
                    f"or provide a dict with 'class' and 'params' keys."
                )

        return {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
        }

    def _import_class(self, class_path: str):
        """Import a class from a fully qualified path."""
        import importlib

        if "." not in class_path:
            raise ValueError(
                f"Class path must be fully qualified (e.g., 'torch.optim.Adam'), "
                f"got: {class_path}"
            )

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
