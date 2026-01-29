"""Image classification task as a PyTorch Lightning module."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from autotimm.backbone import BackboneConfig, create_backbone, get_backbone_out_features
from autotimm.heads import ClassificationHead
from autotimm.metrics import LoggingConfig, MetricConfig, MetricManager, TimmMetricWrapper


class ImageClassifier(pl.LightningModule):
    """End-to-end image classifier backed by a timm backbone.

    Parameters:
        backbone: A timm model name (str) or a :class:`BackboneConfig`.
        num_classes: Number of target classes.
        metrics: A :class:`MetricManager` instance or list of :class:`MetricConfig`
            objects. Required - no default metrics are provided.
        logging_config: Optional :class:`LoggingConfig` for enhanced logging.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        scheduler: One of ``"cosine"``, ``"step"``, ``"none"``.
        scheduler_kwargs: Extra kwargs forwarded to the LR scheduler.
        head_dropout: Dropout before the classification linear layer.
        label_smoothing: Label smoothing factor for cross-entropy.
        freeze_backbone: If ``True``, backbone parameters are frozen
            (useful for linear probing).
        mixup_alpha: If > 0, apply Mixup augmentation with this alpha.

    Example:
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
        ... )
    """

    def __init__(
        self,
        backbone: str | BackboneConfig,
        num_classes: int,
        metrics: MetricManager | list[MetricConfig],
        logging_config: LoggingConfig | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        scheduler_kwargs: dict[str, Any] | None = None,
        head_dropout: float = 0.0,
        label_smoothing: float = 0.0,
        freeze_backbone: bool = False,
        mixup_alpha: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["metrics", "logging_config"])

        # Backbone and head
        self.backbone = create_backbone(backbone)
        in_features = get_backbone_out_features(self.backbone)
        self.head = ClassificationHead(in_features, num_classes, dropout=head_dropout)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Initialize metrics from config
        if isinstance(metrics, list):
            metrics = MetricManager(configs=metrics, num_classes=num_classes)
        self._metric_manager = metrics

        # Register metrics as ModuleDicts for proper device handling
        self.train_metrics = metrics.get_train_metrics()
        self.val_metrics = metrics.get_val_metrics()
        self.test_metrics = metrics.get_test_metrics()

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
        self._scheduler = scheduler
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._mixup_alpha = mixup_alpha
        self._num_classes = num_classes

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
        for name, metric in self.train_metrics.items():
            config = self._metric_manager.get_metric_config("train", name)
            if isinstance(metric, TimmMetricWrapper):
                metric.update(logits, y)  # timm uses logits
            else:
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
        for name, metric in self.val_metrics.items():
            config = self._metric_manager.get_metric_config("val", name)
            if isinstance(metric, TimmMetricWrapper):
                metric.update(logits, y)
            else:
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

        # Log as image if tensorboard supports it
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(10, 10))
            cm_np = cm.cpu().numpy()
            im = ax.imshow(cm_np, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix (Epoch {self.current_epoch})")
            fig.colorbar(im, ax=ax)

            # Try to log to tensorboard
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
        for name, metric in self.test_metrics.items():
            config = self._metric_manager.get_metric_config("test", name)
            if isinstance(metric, TimmMetricWrapper):
                metric.update(logits, y)
            else:
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

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        if self._scheduler == "none":
            return {"optimizer": optimizer}

        if self._scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                **self._scheduler_kwargs,
            )
        elif self._scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1, **self._scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self._scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
