"""Image classification task as a PyTorch Lightning module."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from autotimm.backbone import BackboneConfig, create_backbone, get_backbone_out_features
from autotimm.heads import ClassificationHead


class ImageClassifier(pl.LightningModule):
    """End-to-end image classifier backed by a timm backbone.

    Parameters:
        backbone: A timm model name (str) or a :class:`BackboneConfig`.
        num_classes: Number of target classes.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        scheduler: One of ``"cosine"``, ``"step"``, ``"none"``.
        scheduler_kwargs: Extra kwargs forwarded to the LR scheduler.
        head_dropout: Dropout before the classification linear layer.
        label_smoothing: Label smoothing factor for cross-entropy.
        freeze_backbone: If ``True``, backbone parameters are frozen
            (useful for linear probing).
        mixup_alpha: If > 0, apply Mixup augmentation with this alpha.
    """

    def __init__(
        self,
        backbone: str | BackboneConfig = "resnet50",
        num_classes: int = 10,
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
        self.save_hyperparameters()

        self.backbone = create_backbone(backbone)
        in_features = get_backbone_out_features(self.backbone)
        self.head = ClassificationHead(in_features, num_classes, dropout=head_dropout)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self._lr = lr
        self._weight_decay = weight_decay
        self._scheduler = scheduler
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._mixup_alpha = mixup_alpha

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
        self.train_acc(preds, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log(
            "val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)

        self.test_acc(preds, y)
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

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
