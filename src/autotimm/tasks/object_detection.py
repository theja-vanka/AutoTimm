"""Object detection task as a PyTorch Lightning module."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from autotimm.backbone import (
    FeatureBackboneConfig,
    create_feature_backbone,
    get_feature_channels,
)
from autotimm.data.transform_config import TransformConfig
from autotimm.heads import DetectionHead, FPN, YOLOXHead
from autotimm.losses import FocalLoss, GIoULoss
from autotimm.metrics import LoggingConfig, MetricConfig, MetricManager
from autotimm.tasks.preprocessing_mixin import PreprocessingMixin


class ObjectDetector(PreprocessingMixin, pl.LightningModule):
    """End-to-end object detector supporting FCOS and YOLOX architectures.

    Architecture: timm backbone → FPN → Detection Head (FCOS/YOLOX) → NMS

    Parameters:
        backbone: A timm model name (str) or a :class:`FeatureBackboneConfig`.
        num_classes: Number of object classes (excluding background).
        detection_arch: Detection architecture to use. Options: ``"fcos"`` or ``"yolox"``.
            Default is ``"fcos"``. YOLOX uses a decoupled head and no centerness prediction.
        metrics: A :class:`MetricManager` instance or list of :class:`MetricConfig`
            objects. Optional - if not provided, uses MeanAveragePrecision.
        logging_config: Optional :class:`LoggingConfig` for enhanced logging.
        transform_config: Optional :class:`TransformConfig` for unified transform
            configuration. When provided, enables the ``preprocess()`` method
            for inference-time preprocessing using model-specific normalization.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        optimizer: Optimizer name (``"adamw"``, ``"adam"``, ``"sgd"``, etc.) or dict
            with ``"class"`` and ``"params"`` keys.
        optimizer_kwargs: Additional kwargs for the optimizer.
        scheduler: Scheduler name (``"cosine"``, ``"step"``, ``"onecycle"``, etc.),
            dict config, or ``None`` for no scheduler.
        scheduler_kwargs: Extra kwargs forwarded to the LR scheduler.
        fpn_channels: Number of channels in FPN layers.
        head_num_convs: Number of conv layers in detection head branches.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.
        cls_loss_weight: Weight for classification loss.
        reg_loss_weight: Weight for regression loss.
        centerness_loss_weight: Weight for centerness loss (FCOS only).
        score_thresh: Score threshold for detections during inference.
        nms_thresh: IoU threshold for NMS.
        max_detections_per_image: Maximum detections to keep per image.
        freeze_backbone: If ``True``, backbone parameters are frozen.
        strides: FPN output strides. Default (8, 16, 32, 64, 128) for P3-P7.
        regress_ranges: Regression ranges for each FPN level (FCOS only).

    Example:
        >>> model = ObjectDetector(
        ...     backbone="resnet50",
        ...     num_classes=80,
        ...     metrics=[
        ...         MetricConfig(
        ...             name="mAP",
        ...             backend="torchmetrics",
        ...             metric_class="MeanAveragePrecision",
        ...             params={},
        ...             stages=["val", "test"],
        ...         ),
        ...     ],
        ...     lr=1e-4,
        ... )
    """

    def __init__(
        self,
        backbone: str | FeatureBackboneConfig,
        num_classes: int,
        detection_arch: str = "fcos",
        metrics: MetricManager | list[MetricConfig] | None = None,
        logging_config: LoggingConfig | None = None,
        transform_config: TransformConfig | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str | dict[str, Any] = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: str | dict[str, Any] | None = "cosine",
        scheduler_kwargs: dict[str, Any] | None = None,
        fpn_channels: int = 256,
        head_num_convs: int = 4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 1.0,
        centerness_loss_weight: float = 1.0,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        max_detections_per_image: int = 100,
        freeze_backbone: bool = False,
        strides: tuple[int, ...] = (8, 16, 32, 64, 128),
        regress_ranges: tuple[tuple[int, int], ...] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["metrics", "logging_config", "transform_config"]
        )

        # Validate detection architecture
        if detection_arch not in ["fcos", "yolox"]:
            raise ValueError(
                f"detection_arch must be 'fcos' or 'yolox', got '{detection_arch}'"
            )

        self.detection_arch = detection_arch
        self.num_classes = num_classes
        self._lr = lr
        self._weight_decay = weight_decay
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler = scheduler
        self._scheduler_kwargs = scheduler_kwargs or {}

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_detections_per_image = max_detections_per_image
        self.strides = strides

        # Default regression ranges for FCOS (P3-P7)
        # Note: YOLOX doesn't use regression ranges (handles all scales dynamically)
        if regress_ranges is None:
            self.regress_ranges = (
                (-1, 64),
                (64, 128),
                (128, 256),
                (256, 512),
                (512, float("inf")),
            )
        else:
            self.regress_ranges = regress_ranges

        # Build model
        self.backbone = create_feature_backbone(backbone)
        in_channels = get_feature_channels(self.backbone)

        self.fpn = FPN(
            in_channels_list=in_channels,
            out_channels=fpn_channels,
            num_extra_levels=2,  # P6, P7 from P5
        )

        # Create detection head based on architecture
        if detection_arch == "fcos":
            self.head = DetectionHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                num_convs=head_num_convs,
                prior_prob=0.01,
            )
        elif detection_arch == "yolox":
            self.head = YOLOXHead(
                in_channels=fpn_channels,
                num_classes=num_classes,
                num_convs=head_num_convs
                if head_num_convs <= 2
                else 2,  # YOLOX typically uses 2 convs
                prior_prob=0.01,
                activation="silu",
            )

        # Losses
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.centerness_loss_weight = centerness_loss_weight

        self.focal_loss = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, reduction="sum"
        )
        self.giou_loss = GIoULoss(reduction="sum")

        # Initialize metrics
        if metrics is None:
            # Default: use mAP for validation and test
            metrics = [
                MetricConfig(
                    name="mAP",
                    backend="torchmetrics",
                    metric_class="MeanAveragePrecision",
                    params={"box_format": "xyxy", "iou_type": "bbox"},
                    stages=["val", "test"],
                    prog_bar=True,
                ),
            ]

        if isinstance(metrics, list):
            # For detection metrics, num_classes isn't typically passed to constructor
            # We pass it but MetricManager will handle it appropriately
            self._metric_configs = metrics
            self._use_metric_manager = False
        else:
            self._metric_manager = metrics
            self._use_metric_manager = True

        # Register metrics as ModuleDicts
        self._register_detection_metrics()

        # Logging configuration
        self._logging_config = logging_config or LoggingConfig(
            log_learning_rate=False,
            log_gradient_norm=False,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Setup transforms from config (PreprocessingMixin)
        self._setup_transforms(transform_config, task="detection")

    def _register_detection_metrics(self):
        """Register detection-specific metrics."""
        import torchmetrics.detection

        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()

        if self._use_metric_manager:
            # Use the manager's metrics
            self.val_metrics = self._metric_manager.get_val_metrics()
            self.test_metrics = self._metric_manager.get_test_metrics()
        else:
            # Create metrics from configs
            for config in self._metric_configs:
                if config.backend == "torchmetrics":
                    if hasattr(torchmetrics.detection, config.metric_class):
                        metric_cls = getattr(
                            torchmetrics.detection, config.metric_class
                        )
                    elif hasattr(torchmetrics, config.metric_class):
                        metric_cls = getattr(torchmetrics, config.metric_class)
                    else:
                        raise ValueError(f"Unknown metric: {config.metric_class}")

                    if "val" in config.stages:
                        self.val_metrics[config.name] = metric_cls(**config.params)
                    if "test" in config.stages:
                        self.test_metrics[config.name] = metric_cls(**config.params)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None]:
        """Forward pass through the detector.

        Args:
            images: Input images [B, C, H, W].

        Returns:
            Tuple of (cls_outputs, reg_outputs, centerness_outputs) per FPN level.
            For YOLOX, centerness_outputs is None.
        """
        features = self.backbone(images)
        fpn_features = self.fpn(features)

        if self.detection_arch == "fcos":
            cls_outputs, reg_outputs, centerness_outputs = self.head(fpn_features)
        elif self.detection_arch == "yolox":
            cls_outputs, reg_outputs = self.head(fpn_features)
            centerness_outputs = None
        else:
            raise ValueError(f"Unknown detection_arch: {self.detection_arch}")

        return cls_outputs, reg_outputs, centerness_outputs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        target_boxes = batch["boxes"]  # List of [N_i, 4] tensors
        target_labels = batch["labels"]  # List of [N_i] tensors

        # Forward
        cls_outputs, reg_outputs, centerness_outputs = self(images)

        # Compute targets for each FPN level
        device = images.device
        batch_size = images.shape[0]
        img_h, img_w = images.shape[-2:]

        # Compute loss
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_centerness_loss = torch.tensor(0.0, device=device)
        num_pos = 0

        # Prepare centerness outputs for iteration
        if centerness_outputs is None:
            # YOLOX doesn't use centerness
            centerness_iter = [None] * len(cls_outputs)
        else:
            centerness_iter = centerness_outputs

        for level_idx, (cls_out, reg_out, cent_out) in enumerate(
            zip(cls_outputs, reg_outputs, centerness_iter)
        ):
            stride = self.strides[level_idx]
            feat_h, feat_w = cls_out.shape[-2:]

            # Generate grid points for this level
            grid_y, grid_x = torch.meshgrid(
                torch.arange(feat_h, device=device, dtype=torch.float32),
                torch.arange(feat_w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            # Points are at the center of each cell
            points_x = (grid_x + 0.5) * stride
            points_y = (grid_y + 0.5) * stride
            points = torch.stack([points_x, points_y], dim=-1)  # [H, W, 2]

            # Compute targets for this level
            level_cls_targets = []
            level_reg_targets = []
            level_centerness_targets = [] if self.detection_arch == "fcos" else None

            for b in range(batch_size):
                boxes = target_boxes[b]  # [N, 4] in xyxy format
                labels = target_labels[b]  # [N]

                if len(boxes) == 0:
                    # No objects - all background
                    cls_target = torch.full(
                        (feat_h, feat_w), -1, dtype=torch.long, device=device
                    )
                    reg_target = torch.zeros(feat_h, feat_w, 4, device=device)
                    cent_target = (
                        torch.zeros(feat_h, feat_w, device=device)
                        if self.detection_arch == "fcos"
                        else None
                    )
                else:
                    cls_target, reg_target, cent_target = (
                        self._compute_targets_per_level(
                            points, boxes, labels, stride, level_idx, (img_h, img_w)
                        )
                    )

                level_cls_targets.append(cls_target)
                level_reg_targets.append(reg_target)
                if self.detection_arch == "fcos":
                    level_centerness_targets.append(cent_target)

            # Stack batch
            cls_targets = torch.stack(level_cls_targets)  # [B, H, W]
            reg_targets = torch.stack(level_reg_targets)  # [B, H, W, 4]
            cent_targets = (
                torch.stack(level_centerness_targets)
                if self.detection_arch == "fcos"
                else None
            )  # [B, H, W]

            # Compute classification loss (all locations except ignored=-1)
            cls_out_flat = cls_out.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            cls_targets_flat = cls_targets.reshape(-1)
            valid_mask = cls_targets_flat >= 0

            if valid_mask.any():
                # For focal loss, background class is handled implicitly
                # Positive samples have class labels, negatives have -1 (ignored in loss)
                total_cls_loss = total_cls_loss + self.focal_loss(
                    cls_out_flat, cls_targets_flat
                )

            # Compute regression and centerness loss (positive samples only)
            pos_mask = cls_targets >= 0  # [B, H, W]

            if pos_mask.any():
                pos_reg_pred = reg_out.permute(0, 2, 3, 1)[pos_mask]  # [N_pos, 4]
                pos_reg_target = reg_targets[pos_mask]  # [N_pos, 4]

                # IoU-based regression loss
                reg_loss = self._compute_iou_loss(pos_reg_pred, pos_reg_target)
                total_reg_loss = total_reg_loss + reg_loss

                # Centerness BCE loss (FCOS only)
                if self.detection_arch == "fcos" and cent_out is not None:
                    pos_cent_pred = cent_out.squeeze(1)[pos_mask]  # [N_pos]
                    pos_cent_target = cent_targets[pos_mask]  # [N_pos]
                    cent_loss = F.binary_cross_entropy_with_logits(
                        pos_cent_pred, pos_cent_target, reduction="sum"
                    )
                    total_centerness_loss = total_centerness_loss + cent_loss

                num_pos += pos_mask.sum().item()

        # Normalize by number of positive samples
        num_pos = max(num_pos, 1)

        cls_loss = self.cls_loss_weight * total_cls_loss / num_pos
        reg_loss = self.reg_loss_weight * total_reg_loss / num_pos

        # Centerness loss only for FCOS
        if self.detection_arch == "fcos":
            centerness_loss = (
                self.centerness_loss_weight * total_centerness_loss / num_pos
            )
            total_loss = cls_loss + reg_loss + centerness_loss
        else:
            centerness_loss = torch.tensor(0.0, device=device)
            total_loss = cls_loss + reg_loss

        # Log losses
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/cls_loss", cls_loss)
        self.log("train/reg_loss", reg_loss)
        if self.detection_arch == "fcos":
            self.log("train/centerness_loss", centerness_loss)
        self.log("train/num_pos", float(num_pos))

        # Enhanced logging
        if self._logging_config.log_learning_rate:
            opt = self.optimizers()
            if opt is not None and hasattr(opt, "param_groups"):
                lr = opt.param_groups[0]["lr"]
                self.log("train/lr", lr, on_step=True, on_epoch=False)

        return total_loss

    def _compute_targets_per_level(
        self,
        points: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        stride: int,
        level_idx: int,
        img_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute FCOS targets for a single image at one FPN level.

        Args:
            points: Grid points [H, W, 2].
            boxes: Target boxes [N, 4] in xyxy format.
            labels: Target labels [N].
            stride: Stride for this FPN level.
            level_idx: Index of FPN level.
            img_size: (H, W) of input image.

        Returns:
            cls_target: [H, W] with class labels or -1 for ignore.
            reg_target: [H, W, 4] with (l, t, r, b) distances.
            centerness_target: [H, W] with centerness values.
        """
        device = points.device
        feat_h, feat_w = points.shape[:2]

        # Expand points and boxes for broadcasting
        points_flat = points.reshape(-1, 2)  # [H*W, 2]
        num_points = points_flat.shape[0]

        # Compute distances from each point to each box
        # boxes: [N, 4] -> [1, N, 4] for broadcasting
        boxes_exp = boxes.unsqueeze(0)  # [1, N, 4]
        points_exp = points_flat.unsqueeze(1)  # [H*W, 1, 2]

        # Left, top, right, bottom distances
        left = points_exp[..., 0] - boxes_exp[..., 0]  # [H*W, N]
        top = points_exp[..., 1] - boxes_exp[..., 1]
        right = boxes_exp[..., 2] - points_exp[..., 0]
        bottom = boxes_exp[..., 3] - points_exp[..., 1]

        reg_targets_per_box = torch.stack(
            [left, top, right, bottom], dim=-1
        )  # [H*W, N, 4]

        # Check if point is inside box
        inside_box = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)  # [H*W, N]

        # Check regression range constraint
        max_reg = reg_targets_per_box.max(dim=-1)[0]  # [H*W, N]
        min_range, max_range = self.regress_ranges[level_idx]
        in_range = (max_reg >= min_range) & (max_reg < max_range)

        # Valid assignments: inside box AND in regression range
        valid = inside_box & in_range  # [H*W, N]

        # For each point, find the box with minimum area (most specific)
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [N]
        box_areas_exp = box_areas.unsqueeze(0).expand(num_points, -1)  # [H*W, N]

        # Set invalid assignments to inf area
        box_areas_masked = torch.where(
            valid, box_areas_exp, torch.tensor(float("inf"), device=device)
        )

        # Find best box for each point
        min_areas, best_box_idx = box_areas_masked.min(dim=1)  # [H*W]
        has_assignment = min_areas < float("inf")

        # Create targets
        cls_target = torch.full((num_points,), -1, dtype=torch.long, device=device)
        reg_target = torch.zeros(num_points, 4, device=device)
        centerness_target = torch.zeros(num_points, device=device)

        if has_assignment.any():
            assigned_idx = best_box_idx[has_assignment]
            cls_target[has_assignment] = labels[assigned_idx]

            # Get regression targets for assigned points
            point_indices = torch.arange(num_points, device=device)[has_assignment]
            reg_target[has_assignment] = reg_targets_per_box[
                point_indices, assigned_idx
            ]

            # Compute centerness
            lr = reg_target[has_assignment]
            left_right_min = torch.min(lr[:, 0], lr[:, 2])
            left_right_max = torch.max(lr[:, 0], lr[:, 2])
            top_bottom_min = torch.min(lr[:, 1], lr[:, 3])
            top_bottom_max = torch.max(lr[:, 1], lr[:, 3])

            centerness = torch.sqrt(
                (left_right_min / left_right_max.clamp(min=1e-7))
                * (top_bottom_min / top_bottom_max.clamp(min=1e-7))
            )
            centerness_target[has_assignment] = centerness

        # Reshape to spatial dimensions
        cls_target = cls_target.reshape(feat_h, feat_w)
        reg_target = reg_target.reshape(feat_h, feat_w, 4)
        centerness_target = centerness_target.reshape(feat_h, feat_w)

        return cls_target, reg_target, centerness_target

    def _compute_iou_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU-based regression loss for LTRB predictions."""
        # Compute areas from LTRB distances
        pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
        target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])

        # Intersection
        inter_w = torch.min(pred[:, 0], target[:, 0]) + torch.min(
            pred[:, 2], target[:, 2]
        )
        inter_h = torch.min(pred[:, 1], target[:, 1]) + torch.min(
            pred[:, 3], target[:, 3]
        )
        inter_area = inter_w * inter_h

        # Union
        union_area = pred_area + target_area - inter_area

        # IoU loss (negative log)
        iou = inter_area / union_area.clamp(min=1e-7)
        loss = -torch.log(iou.clamp(min=1e-7))

        return loss.sum()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        images = batch["images"]
        target_boxes = batch["boxes"]
        target_labels = batch["labels"]

        # Get predictions
        detections = self.predict(images)

        # Convert to format expected by torchmetrics.detection.MeanAveragePrecision
        preds = []
        targets = []

        for i in range(len(detections)):
            preds.append(
                {
                    "boxes": detections[i]["boxes"],
                    "scores": detections[i]["scores"],
                    "labels": detections[i]["labels"],
                }
            )
            targets.append(
                {
                    "boxes": target_boxes[i],
                    "labels": target_labels[i],
                }
            )

        # Update metrics
        for name, metric in self.val_metrics.items():
            metric.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        for name, metric in self.val_metrics.items():
            result = metric.compute()
            # MeanAveragePrecision returns a dict
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        self.log(f"val/{key}", value, prog_bar=(key == "map"))
            else:
                self.log(f"val/{name}", result)
            metric.reset()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        images = batch["images"]
        target_boxes = batch["boxes"]
        target_labels = batch["labels"]

        detections = self.predict(images)

        preds = []
        targets = []

        for i in range(len(detections)):
            preds.append(
                {
                    "boxes": detections[i]["boxes"],
                    "scores": detections[i]["scores"],
                    "labels": detections[i]["labels"],
                }
            )
            targets.append(
                {
                    "boxes": target_boxes[i],
                    "labels": target_labels[i],
                }
            )

        for name, metric in self.test_metrics.items():
            metric.update(preds, targets)

    def on_test_epoch_end(self) -> None:
        for name, metric in self.test_metrics.items():
            result = metric.compute()
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        self.log(f"test/{key}", value)
            else:
                self.log(f"test/{name}", result)
            metric.reset()

    def predict(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Run inference and return detections with NMS.

        Args:
            images: Input images [B, C, H, W].

        Returns:
            List of dicts per image with 'boxes', 'scores', 'labels'.
        """
        cls_outputs, reg_outputs, centerness_outputs = self(images)

        batch_size = images.shape[0]
        img_h, img_w = images.shape[-2:]
        device = images.device

        all_detections = []

        # Prepare centerness outputs for iteration
        if centerness_outputs is None:
            centerness_iter = [None] * len(cls_outputs)
        else:
            centerness_iter = centerness_outputs

        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            for level_idx, (cls_out, reg_out, cent_out) in enumerate(
                zip(cls_outputs, reg_outputs, centerness_iter)
            ):
                stride = self.strides[level_idx]
                feat_h, feat_w = cls_out.shape[-2:]

                # Get predictions for this image
                cls_logits = cls_out[b]  # [C, H, W]
                reg_pred = reg_out[b]  # [4, H, W]

                # Generate grid points
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(feat_h, device=device, dtype=torch.float32),
                    torch.arange(feat_w, device=device, dtype=torch.float32),
                    indexing="ij",
                )
                points_x = (grid_x + 0.5) * stride
                points_y = (grid_y + 0.5) * stride

                # Flatten spatial dimensions
                cls_logits = cls_logits.permute(1, 2, 0).reshape(
                    -1, self.num_classes
                )  # [H*W, C]
                reg_pred = reg_pred.permute(1, 2, 0).reshape(-1, 4)  # [H*W, 4]
                points_x = points_x.reshape(-1)
                points_y = points_y.reshape(-1)

                # Compute scores
                cls_scores = cls_logits.sigmoid()
                if cent_out is not None:
                    # FCOS: classification * centerness
                    cent_pred = cent_out[b, 0]  # [H, W]
                    cent_pred = cent_pred.reshape(-1)  # [H*W]
                    centerness = cent_pred.sigmoid()
                    scores = cls_scores * centerness.unsqueeze(-1)  # [H*W, C]
                else:
                    # YOLOX: just use classification scores
                    scores = cls_scores  # [H*W, C]

                # Get max score per location
                max_scores, class_ids = scores.max(dim=-1)  # [H*W]

                # Filter by score threshold
                keep = max_scores > self.score_thresh
                if not keep.any():
                    continue

                max_scores = max_scores[keep]
                class_ids = class_ids[keep]
                reg_pred = reg_pred[keep]
                points_x = points_x[keep]
                points_y = points_y[keep]

                # Convert LTRB to xyxy boxes
                left = reg_pred[:, 0]
                top = reg_pred[:, 1]
                right = reg_pred[:, 2]
                bottom = reg_pred[:, 3]

                x1 = points_x - left
                y1 = points_y - top
                x2 = points_x + right
                y2 = points_y + bottom

                # Clamp to image bounds
                x1 = x1.clamp(min=0, max=img_w)
                y1 = y1.clamp(min=0, max=img_h)
                x2 = x2.clamp(min=0, max=img_w)
                y2 = y2.clamp(min=0, max=img_h)

                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                all_boxes.append(boxes)
                all_scores.append(max_scores)
                all_labels.append(class_ids)

            # Concatenate all levels
            if len(all_boxes) > 0:
                boxes = torch.cat(all_boxes, dim=0)
                scores = torch.cat(all_scores, dim=0)
                labels = torch.cat(all_labels, dim=0)

                # Apply NMS per class
                keep_indices = ops.batched_nms(boxes, scores, labels, self.nms_thresh)

                # Limit number of detections
                keep_indices = keep_indices[: self.max_detections_per_image]

                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]
            else:
                boxes = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)

            all_detections.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
            )

        return all_detections

    def predict_step(self, batch: Any, batch_idx: int) -> list[dict[str, torch.Tensor]]:
        images = batch["images"] if isinstance(batch, dict) else batch
        return self.predict(images)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler."""
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self._create_optimizer(params)

        if self._scheduler is None or self._scheduler == "none":
            return {"optimizer": optimizer}

        scheduler_config = self._create_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _create_optimizer(self, params) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_kwargs = {"lr": self._lr, "weight_decay": self._weight_decay}
        opt_kwargs.update(self._optimizer_kwargs)

        if isinstance(self._optimizer, dict):
            opt_class_path = self._optimizer["class"]
            opt_params = self._optimizer.get("params", {})
            opt_kwargs.update(opt_params)
            optimizer_cls = self._import_class(opt_class_path)
            return optimizer_cls(params, **opt_kwargs)

        optimizer_name = self._optimizer.lower()

        torch_optimizers = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }

        if optimizer_name in torch_optimizers:
            return torch_optimizers[optimizer_name](params, **opt_kwargs)

        try:
            import timm.optim as timm_optim

            timm_optimizers = {
                "adamp": timm_optim.AdamP,
                "sgdp": timm_optim.SGDP,
                "lamb": timm_optim.Lamb,
            }
            if optimizer_name in timm_optimizers:
                return timm_optimizers[optimizer_name](params, **opt_kwargs)
        except ImportError:
            pass

        raise ValueError(f"Unknown optimizer: {self._optimizer}")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        """Create scheduler config."""
        sched_kwargs = self._scheduler_kwargs.copy()
        interval = sched_kwargs.pop("interval", "step")
        frequency = sched_kwargs.pop("frequency", 1)

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

        scheduler_name = self._scheduler.lower()

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
                step_size=sched_kwargs.pop("step_size", 8),
                gamma=sched_kwargs.pop("gamma", 0.1),
                **sched_kwargs,
            )
        elif scheduler_name == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_kwargs.pop("milestones", [8, 11]),
                gamma=sched_kwargs.pop("gamma", 0.1),
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
            raise ValueError(f"Unknown scheduler: {self._scheduler}")

        return {"scheduler": scheduler, "interval": interval, "frequency": frequency}

    def _import_class(self, class_path: str):
        """Import a class from a fully qualified path."""
        import importlib

        if "." not in class_path:
            raise ValueError(f"Class path must be fully qualified, got: {class_path}")

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def on_before_optimizer_step(self, optimizer) -> None:
        """Hook for gradient norm logging."""
        if self._logging_config.log_gradient_norm:
            grad_norm = self._compute_gradient_norm()
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

        if self._logging_config.log_weight_norm:
            weight_norm = self._compute_weight_norm()
            self.log("train/weight_norm", weight_norm, on_step=True, on_epoch=False)

    def _compute_gradient_norm(self) -> torch.Tensor:
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return torch.tensor(total_norm**0.5, device=self.device)

    def _compute_weight_norm(self) -> torch.Tensor:
        total_norm = 0.0
        for p in self.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return torch.tensor(total_norm**0.5, device=self.device)
