"""Task-specific heads."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Linear classification head with optional dropout.

    Parameters:
        in_features: Dimensionality of the backbone output.
        num_classes: Number of target classes.
        dropout: Dropout probability before the final linear layer.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale object detection.

    Takes feature maps from a backbone at multiple scales (C2-C5) and produces
    a feature pyramid (P3-P7) with top-down pathway and lateral connections.

    Parameters:
        in_channels_list: List of input channel counts from backbone features.
            Typically [256, 512, 1024, 2048] for ResNet-50.
        out_channels: Number of output channels for all pyramid levels.
        num_extra_levels: Number of extra levels to add beyond the backbone
            features. Default 2 adds P6 and P7 from P5.
        use_depthwise: Whether to use depthwise separable convolutions.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        out_channels: int = 256,
        num_extra_levels: int = 2,
        use_depthwise: bool = False,
    ):
        super().__init__()
        self.in_channels_list = list(in_channels_list)
        self.out_channels = out_channels
        self.num_extra_levels = num_extra_levels

        # Lateral (1x1) connections
        self.lateral_convs = nn.ModuleList()
        for in_channels in self.in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        # Top-down pathway (3x3) convolutions
        self.output_convs = nn.ModuleList()
        conv_fn = _depthwise_conv if use_depthwise else _standard_conv
        for _ in range(len(self.in_channels_list)):
            self.output_convs.append(conv_fn(out_channels, out_channels))

        # Extra levels (P6, P7) from P5
        self.extra_convs = nn.ModuleList()
        for i in range(num_extra_levels):
            in_ch = out_channels if i == 0 else out_channels
            self.extra_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=2, padding=1)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through FPN.

        Args:
            features: List of feature maps from backbone [C2, C3, C4, C5] or similar.

        Returns:
            List of pyramid features [P3, P4, P5, P6, P7] with uniform channels.
        """
        assert len(features) == len(self.in_channels_list), (
            f"Expected {len(self.in_channels_list)} features, got {len(features)}"
        )

        # Build top-down pathway
        laterals = [
            lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)
        ]

        # Top-down fusion (from highest level to lowest)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        # Apply 3x3 convs to remove aliasing
        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

        # Add extra levels from the last output
        last_feat = outputs[-1]
        for extra_conv in self.extra_convs:
            last_feat = F.relu(extra_conv(last_feat))
            outputs.append(last_feat)

        return outputs


def _standard_conv(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


def _depthwise_conv(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        ),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
    )


class ScaleModule(nn.Module):
    """Learnable scalar for FCOS centerness/regression scaling."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DetectionHead(nn.Module):
    """FCOS-style detection head with classification, regression, and centerness branches.

    This is an anchor-free detection head that predicts:
    - Class scores for each spatial location
    - Bounding box regression (distances to left, top, right, bottom)
    - Centerness scores to downweight predictions far from object centers

    Parameters:
        in_channels: Number of input channels from FPN.
        num_classes: Number of object classes (excluding background).
        num_convs: Number of conv layers in each branch before prediction.
        prior_prob: Prior probability for focal loss initialization.
        use_group_norm: Whether to use GroupNorm after conv layers.
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_convs: int = 4,
        prior_prob: float = 0.01,
        use_group_norm: bool = True,
        num_groups: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Shared conv layers for classification branch
        cls_convs = []
        for _ in range(num_convs):
            cls_convs.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            if use_group_norm:
                cls_convs.append(nn.GroupNorm(num_groups, in_channels))
            cls_convs.append(nn.ReLU(inplace=True))
        self.cls_convs = nn.Sequential(*cls_convs)

        # Shared conv layers for regression branch
        reg_convs = []
        for _ in range(num_convs):
            reg_convs.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            if use_group_norm:
                reg_convs.append(nn.GroupNorm(num_groups, in_channels))
            reg_convs.append(nn.ReLU(inplace=True))
        self.reg_convs = nn.Sequential(*reg_convs)

        # Prediction layers
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # Per-level learnable scales for regression
        self.scales = nn.ModuleList([ScaleModule(1.0) for _ in range(5)])

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize classification bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through detection head.

        Args:
            features: List of FPN features [P3, P4, P5, P6, P7].

        Returns:
            Tuple of (cls_outputs, reg_outputs, centerness_outputs), each a list
            of tensors with shapes:
            - cls: [B, num_classes, H, W]
            - reg: [B, 4, H, W] (left, top, right, bottom distances)
            - centerness: [B, 1, H, W]
        """
        cls_outputs = []
        reg_outputs = []
        centerness_outputs = []

        for i, feat in enumerate(features):
            cls_feat = self.cls_convs(feat)
            reg_feat = self.reg_convs(feat)

            # Classification prediction
            cls_out = self.cls_pred(cls_feat)

            # Regression prediction (with per-level scaling)
            scale_idx = min(i, len(self.scales) - 1)
            reg_out = self.scales[scale_idx](self.reg_pred(reg_feat))
            reg_out = F.relu(reg_out)  # Distances must be positive

            # Centerness prediction (from regression branch)
            centerness_out = self.centerness_pred(reg_feat)

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            centerness_outputs.append(centerness_out)

        return cls_outputs, reg_outputs, centerness_outputs

    def forward_single(
        self, feat: torch.Tensor, scale_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single feature level.

        Useful for inference or when processing levels independently.
        """
        cls_feat = self.cls_convs(feat)
        reg_feat = self.reg_convs(feat)

        cls_out = self.cls_pred(cls_feat)
        scale_idx = min(scale_idx, len(self.scales) - 1)
        reg_out = F.relu(self.scales[scale_idx](self.reg_pred(reg_feat)))
        centerness_out = self.centerness_pred(reg_feat)

        return cls_out, reg_out, centerness_out
