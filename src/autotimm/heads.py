"""Task-specific heads."""

from __future__ import annotations

import torch
import torch.nn as nn


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
