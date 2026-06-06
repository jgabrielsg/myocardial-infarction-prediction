"""Shared PyTorch multi-task architecture."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MultiTaskMI(nn.Module):
    """Shared-backbone multi-task network for binary + multiclass targets."""

    def __init__(
        self,
        input_size: int,
        num_binary_targets: int,
        num_multiclass_classes: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.shared_fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.shared_fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.binary_head = nn.Linear(64, num_binary_targets)
        self.multiclass_head = nn.Linear(64, num_multiclass_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout1(self.relu(self.bn1(self.shared_fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.shared_fc2(x))))
        return self.binary_head(x), self.multiclass_head(x)
