from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Baseline MLP:
        d -> 128 -> num_classes

    No dropout, no special regularization (handled by optimizer weight_decay if desired).
    """

    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegularized(nn.Module):
    """
    Regularized MLP:
        d -> 256 -> 128 -> num_classes

    Includes dropout to reduce overfitting.
    Weight decay should be applied in the optimizer (Adam weight_decay).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim1: int = 256,
        hidden_dim2: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPWide(nn.Module):
    """
    Capacity-sweep MLP (wide, shallow):
        d -> 512 -> num_classes

    This isolates the effect of increasing model capacity without changing depth.
    """

    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)