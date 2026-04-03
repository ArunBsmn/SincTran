"""Loss functions for EEG classification."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CELoss(nn.Module):
    """
    Cross-entropy loss wrapper that returns predictions.

    Args:
        label_smoothing (float): Label smoothing factor in [0, 1). Default: 0.0.
        class_weights   (Tensor, optional): Per-class loss weights, shape (n_classes,).

    Raises:
        ValueError: If ``label_smoothing`` is outside [0, 1).
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights:   Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1).")

        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)

    def forward(self, logits: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            logits (Tensor): Shape (B, n_classes).
            labels (Tensor): Shape (B,), integer class indices.

        Returns:
            Tuple[Tensor, Tensor]: Scalar loss and (B,) predicted class indices.
        """
        ce = F.cross_entropy(
            logits,
            labels,
            weight          = getattr(self, "class_weights", None),
            label_smoothing = self.label_smoothing,
            reduction       = "none",
        )
        loss = ce.mean()

        with torch.no_grad():
            preds = logits.argmax(dim=1)

        return loss, preds