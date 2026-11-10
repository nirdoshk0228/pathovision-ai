"""Segmentation losses.

Tumour pixels are usually a small fraction of a patch set, so plain BCE lets the
model score well by predicting "normal" everywhere. Soft Dice optimises overlap
directly and is insensitive to class imbalance; combining it with BCE keeps
gradients well-behaved early in training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.dice = SoftDiceLoss()

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        return self.bce_weight * bce + (1 - self.bce_weight) * self.dice(logits, target)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) for patch classification with hard negatives."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        a_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()
