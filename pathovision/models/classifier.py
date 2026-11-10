"""Patch-level tumour classifier (CNN backbone + single-logit head).

ImageNet-pretrained backbones transfer well to histology despite the domain gap:
low-level filters (edges, blobs, texture) are generic, and pretraining cuts the
data and epochs needed. Output is one logit trained with BCE: P(patch contains
metastatic tissue in its centre region).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

ARCHS = ("resnet18", "resnet34", "resnet50", "efficientnet_b0", "densenet121")


def build_classifier(arch: str = "resnet18", pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}")
    weights = "DEFAULT" if pretrained else None
    if arch.startswith("resnet"):
        net = getattr(tvm, arch)(weights=weights)
        in_f = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
    elif arch == "efficientnet_b0":
        net = tvm.efficientnet_b0(weights=weights)
        in_f = net.classifier[1].in_features
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
    else:  # densenet121
        net = tvm.densenet121(weights=weights)
        in_f = net.classifier.in_features
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
    return net


class PatchClassifier(nn.Module):
    """Wraps a backbone so that ``forward`` returns logits of shape (B,)."""

    def __init__(self, arch: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.arch = arch
        self.net = build_classifier(arch, pretrained, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
