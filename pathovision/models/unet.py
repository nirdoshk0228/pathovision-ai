"""U-Net for pixel-level tumour segmentation.

Two variants:

* ``UNet``: the classic encoder-decoder (Ronneberger et al., 2015) with skip
  connections. Encoder halves resolution and doubles channels at each stage; the
  decoder upsamples and concatenates the matching encoder feature map so fine
  spatial detail (lesion borders) lost by pooling is recovered.
* ``ResNetUNet``: same decoder, but the encoder is an ImageNet-pretrained
  ResNet-18/34. Converges faster and needs fewer annotated pixels — the usual
  choice when labelled pathology data is scarce.

Both return raw logits of shape (B, 1, H, W).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = ConvBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.downs.append(ConvBlock(c, ch))
            c = ch
        self.bottleneck = ConvBlock(chs[-2], chs[-1], dropout)
        self.ups = nn.ModuleList()
        c = chs[-1]
        for ch in reversed(chs[:-1]):
            self.ups.append(UpBlock(c, ch, ch))
            c = ch
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips: List[torch.Tensor] = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, s in zip(self.ups, reversed(skips)):
            x = up(x, s)
        return self.head(x)


class ResNetUNet(nn.Module):
    """U-Net decoder on a torchvision ResNet encoder (feature strides 2, 4, 8, 16, 32)."""

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        r = getattr(tvm, encoder)(weights=weights)
        enc_ch = [64, 64, 128, 256, 512] if encoder in ("resnet18", "resnet34") else [64, 256, 512, 1024, 2048]
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # /2
        self.pool = r.maxpool  # /4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # /4 /8 /16 /32
        dec = [256, 128, 64, 32]
        self.up4 = UpBlock(enc_ch[4], enc_ch[3], dec[0])
        self.up3 = UpBlock(dec[0], enc_ch[2], dec[1])
        self.up2 = UpBlock(dec[1], enc_ch[1], dec[2])
        self.up1 = UpBlock(dec[2], enc_ch[0], dec[3])
        self.final = nn.Sequential(ConvBlock(dec[3], 16), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = F.interpolate(d, size=size, mode="bilinear", align_corners=False)
        return self.final(d)


def build_segmenter(arch: str = "resnet34_unet", pretrained: bool = True, base: int = 32) -> nn.Module:
    if arch == "unet":
        return UNet(base=base)
    if arch in ("resnet18_unet", "resnet34_unet", "resnet50_unet"):
        return ResNetUNet(arch.split("_")[0], pretrained)
    raise ValueError(f"unknown segmentation arch {arch}")
