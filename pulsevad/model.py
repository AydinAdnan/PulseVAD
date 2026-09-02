"""PulseVAD CNN backbone (spec phase-02).

CNN-only: depthwise-separable + dilated convolutions, GAP head, no recurrence,
no custom activations. Every conv is bias-free; every BatchNorm1d uses
eps=1e-3 / momentum=0.1 (spec rule #4). Unpruned model is exactly 81,090 params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-3
BN_MOMENTUM = 0.1


class ConvBNReLU(nn.Module):
    """1D conv (bias-free) + BatchNorm1d(eps=1e-3) + optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        relu: bool = True,
        eps: float = BN_EPS,
        momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels, eps=eps, momentum=momentum)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """block3: two depthwise-separable sub-blocks with a 1x1-conv skip."""

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 17,
        dropout: float = 0.1,
        eps: float = BN_EPS,
        momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2  # 'same' padding for stride 1
        # Sub-block A: DW(k=17) -> PW(1x1)+BN+ReLU+Dropout
        self.subA_dw = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, groups=channels, bias=False
        )
        self.subA_pw = ConvBNReLU(channels, channels, 1, eps=eps, momentum=momentum)
        self.subA_drop = nn.Dropout(dropout)
        # Sub-block B: DW(k=17) -> PW(1x1)+BN (no activation, no dropout)
        self.subC_dw = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, groups=channels, bias=False
        )
        self.subC_pw = ConvBNReLU(
            channels, channels, 1, relu=False, eps=eps, momentum=momentum
        )
        # Skip branch: 1x1 conv + BN
        self.skip = ConvBNReLU(channels, channels, 1, relu=False, eps=eps, momentum=momentum)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.subA_drop(self.subA_pw(self.subA_dw(x)))
        main = self.subC_pw(self.subC_dw(main))
        return self.out_drop(F.relu(main + self.skip(x)))


class PulseVAD(nn.Module):
    """Unpruned baseline: (B, 64, 21) -> logits (B, 2) [non-speech, speech]."""

    def __init__(
        self,
        bn_eps: float = BN_EPS,
        bn_momentum: float = BN_MOMENTUM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = ConvBNReLU(64, 128, 1, eps=bn_eps, momentum=bn_momentum)  # 8,448
        self.conv0_dw = nn.Conv1d(
            128, 128, 11, padding=5, groups=128, bias=False
        )  # 1,408 (no BN, no ReLU)
        self.conv0_pw = ConvBNReLU(128, 128, 1, eps=bn_eps, momentum=bn_momentum)  # 16,640
        self.block1 = ConvBNReLU(128, 64, 1, eps=bn_eps, momentum=bn_momentum)  # 8,320
        self.block2 = ConvBNReLU(64, 64, 1, eps=bn_eps, momentum=bn_momentum)  # 4,224
        self.block3 = ResidualBlock(64, 17, dropout=dropout)  # 14,848
        self.conv4_dw = nn.Conv1d(
            64, 64, 29, dilation=2, padding=28, groups=64, bias=False
        )  # 1,856 (no BN, no ReLU)
        self.conv4_pw = ConvBNReLU(64, 128, 1, eps=bn_eps, momentum=bn_momentum)  # 8,448
        self.conv5 = ConvBNReLU(128, 128, 1, eps=bn_eps, momentum=bn_momentum)  # 16,640
        self.gap = nn.AdaptiveAvgPool1d(1)  # 0
        self.classifier = nn.Linear(128, 2, bias=True)  # 258

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        x = self.adapter(x)  # (B, 128, 21)
        x = self.conv0_dw(x)
        x = self.conv0_pw(x)
        x = self.block1(x)  # (B, 64, 21)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv4_dw(x)
        x = self.conv4_pw(x)  # (B, 128, 21)
        x = self.conv5(x)
        x = self.gap(x).squeeze(-1)  # (B, 128), width-independent
        logits = self.classifier(x)  # (B, 2)
        if return_logits:
            return logits
        return torch.softmax(logits, dim=-1)[:, 1]
