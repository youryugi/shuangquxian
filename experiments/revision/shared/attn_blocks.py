"""通用即插即用注意力模块（无监督），实验 4（注意力机制对比）用。

从投稿版 Table III 的脚本原样搬来（experiments/submission/primary_yolohead/generic_attention_baselines.py，
SEBlock / CBAM / NonLocalBlock / CoordAttn / build_attn_block），超参数不变：
  se       SE 通道注意力（Hu 2018），r = 16
  cbam     CBAM（Woo 2018）：通道注意力 + 7×7 空间注意力串联，r = 16
  nonlocal Non-local（Wang 2018，embedded Gaussian），phi / g 2×2 maxpool 子采样，输出卷积零初始化（起点为恒等）
  coord    Coordinate Attention（Hou 2021），r = 32
全部只用确定性算子，兼容 torch.use_deterministic_algorithms(True)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

ATTN_BLOCKS = ("none", "se", "cbam", "nonlocal", "coord")


class SEBlock(nn.Module):
    """SE 通道注意力：全局平均池化 → 两层 1×1 MLP → sigmoid 通道权重。"""

    def __init__(self, ch, r=16):
        super().__init__()
        h = max(ch // r, 4)
        self.fc = nn.Sequential(nn.Conv2d(ch, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, ch, 1))

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc(s))


class _ChannelAttn(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        h = max(ch // r, 4)
        self.mlp = nn.Sequential(nn.Conv2d(ch, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, ch, 1))

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class _SpatialAttn(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """CBAM：通道注意力 + 空间注意力串联。"""

    def __init__(self, ch, r=16, k=7):
        super().__init__()
        self.ca = _ChannelAttn(ch, r)
        self.sa = _SpatialAttn(k)

    def forward(self, x):
        return self.sa(self.ca(x))


class NonLocalBlock(nn.Module):
    """Non-local 空间自注意力（embedded Gaussian）；phi / g 子采样，out 卷积零初始化 → 残差起点为恒等映射。"""

    def __init__(self, ch, sub=2):
        super().__init__()
        inter = max(ch // 2, 1)
        self.inter = inter
        self.theta = nn.Conv2d(ch, inter, 1)
        self.phi = nn.Conv2d(ch, inter, 1)
        self.g = nn.Conv2d(ch, inter, 1)
        self.pool = nn.MaxPool2d(sub) if sub > 1 else nn.Identity()
        self.out = nn.Conv2d(inter, ch, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        theta = self.theta(x).reshape(b, self.inter, h * w).permute(0, 2, 1)
        phi = self.pool(self.phi(x)).reshape(b, self.inter, -1)
        attn = torch.softmax(torch.bmm(theta, phi) / (self.inter ** 0.5), dim=-1)
        g = self.pool(self.g(x)).reshape(b, self.inter, -1).permute(0, 2, 1)
        y = torch.bmm(attn, g).permute(0, 2, 1).reshape(b, self.inter, h, w)
        return x + self.out(y)


class CoordAttn(nn.Module):
    """Coordinate Attention：沿 H、W 分别池化，编码位置信息后做注意力。"""

    def __init__(self, ch, r=32):
        super().__init__()
        h = max(ch // r, 8)
        self.conv1 = nn.Conv2d(ch, h, 1)
        self.bn1 = nn.BatchNorm2d(h)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(h, ch, 1)
        self.conv_w = nn.Conv2d(h, ch, 1)

    def forward(self, x):
        b, c, H, W = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [H, W], dim=2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))
        return x * a_h * a_w


def build_attn_block(name, ch):
    name = (name or "none").lower()
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SEBlock(ch)
    if name == "cbam":
        return CBAM(ch)
    if name == "nonlocal":
        return NonLocalBlock(ch)
    if name == "coord":
        return CoordAttn(ch)
    raise ValueError(f"unknown attn block: {name} (choose from {ATTN_BLOCKS})")
