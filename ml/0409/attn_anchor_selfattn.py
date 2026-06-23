"""
在 attn_anchor_nms.py 基础上加 self-attention（自注意力），建模双曲线两臂的长程/对称关系。

动机：现有的 band 门控注意力是【逐像素、无位置交互】的——只能说"这里重不重要"，
表达不了"左臂和右臂对称关联"这种成对关系。self-attention 计算每对位置的相关性，
理论上能让弧上的点互相聚合信息、发现对称结构。

实现：
  - 复用 attn_anchor_nms 的全部组件（anchor 生成/编解码/NMS/数据集/loss/predict/evaluate）。
  - 网络在 bottleneck(stride8) 特征上插入一个 SAGAN 风格 self-attention 块（残差 + γ 初始 0，
    训练稳定）。为省显存，在下采样到 40×40 后做注意力(O(N²) 的 N 从 6400 降到 1600)，
    再上采样残差融合。
  - 保留 band 门控注意力(use_attn=True)，只消融"加不加 self-attention"(use_sa)。
单 seed，70/15/15，with_sa vs no_sa，存 results.csv。

诚实提醒：双曲线有明确数学形式，HyperbolaNet 的参数化拟合是更强的结构先验；
self-attention 是"让网络自己发现结构"的较弱先验。这个实验是为了验证它到底有没有用。
我觉得自注意力是务必要的
"""
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import attn_anchor_nms as aan

exp        = aan.exp
device     = aan.device
input_size = aan.input_size
batch_size = aan.batch_size
LR         = aan.LR
SEED       = aan.SEED
EPOCHS     = 100          # 充分训练（attn_anchor_nms 里被改成 20，这里独立设 100）
SA_DOWN    = 2            # self-attention 前的下采样倍数（80x80 -> 40x40，省显存）


class SelfAttention2d(nn.Module):
    """SAGAN 风格 self-attention：q·k 相关 -> softmax -> 聚合 v，残差 γ 初始 0。"""
    def __init__(self, ch, reduction=8):
        super().__init__()
        c = max(ch // reduction, 1)
        self.q = nn.Conv2d(ch, c, 1)
        self.k = nn.Conv2d(ch, c, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).flatten(2).permute(0, 2, 1)        # B, N, c
        k = self.k(x).flatten(2)                         # B, c, N
        attn = torch.softmax(torch.bmm(q, k), dim=-1)    # B, N, N（每个位置对所有位置的关注）
        v = self.v(x).flatten(2)                         # B, C, N
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class AnchorSelfAttnNet(nn.Module):
    """attn_anchor_nms 的网络 + 可选 self-attention 块。接口与 AnchorAttnNet 一致。"""
    def __init__(self, in_ch=1, base_ch=32, num_anchors=aan.NUM_ANCHORS, use_attn=True, use_sa=True):
        super().__init__()
        self.use_attn = use_attn
        self.use_sa = use_sa
        self.K = num_anchors
        self.down1 = exp.DownBlock(in_ch, base_ch)
        self.down2 = exp.DownBlock(base_ch, base_ch * 2)
        self.down3 = exp.DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = exp.ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
        if use_sa:
            self.self_attn = SelfAttention2d(mid, reduction=8)
        if use_attn:
            self.attn_head = nn.Sequential(
                nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 2, 1, 1))
        self.cls_head = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, num_anchors, 1))
        self.reg_head = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, num_anchors * 4, 1))
        import math
        nn.init.constant_(self.cls_head[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        _, x = self.down1(x); _, x = self.down2(x)
        f3, x = self.down3(x)
        feat = self.bottleneck(x)                        # stride8 (80x80)
        if self.use_sa:
            fs = F.avg_pool2d(feat, SA_DOWN)             # 40x40，省显存
            fs = self.self_attn(fs)
            fs = F.interpolate(fs, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            feat = feat + fs                             # 残差融合长程结构信息
        a_logit = None
        if self.use_attn:
            a_logit = self.attn_head(f3)
            gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)
            feat = feat * (1.0 + gate)
        return self.cls_head(feat), self.reg_head(feat), a_logit


def train_model(use_attn, use_sa, full, train_idx, val_idx, n_epochs, work, tag):
    exp.set_seed(SEED)
    anchors = aan.get_anchors(device)
    tl = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True,
                    num_workers=0, collate_fn=aan.collate)
    vl = DataLoader(Subset(full, val_idx), batch_size=batch_size, shuffle=False,
                    num_workers=0, collate_fn=aan.collate)
    model = AnchorSelfAttnNet(in_ch=1, base_ch=32, use_attn=use_attn, use_sa=use_sa).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_epochs + 1):
        model.train()
        for img, band, gts, _ in tl:
            img, band = img.to(device), band.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = aan.compute_loss(model, img, gts, band, anchors)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, band, gts, _ in vl:
                img, band = img.to(device), band.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += aan.compute_loss(model, img, gts, band, anchors).item()
        va = tot / max(len(vl), 1)
        if va < best:
            best = va; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_epochs:
            print(f"  [{tag}] epoch {ep}/{n_epochs} val={va:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


def main():
    exp.set_seed(SEED)
    full = aan.AnchorDataset(input_size=input_size, hm_stride=aan.HM_STRIDE, sigma=aan.HM_SIGMA)
    n = len(full)
    train_idx, val_idx, test_idx = exp.make_split(n, SEED, train_frac=0.70, val_frac=0.15)
    work = os.path.join(os.getcwd(), "anchor_selfattn_out"); os.makedirs(work, exist_ok=True)
    print(f"[anchor + self-attention]  total={n}  "
          f"split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

    rows = []
    for use_sa, tag in [(False, "no_sa"), (True, "with_sa")]:
        print(f"\n=== {tag} (use_attn=True, use_sa={use_sa}) ===", flush=True)
        model = train_model(True, use_sa, full, train_idx, val_idx, EPOCHS, work, tag)
        m = aan.evaluate(model, full, test_idx)
        print(f"[{tag}] P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
              f"F1={m['bbox_F1']:.4f} attn_band_iou={m['attn_band_iou']:.4f}", flush=True)
        row = {"config": tag}; row.update(m); rows.append(row)

    csv_path = os.path.join(work, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved metrics -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
