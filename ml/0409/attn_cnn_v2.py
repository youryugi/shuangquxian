"""
attn_cnn 的加强版：让显式注意力真正影响 bbox 检测（方向 A + B）。
任务不变：from-scratch CNN（CenterNet 头）预测 bbox，用双曲线带 mask 监督注意力图。

改进（针对"注意力学到了带、但没传导到检测"的诊断）：
  A. 多尺度门控——注意力图 A 同时门控 stride4(f3) 和 stride8(bottleneck) 特征，
     而非原版只门控单层 bottleneck（原版门控太弱）。
  B. 注意力先验 concat 进检测头输入——把 A(stride8) 作为额外通道喂给检测头，
     让网络自适应学习"怎么用带的位置先验"，而非只靠固定乘法门控公式。

门控仍只增强 (1+A) 不抑制背景（抑制背景实测有害，背景上下文对 GPR 双曲线检测有用）。
no_attn 配置与原版 baseline 完全等价（无 attn_head / 无门控 / 无 concat），保证消融公平。
复用 attn_cnn 的 AttnDataset / compute_loss / predict / evaluate；只替换网络。
多 seed（70/15/15）消融 with_attn vs no_attn，数值可与原版 attn_multiseed 直接对比。
"""
import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import attn_cnn as ac

exp        = ac.exp
device     = ac.device
input_size = ac.input_size

# 70/15/15：make_split 默认参数绑定 0.5/0.25，必须 patch 显式传比例。
_orig_make_split = exp.make_split
exp.make_split = lambda n, s: _orig_make_split(n, s, train_frac=0.70, val_frac=0.15)

SEEDS = [0, 1, 2, 3, 4]


class AttnBBoxNetV2(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.down1 = exp.DownBlock(in_ch, base_ch)
        self.down2 = exp.DownBlock(base_ch, base_ch * 2)
        self.down3 = exp.DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = exp.ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
        if use_attn:
            self.attn_head = nn.Sequential(
                nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
                nn.Conv2d(base_ch * 2, 1, 1))
        head_in = mid + (1 if use_attn else 0)   # B. concat 注意力先验 → 检测头多 1 通道
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(head_in, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, 1, 1))
        self.wh_head = nn.Sequential(
            nn.Conv2d(head_in, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, 2, 1))
        self.offset_head = nn.Sequential(
            nn.Conv2d(head_in, mid // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 4), nn.ReLU(inplace=True), nn.Conv2d(mid // 4, 2, 1))

    def forward(self, x):
        f1, x = self.down1(x)        # x: base_ch    @ s2
        f2, x = self.down2(x)        # x: base_ch*2  @ s4
        f3, _ = self.down3(x)        # f3: base_ch*4 @ s4（忽略 down3 的池化，下面用门控后的 f3 重新池化）
        a_logit = None
        if self.use_attn:
            a_logit = self.attn_head(f3)          # 1 @ s4（双曲线带监督）
            A = torch.sigmoid(a_logit)
            f3 = f3 * (1.0 + A)                    # A. 门控 stride4 特征
        x = F.max_pool2d(f3, 2)                   # base_ch*4 @ s8（等价 down3 内部 MaxPool2d(2)）
        feat = self.bottleneck(x)                 # base_ch*8 @ s8
        if self.use_attn:
            A8 = F.avg_pool2d(torch.sigmoid(a_logit), 2)   # 1 @ s8
            feat = feat * (1.0 + A8)              # A. 门控 stride8 特征
            feat = torch.cat([feat, A8], dim=1)   # B. concat 注意力先验给检测头
        return self.heatmap_head(feat), torch.sigmoid(self.wh_head(feat)), self.offset_head(feat), a_logit


def train_model(use_attn, full, tr, va, n_ep, work, tag):
    exp.set_seed(ac.SEED)
    tl = DataLoader(Subset(full, tr), batch_size=ac.batch_size, shuffle=True,
                    num_workers=0, collate_fn=ac.collate)
    vl = DataLoader(Subset(full, va), batch_size=ac.batch_size, shuffle=False,
                    num_workers=0, collate_fn=ac.collate)
    model = AttnBBoxNetV2(in_ch=1, base_ch=32, use_attn=use_attn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ac.LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_ep + 1):
        model.train()
        for img, hm, wh, off, pk, band, _ in tl:
            img, hm, wh, off, pk, band = [t.to(device) for t in (img, hm, wh, off, pk, band)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = ac.compute_loss(model, img, hm, wh, off, pk, band)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, hm, wh, off, pk, band, _ in vl:
                img, hm, wh, off, pk, band = [t.to(device) for t in (img, hm, wh, off, pk, band)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += ac.compute_loss(model, img, hm, wh, off, pk, band).item()
        va_loss = tot / max(len(vl), 1)
        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} val={va_loss:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


def main():
    now = datetime.now()
    work = os.path.join(os.getcwd(), f"attn_cnn_v2_{now.strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = ac.AttnDataset(input_size=input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    n_total = len(full)
    print(f"[V2 多尺度门控 + 注意力 concat]  n_total={n_total}  seeds={SEEDS}")

    rows = []
    for seed in SEEDS:
        tr, va, te = exp.make_split(n_total, seed)
        print(f"\n=== seed {seed}  train={len(tr)} val={len(va)} test={len(te)} ===")
        for use_attn, tag in [(True, "with_attn"), (False, "no_attn")]:
            ac.SEED = seed
            model = train_model(use_attn, full, tr, va, ac.num_epochs, work, f"seed{seed}_{tag}")
            m = ac.evaluate(model, full, te)
            print(f"  seed{seed} {tag}: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
                  f"F1={m['bbox_F1']:.4f} attn_iou={m['attn_band_iou']:.4f}")
            row = {"seed": seed, "config": tag}; row.update(m); rows.append(row)

    keys = ["bbox_P", "bbox_R", "bbox_F1", "attn_band_iou"]
    print("\n" + "=" * 70)
    print(f"{'config':>12}" + "".join(f"{k:>16}" for k in keys))
    for tag in ["with_attn", "no_attn"]:
        sub = [r for r in rows if r["config"] == tag]
        line = f"{tag:>12}"
        for k in keys:
            vals = [r[k] for r in sub if not (isinstance(r[k], float) and np.isnan(r[k]))]
            line += f"{np.mean(vals):>8.4f}±{np.std(vals):<7.4f}" if vals else f"{'nan':>16}"
        print(line)
    print("=" * 70)

    with open(os.path.join(work, "v2_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}")


if __name__ == "__main__":
    main()
