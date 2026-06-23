"""
简单分割 CNN baseline：直接像素分割出【双曲线带 mask】，而不是预测框。

定位（对比谱系里的一环）：
  - 简单分割 CNN（本文件）：U-Net 直接分割带，无双曲线结构先验，纯数据驱动。
  - HyperbolaNet（你的方法）：参数化(顶点+开口+高度) → 渲染带，有强结构先验。
  - YOLO / bbox CNN：只给框，丢了带形状。
用同样的【带级指标 global_iou / pixel_f1】评估，可与 HyperbolaNet 直接并排，
衬托"双曲线参数化先验"相对"纯分割"的价值。

网络 = HyperbolaNet 的 U-Net 部分（DownBlock×3 + bottleneck + UpBlock×3 + seg_head），
监督 = band mask（exp.seg_loss_fn，BCE+Dice）。
注意：纯语义分割不区分实例（两条挨近的双曲线会连成一片）——这正是它的弱点。
多 seed，70/15/15，输出 global_iou / pixel_f1 + csv。
"""
import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
exp = _load("0616-1.py", "exp0616")

device     = exp.device
input_size = exp.input_size
SEEDS      = [0, 1, 2, 3, 4]
N_EPOCHS   = exp.num_epochs
SEED       = 0


class SimpleBandCNN(nn.Module):
    """U-Net 分割：输出整张双曲线带 mask logit（与 HyperbolaNet 的分割分支同结构）。"""
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.down1 = exp.DownBlock(in_ch, base_ch)
        self.down2 = exp.DownBlock(base_ch, base_ch * 2)
        self.down3 = exp.DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = exp.ConvBlock(base_ch * 4, base_ch * 8)
        self.up3 = exp.UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = exp.UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = exp.UpBlock(base_ch * 2, base_ch, base_ch)
        self.seg_head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        f1, x = self.down1(x)
        f2, x = self.down2(x)
        f3, x = self.down3(x)
        x = self.bottleneck(x)
        d = self.up3(x, f3)
        d = self.up2(d, f2)
        d = self.up1(d, f1)
        return self.seg_head(d)        # (B,1,H,W) band logit


def train_model(full, tr, va, n_ep, work, tag):
    exp.set_seed(SEED)
    tl = DataLoader(exp.AugWrapper(Subset(full, tr)), batch_size=exp.batch_size,
                    shuffle=True, num_workers=0, collate_fn=exp.hyperbola_collate_fn)
    vl = DataLoader(Subset(full, va), batch_size=exp.batch_size,
                    shuffle=False, num_workers=0, collate_fn=exp.hyperbola_collate_fn)
    model = SimpleBandCNN(in_ch=1, base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=exp.LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_ep + 1):
        model.train()
        for images, _, _, _, _, gt_seg, _ in tl:
            images, gt_seg = images.to(device), gt_seg.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = exp.seg_loss_fn(model(images), gt_seg)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for images, _, _, _, _, gt_seg, _ in vl:
                images, gt_seg = images.to(device), gt_seg.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += exp.seg_loss_fn(model(images), gt_seg).item()
        va_loss = tot / max(len(vl), 1)
        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} val={va_loss:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


@torch.no_grad()
def evaluate(model, full, test_idx):
    """带级指标：global_iou(整体像素 IoU) + pixel_f1。"""
    inter = union = 0.0
    tp = fp = fn = 0.0
    for i in test_idx:
        images = full[i][0].unsqueeze(0).to(device)
        gt_seg = full[i][5][0].numpy() > 0.5
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = torch.sigmoid(model(images))[0, 0].float().cpu().numpy() > 0.5
        inter += np.logical_and(pred, gt_seg).sum()
        union += np.logical_or(pred, gt_seg).sum()
        tp += np.logical_and(pred, gt_seg).sum()
        fp += np.logical_and(pred, ~gt_seg).sum()
        fn += np.logical_and(~pred, gt_seg).sum()
    return {"global_iou": float(inter / max(union, 1e-6)),
            "pixel_f1": float(2 * tp / max(2 * tp + fp + fn, 1e-6))}


def main():
    global SEED
    work = os.path.join(os.getcwd(), f"simple_band_cnn_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    src = exp.data_sources[0]
    full = exp.HyperbolaDataset(image_dir=src["image_dir"], annotation_json=src["annotation_json"],
                                input_size=input_size, hm_stride=exp.HM_STRIDE, sigma=exp.HM_SIGMA)
    n = len(full)
    print(f"[简单分割 CNN(预测双曲线带)]  n_total={n}  seeds={SEEDS}  70/15/15", flush=True)

    rows = []
    for seed in SEEDS:
        SEED = seed
        tr, va, te = exp.make_split(n, seed, train_frac=0.70, val_frac=0.15)
        print(f"\n=== seed {seed}  train={len(tr)} val={len(va)} test={len(te)} ===", flush=True)
        model = train_model(full, tr, va, N_EPOCHS, work, f"seed{seed}")
        m = evaluate(model, full, te)
        print(f"  seed{seed}: global_iou={m['global_iou']:.4f} pixel_f1={m['pixel_f1']:.4f}", flush=True)
        row = {"seed": seed}; row.update(m); rows.append(row)

    keys = ["global_iou", "pixel_f1"]
    print("\n" + "=" * 50)
    print(f"{'metric':>14}{'mean':>12}{'std':>12}")
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"{k:>14}{np.mean(vals):>12.4f}{np.std(vals):>12.4f}")
    print("=" * 50)

    with open(os.path.join(work, "simple_band_cnn_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}", flush=True)


if __name__ == "__main__":
    main()
