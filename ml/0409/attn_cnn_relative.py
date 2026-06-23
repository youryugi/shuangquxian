"""
相对约束注意力：不要求 A(x) ≈ band mask（绝对逐像素匹配），只要求
【带内平均注意力 显著高于 带外平均注意力】：

    L_attn = mean_batch  max(0, margin − (mean_attn_in − mean_attn_out))

设计来源（严谨定位，勿过度声称原创）：
  - 数学形式就是经典的 margin / hinge ranking loss；"正区域 > 负区域 + margin"
    的思想见于 triplet loss、对比学习、以及弱监督定位/多示例学习(MIL/WSOL)。
  - 本文的具体之处：把它用于"双曲线带内 vs 带外注意力"，motivation 是——
    绝对匹配(BCE+Dice)逼 A 精确贴合薄带 → attn_iou 仅 0.19(薄结构+边界对齐太苛刻)；
    而真正想要的是"多看带、少看背景"这个【相对关系】(实测带内/带外≈4.3×,gap≈0.28)，
    相对约束直接优化它、对边界误差几乎免疫。

复用 attn_cnn 的数据集 / 网络 / predict / evaluate；只替换注意力 loss。
main：同 seed/同 70-15-15，三方对比 no_attn / abs(BCE+Dice) / rel(margin)，看相对约束是否更好。
"""
import os
import csv
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import attn_cnn as ac

exp        = ac.exp
device     = ac.device
input_size = ac.input_size

SEEDS   = [0, 1, 2, 3, 4]
MARGIN  = 0.5     # 要求 带内均值 − 带外均值 ≥ margin
LAM_ATT = 1.0


def attn_loss_relative(a_logit, band, margin=MARGIN):
    """margin hinge：只约束带内/带外平均注意力的相对差，不做逐像素匹配。"""
    A = torch.sigmoid(a_logit)                       # (B,1,h,w)
    eps = 1e-6
    s_in  = (A * band).sum(dim=(1, 2, 3))
    s_out = (A * (1.0 - band)).sum(dim=(1, 2, 3))
    n_in  = band.sum(dim=(1, 2, 3)).clamp(min=eps)
    n_out = (1.0 - band).sum(dim=(1, 2, 3)).clamp(min=eps)
    mean_in  = s_in / n_in
    mean_out = s_out / n_out
    return torch.relu(margin - (mean_in - mean_out)).mean()


def compute_loss(model, img, hm, wh, off, peak, band, attn_mode):
    hm_logit, wh_p, off_p, a_logit = model(img)
    loss = (exp.focal_loss_heatmap(hm_logit, hm, peak)
            + ac.masked_l1(wh_p, wh, peak) + ac.masked_l1(off_p, off, peak))
    if model.use_attn:
        if attn_mode == "abs":
            loss = loss + LAM_ATT * ac.attn_loss(a_logit, band)            # BCE+Dice 绝对匹配
        elif attn_mode == "rel":
            loss = loss + LAM_ATT * attn_loss_relative(a_logit, band)      # margin 相对约束
    return loss


def train_model(attn_mode, full, tr, va, n_ep, work, tag):
    exp.set_seed(ac.SEED)
    use_attn = (attn_mode != "none")
    tl = DataLoader(Subset(full, tr), batch_size=ac.batch_size, shuffle=True,
                    num_workers=0, collate_fn=ac.collate)
    vl = DataLoader(Subset(full, va), batch_size=ac.batch_size, shuffle=False,
                    num_workers=0, collate_fn=ac.collate)
    model = ac.AttnBBoxNet(in_ch=1, base_ch=32, use_attn=use_attn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ac.LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_ep + 1):
        model.train()
        for img, hm, wh, off, pk, band, _ in tl:
            img, hm, wh, off, pk, band = [t.to(device) for t in (img, hm, wh, off, pk, band)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = compute_loss(model, img, hm, wh, off, pk, band, attn_mode)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, hm, wh, off, pk, band, _ in vl:
                img, hm, wh, off, pk, band = [t.to(device) for t in (img, hm, wh, off, pk, band)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += compute_loss(model, img, hm, wh, off, pk, band, attn_mode).item()
        va_loss = tot / max(len(vl), 1)
        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} val={va_loss:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


@torch.no_grad()
def attn_gap(model, full, test_idx):
    """test 上的带内/带外平均注意力差(relative 的直接目标)。"""
    ins, outs = [], []
    for i in test_idx:
        img, _, _, _, _, band, _ = full[i]
        _, _, A = ac.predict(model, img.unsqueeze(0))
        if A is None:
            return float("nan"), float("nan"), float("nan")
        b = band[0].numpy()
        m = b > 0.5
        if m.sum() < 1:
            continue
        ins.append(float(A[m].mean())); outs.append(float(A[~m].mean()))
    mi, mo = (np.mean(ins), np.mean(outs)) if ins else (float("nan"), float("nan"))
    return mi, mo, mi - mo


def main():
    work = os.path.join(os.getcwd(), f"attn_cnn_relative_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = ac.AttnDataset(input_size=input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    n = len(full)
    print(f"[相对约束注意力] n_total={n}  margin={MARGIN}  seeds={SEEDS}  70/15/15", flush=True)

    rows = []
    for seed in SEEDS:
        tr, va, te = exp.make_split(n, seed, train_frac=0.70, val_frac=0.15)
        print(f"\n=== seed {seed}  train={len(tr)} val={len(va)} test={len(te)} ===", flush=True)
        for attn_mode in ["abs", "rel"]:   # no_attn 基线已有(simple_bbox_cnn F1 0.627±0.074),不重训
            ac.SEED = seed
            model = train_model(attn_mode, full, tr, va, ac.num_epochs, work, f"seed{seed}_{attn_mode}")
            m = ac.evaluate(model, full, te)
            mi, mo, gap = attn_gap(model, full, te)
            print(f"  seed{seed} {attn_mode:>4}: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
                  f"F1={m['bbox_F1']:.4f} attn_iou={m['attn_band_iou']:.4f} gap={gap:.4f}", flush=True)
            row = {"seed": seed, "config": attn_mode, "bbox_P": m["bbox_P"], "bbox_R": m["bbox_R"],
                   "bbox_F1": m["bbox_F1"], "attn_band_iou": m["attn_band_iou"], "attn_gap": gap}
            rows.append(row)

    keys = ["bbox_P", "bbox_R", "bbox_F1", "attn_band_iou", "attn_gap"]
    print("\n" + "=" * 88)
    print(f"{'config':>8}" + "".join(f"{k:>16}" for k in keys))
    for cfg in ["rel"]:
        sub = [r for r in rows if r["config"] == cfg]
        line = f"{cfg:>8}"
        for k in keys:
            vals = [r[k] for r in sub if not (isinstance(r[k], float) and np.isnan(r[k]))]
            line += f"{np.mean(vals):>8.4f}±{np.std(vals):<7.4f}" if vals else f"{'nan':>16}"
        print(line)
    print("=" * 88)

    with open(os.path.join(work, "relative_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}", flush=True)


if __name__ == "__main__":
    main()
