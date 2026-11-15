"""
HyperbolaNet「不预测厚度」版：厚度固定为常数，不再作为有意义的回归目标。

动机：厚度对"双曲线在哪、开口多大、多高"不是必需信息，固定它可简化任务。

实现（零改 0616-1.py 管线，最大限度复用已验证代码）：
  - 子类数据集把厚度全程固定成常数 FIXED_NORM（GT 的 param[2]、band、gt_seg 都用它），
    常数取训练集厚度（归一化）的中位数。
  - 网络仍用原 exp.HyperbolaNet（param_head 输出 3 通道），但因 GT 厚度恒定，
    厚度通道会被训练成常数 → 等价于"厚度不参与有意义的预测"。
  - loss / 精修 / band / 解码 / 评估全部复用 exp，保证与原版同口径、无 bug。

main：同条件（同 seed / 同 70-15-15 split）对比
  - pred_thick : 原版（预测厚度）
  - fixed_thick: 本方案（固定厚度）
直接回答"不预测厚度好不好"。指标存 csv。
"""
import os
import csv
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
exp = _load("0616-1.py", "exp0616")

device     = exp.device
input_size = exp.input_size
SEED       = 0
N_EPOCHS   = exp.num_epochs


class HyperbolaDatasetNoThick(exp.HyperbolaDataset):
    """复用原数据集，只把厚度固定成常数（param[2] / gt_seg / meta 都用 fixed）。"""
    def __init__(self, *args, fixed_norm, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_norm = float(fixed_norm)
        self.fixed_px = self.fixed_norm * self.input_h   # input 尺度像素

    def __getitem__(self, idx):
        img, hm, pm, om, pk, seg, meta = super().__getitem__(idx)
        pm[2] = self.fixed_norm                          # GT 厚度通道固定（loss 只看 peak 处）
        for o in meta["objects"]:
            o["thickness"] = self.fixed_px               # meta 厚度固定（评估/解码用）
        # 用固定厚度重渲染 GT band（保持 seg/band 监督与固定厚度自洽）
        seg_np = np.zeros((self.input_h, self.input_w), np.float32)
        for o in meta["objects"]:
            seg_np = np.maximum(seg_np, exp.detection_to_mask(o, (self.input_h, self.input_w)))
        seg = torch.from_numpy(seg_np).unsqueeze(0).float()
        return img, hm, pm, om, pk, seg, meta


def compute_fixed_norm(base):
    """训练集所有双曲线的归一化厚度中位数（峰值处 param[2]）。"""
    vals = []
    for i in range(len(base)):
        _, _, pm, _, pk, _, _ = base[i]
        m = pk[0] > 0.5
        if m.any():
            vals.extend(pm[2][m].tolist())
    return float(np.median(vals)) if vals else 0.08


def train_model(full, tr, va, n_ep, work, tag):
    exp.set_seed(SEED)
    tl = DataLoader(exp.AugWrapper(Subset(full, tr)), batch_size=exp.batch_size,
                    shuffle=True, num_workers=0, collate_fn=exp.hyperbola_collate_fn)
    vl = DataLoader(Subset(full, va), batch_size=exp.batch_size,
                    shuffle=False, num_workers=0, collate_fn=exp.hyperbola_collate_fn)
    model = exp.HyperbolaNet(in_ch=1, base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=exp.LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_ep + 1):
        model.train()
        for images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg, _ in tl:
            images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg = [
                t.to(device) for t in (images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = exp._compute_total_loss(model, images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg, _ in vl:
                images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg = [
                    t.to(device) for t in (images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += exp._compute_total_loss(model, images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg).item()
        va_loss = tot / max(len(vl), 1)
        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} val={va_loss:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


def main():
    work = os.path.join(os.getcwd(), f"hyperbola_nothick_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)

    src = exp.data_sources[0]
    base = exp.HyperbolaDataset(image_dir=src["image_dir"], annotation_json=src["annotation_json"],
                                input_size=input_size, hm_stride=exp.HM_STRIDE, sigma=exp.HM_SIGMA)
    n = len(base)
    fixed_norm = compute_fixed_norm(base)
    print(f"[HyperbolaNet 厚度消融]  total={n}  固定厚度 FIXED_NORM={fixed_norm:.4f} "
          f"(≈{fixed_norm*input_size[0]:.1f}px@{input_size[0]})", flush=True)

    keys = ["global_iou", "pixel_f1", "mAP50", "instance_recall", "mean_vertex_dist"]
    rows = []
    for tag in ["pred_thick", "fixed_thick"]:
        if tag == "pred_thick":
            full = base
        else:
            full = HyperbolaDatasetNoThick(
                image_dir=src["image_dir"], annotation_json=src["annotation_json"],
                input_size=input_size, hm_stride=exp.HM_STRIDE, sigma=exp.HM_SIGMA,
                fixed_norm=fixed_norm)
        tr, va, te = exp.make_split(n, SEED, train_frac=0.70, val_frac=0.15)
        print(f"\n=== {tag}  train={len(tr)} val={len(va)} test={len(te)} ===", flush=True)
        model = train_model(full, tr, va, N_EPOCHS, work, tag)
        m = exp.evaluate(model, Subset(full, te), device, input_size, obj_thresh=exp.HM_THRESH)
        print(f"  {tag}: " + "  ".join(f"{k}={m[k]:.4f}" for k in keys), flush=True)
        row = {"config": tag}; row.update({k: m[k] for k in keys}); rows.append(row)

    print("\n" + "=" * 90)
    print(f"{'config':>12}" + "".join(f"{k:>16}" for k in keys))
    for r in rows:
        print(f"{r['config']:>12}" + "".join(f"{r[k]:>16.4f}" for k in keys))
    print("=" * 90)

    with open(os.path.join(work, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}", flush=True)


if __name__ == "__main__":
    main()
