"""
普通 CNN bbox 检测 baseline（CenterNet 风格）。

定位：作为论文里的"朴素 CNN"对照——用与 0616-1.py 完全相同的 backbone，
但不带任何创新（无 band loss / 无注意力精修 / 无分割头 / 无双曲线参数化），
直接预测 bbox（中心点 heatmap + 宽高回归 + 中心 offset）。

公平性：
  - from scratch（无预训练）
  - 与主方法相同的 315 张图、相同的 5-seed 划分（复用 0616-1 的 make_split）
  - 相同的 bbox 评估口径（IoU>=0.5 -> bbox_recall / bbox_mAP50）
GT 框来自 annotations_rect.json。
"""
import os
import csv
import json
import random
import importlib.util
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ── 复用 0616-1.py（保证 backbone / 划分 / 超参完全一致）──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
def load_exp(path):
    spec = importlib.util.spec_from_file_location("exp0616", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
exp = load_exp(os.path.join(_HERE, "0616-1.py"))

device       = exp.device
input_size   = exp.input_size
HM_STRIDE    = exp.HM_STRIDE
HM_SIGMA     = exp.HM_SIGMA
batch_size   = exp.batch_size
num_epochs   = exp.num_epochs
LR           = exp.LR
HM_THRESH    = exp.HM_THRESH
nms_kernel   = exp.nms_kernel
max_det      = exp.max_det
SEEDS        = exp.SEEDS
RECT_JSON_NAME = "annotations_rect.json"
BBOX_IOU_THR   = 0.5


# ── bbox 工具 ─────────────────────────────────────────────────────────────────
def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter  = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def rect_to_bbox_scaled(r, sx, sy):
    return [r["x1"] * sx, r["y1"] * sy, (r["x1"] + r["width"]) * sx, (r["y1"] + r["height"]) * sy]


# ── 数据集：bbox 中心点 heatmap + 宽高 + offset ────────────────────────────────
class BBoxDataset(Dataset):
    def __init__(self, image_dir, hyp_json, rect_json, input_size, hm_stride, sigma):
        super().__init__()
        self.image_dir = image_dir
        self.input_h, self.input_w = input_size
        self.hm_h, self.hm_w = self.input_h // hm_stride, self.input_w // hm_stride
        self.hm_stride, self.sigma = hm_stride, sigma
        with open(rect_json, "r", encoding="utf-8") as f:
            self.rect = json.load(f)
        # 关键：image_names 与 0616-1 的 HyperbolaDataset 完全一致（同 315 张、同排序），
        # 这样 make_split(seed) 给出的 test_idx 指向相同图片。
        with open(hyp_json, "r", encoding="utf-8") as f:
            hyp = json.load(f)
        import re
        self.image_names = sorted(list(hyp.keys()),
                                  key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.image_dir, name)
        img  = Image.open(path).convert("L")
        orig_w, orig_h = img.size
        img  = img.resize((self.input_w, self.input_h), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        heatmap = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)
        wh_map  = np.zeros((2, self.hm_h, self.hm_w), dtype=np.float32)
        off_map = np.zeros((2, self.hm_h, self.hm_w), dtype=np.float32)
        peak    = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)

        sx, sy = self.input_w / orig_w, self.input_h / orig_h
        for r in self.rect.get(name, []):
            if r.get("label", "") != "hyperbola":
                continue
            x1, y1 = r["x1"] * sx, r["y1"] * sy
            bw, bh = r["width"] * sx, r["height"] * sy
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0           # bbox 中心（input 尺度）
            hm_cx, hm_cy = cx / self.hm_stride, cy / self.hm_stride
            xi = int(np.clip(round(hm_cx), 0, self.hm_w - 1))
            yi = int(np.clip(round(hm_cy), 0, self.hm_h - 1))
            exp.render_gaussian(heatmap, hm_cx, hm_cy, self.sigma)
            wh_map[0, yi, xi] = bw / self.input_w           # 归一化宽高
            wh_map[1, yi, xi] = bh / self.input_h
            off_map[0, yi, xi] = hm_cx - xi
            off_map[1, yi, xi] = hm_cy - yi
            peak[yi, xi] = 1.0

        meta = {"image_name": name, "image_path": path, "orig_size": (orig_h, orig_w)}
        return (
            torch.from_numpy(img_np).unsqueeze(0).float(),
            torch.from_numpy(heatmap).unsqueeze(0).float(),
            torch.from_numpy(wh_map).float(),
            torch.from_numpy(off_map).float(),
            torch.from_numpy(peak).unsqueeze(0).float(),
            meta,
        )


def collate(batch):
    imgs, hms, whs, offs, pks, metas = zip(*batch)
    return (torch.stack(imgs), torch.stack(hms), torch.stack(whs),
            torch.stack(offs), torch.stack(pks), list(metas))


# 简单水平翻转 + 亮度增强（与主方法增强对齐，保证公平）
class AugWrapper(Dataset):
    def __init__(self, subset): self.subset = subset
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, hm, wh, off, pk, meta = self.subset[idx]
        img, hm, wh, off, pk = img.numpy()[0], hm.numpy()[0], wh.numpy(), off.numpy(), pk.numpy()[0]
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
            hm  = hm[:, ::-1].copy()
            wh  = wh[:, :, ::-1].copy()
            o   = off[:, :, ::-1].copy(); o[0] = -o[0]; off = o
            pk  = pk[:, ::-1].copy()
        img = np.clip(img * random.uniform(0.7, 1.3), 0.0, 1.0)
        return (torch.from_numpy(img).unsqueeze(0).float(),
                torch.from_numpy(hm).unsqueeze(0).float(),
                torch.from_numpy(wh).float(),
                torch.from_numpy(off).float(),
                torch.from_numpy(pk).unsqueeze(0).float(),
                meta)


# ── 模型：相同 backbone + bbox 三个 head（无任何创新）─────────────────────────
class BBoxNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.down1      = exp.DownBlock(in_ch,       base_ch)
        self.down2      = exp.DownBlock(base_ch,     base_ch * 2)
        self.down3      = exp.DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = exp.ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, 1, 1))
        self.wh_head = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, 2, 1))
        self.offset_head = nn.Sequential(
            nn.Conv2d(mid, mid // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 4), nn.ReLU(inplace=True), nn.Conv2d(mid // 4, 2, 1))

    def forward(self, x):
        _, x = self.down1(x)
        _, x = self.down2(x)
        _, x = self.down3(x)
        x    = self.bottleneck(x)
        return self.heatmap_head(x), torch.sigmoid(self.wh_head(x)), self.offset_head(x)


def masked_l1(pred, gt, peak):
    mask = peak.expand_as(pred)
    n = mask.sum()
    if n == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred * mask, gt * mask, reduction="sum") / (n / pred.shape[1] + 1e-6)


def compute_loss(model, img, gt_hm, gt_wh, gt_off, peak):
    hm_logit, wh, off = model(img)
    return (exp.focal_loss_heatmap(hm_logit, gt_hm, peak)
            + masked_l1(wh, gt_wh, peak)
            + masked_l1(off, gt_off, peak))


# AMP 混合精度：用 bfloat16 算前向/反向，显著减少 GPU 计算量。
# 选 bf16 而非 fp16：focal loss 里有 log/clamp(1e-6)，fp16 会下溢，bf16 动态范围同 fp32 更稳，且无需 GradScaler。
USE_AMP   = True
AMP_DTYPE = torch.bfloat16


def train_one_epoch(model, loader, opt, verbose=False):
    model.train(); total = 0.0; n = len(loader)
    for step, (img, hm, wh, off, pk, _) in enumerate(loader, 1):
        img, hm, wh, off, pk = img.to(device), hm.to(device), wh.to(device), off.to(device), pk.to(device)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=(USE_AMP and device.type == "cuda")):
            loss = compute_loss(model, img, hm, wh, off, pk)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
        if verbose:
            print(f"    step {step:03d}/{n}  loss={loss.item():.5f}", flush=True)
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader):
    model.eval(); total = 0.0
    for img, hm, wh, off, pk, _ in loader:
        img, hm, wh, off, pk = img.to(device), hm.to(device), wh.to(device), off.to(device), pk.to(device)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=(USE_AMP and device.type == "cuda")):
            total += compute_loss(model, img, hm, wh, off, pk).item()
    return total / max(len(loader), 1)


# ── 推理：解码 bbox ───────────────────────────────────────────────────────────
@torch.no_grad()
def predict_boxes(model, image_path):
    input_h, input_w = input_size
    hm_h, hm_w = input_h // HM_STRIDE, input_w // HM_STRIDE
    img = Image.open(image_path).convert("L").resize((input_w, input_h), Image.BILINEAR)
    x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    hm_logit, wh, off = model(x)
    hm = torch.sigmoid(hm_logit[0, 0]).cpu().numpy()
    hm_nms = exp._heatmap_nms(hm, nms_kernel)
    ys, xs = np.where(hm_nms >= HM_THRESH)
    if len(ys) == 0:
        return [], []
    scores = hm_nms[ys, xs]
    order  = np.argsort(scores)[::-1][:max_det]
    ys, xs, scores = ys[order], xs[order], scores[order]
    wh_np, off_np = wh[0].cpu().numpy(), off[0].cpu().numpy()
    boxes = []
    for yi, xi in zip(ys, xs):
        bw = float(wh_np[0, yi, xi]) * input_w
        bh = float(wh_np[1, yi, xi]) * input_h
        cx = (xi + float(np.clip(off_np[0, yi, xi], -0.5, 0.5))) / hm_w * input_w
        cy = (yi + float(np.clip(off_np[1, yi, xi], -0.5, 0.5))) / hm_h * input_h
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return boxes, [float(s) for s in scores]


@torch.no_grad()
def evaluate_bbox(model, dataset, rect_ann):
    input_h, input_w = input_size
    n_gt = n_detected = 0
    ap_tp, ap_scores = [], []
    for i in range(len(dataset)):
        meta = dataset[i][-1]
        boxes, scores = predict_boxes(model, meta["image_path"])
        orig_h, orig_w = meta["orig_size"]
        sx, sy = input_w / orig_w, input_h / orig_h
        gt_boxes = [rect_to_bbox_scaled(r, sx, sy)
                    for r in rect_ann.get(meta["image_name"], []) if r.get("label", "") == "hyperbola"]
        n_gt += len(gt_boxes)
        matched = [False] * len(gt_boxes)
        for box, sc in sorted(zip(boxes, scores), key=lambda z: -z[1]):
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if matched[j]:
                    continue
                iou = bbox_iou(box, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = best_iou >= BBOX_IOU_THR and best_j >= 0
            if is_tp:
                matched[best_j] = True
            ap_tp.append(is_tp); ap_scores.append(sc)
        n_detected += sum(matched)
    return {
        "n_gt": n_gt,
        "bbox_recall": n_detected / max(n_gt, 1),
        "bbox_mAP50":  exp.compute_ap50(ap_tp, ap_scores, n_gt),
    }


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now()
    work_dir = os.path.join(os.getcwd(), f"bbox_cnn_baseline_{now.strftime('%m%d_%H%M')}")
    exp.ensure_dir(work_dir)
    print("Using device:", device)
    print(f"BBox-CNN baseline  {len(SEEDS)}-seed  input={input_size}  lr={LR}  epochs={num_epochs}")

    image_dir = exp.data_sources[0]["image_dir"]
    hyp_json  = exp.data_sources[0]["annotation_json"]
    rect_json = os.path.join(os.path.dirname(hyp_json), RECT_JSON_NAME)
    with open(rect_json, "r", encoding="utf-8") as f:
        rect_ann = json.load(f)

    full = BBoxDataset(image_dir, hyp_json, rect_json, input_size, HM_STRIDE, HM_SIGMA)
    n_total = len(full)
    print(f"Total samples: {n_total}")

    rows = []
    for seed in SEEDS:
        exp.set_seed(seed)
        train_idx, val_idx, test_idx = exp.make_split(n_total, seed)
        print(f"\n{'='*70}\nSeed {seed}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}\n{'='*70}")

        train_loader = DataLoader(AugWrapper(Subset(full, train_idx)), batch_size=batch_size,
                                  shuffle=True, num_workers=0, collate_fn=collate)
        val_loader   = DataLoader(Subset(full, val_idx), batch_size=batch_size,
                                  shuffle=False, num_workers=0, collate_fn=collate)
        test_set     = Subset(full, test_idx)

        ckpt_dir = os.path.join(work_dir, f"seed{seed:02d}", "checkpoints")
        exp.ensure_dir(ckpt_dir)
        best_path = os.path.join(ckpt_dir, "best_model.pth")

        model = BBoxNet(in_ch=1, base_ch=32).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        best_val = float("inf")
        for epoch in range(1, num_epochs + 1):
            tr = train_one_epoch(model, train_loader, opt, verbose=(epoch == 1))
            va = validate(model, val_loader)
            print(f"  [Epoch {epoch:03d}/{num_epochs}] train={tr:.5f}  val={va:.5f}", flush=True)
            if va < best_val:
                best_val = va
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path, map_location=device))
        m = evaluate_bbox(model, test_set, rect_ann)
        print(f"  [Seed {seed}]  n_gt={m['n_gt']}  bbox_recall={m['bbox_recall']:.4f}  bbox_mAP50={m['bbox_mAP50']:.4f}")
        row = {"seed": seed, "best_val_loss": round(best_val, 6)}; row.update(m)
        rows.append(row)

    # 汇总
    keys = ["bbox_recall", "bbox_mAP50"]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    stds  = {k: float(np.std ([r[k] for r in rows])) for k in keys}
    print("\n" + "=" * 60)
    print(f"BBox-CNN baseline  {len(rows)}-seed  (IoU>={BBOX_IOU_THR})")
    print("=" * 60)
    for k in keys:
        print(f"  {k:<12} = {means[k]:.4f} ± {stds[k]:.4f}")
    print("=" * 60)

    csv_path = os.path.join(work_dir, "bbox_cnn_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
