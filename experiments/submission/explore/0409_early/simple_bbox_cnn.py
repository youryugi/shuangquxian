"""
最简 baseline：朴素全卷积 CNN + bbox 检测头，from scratch 识别双曲线。

- 只用矩形标注 annotations_rect.json（bbox）；不用双曲线带 / 注意力 / 顶点参数回归。
- backbone：3 层下采样的小 CNN（base_ch=32）+ bottleneck。
- 检测头：每像素 objectness heatmap + wh + 中心 offset（CenterNet 式密集检测）。
  选它是因为这是能从零稳定训起来的最简全卷积检测方案——手写 YOLO anchor 头正样本
  太稀疏，之前两次都训不起来（F1≈0）。
- 评估：bbox 级 P/R/F1 @IoU0.5（固定阈值）+ mAP50（扫 score，不依赖阈值）。
- 多 seed 70/15/15，作为论文里最基础的 CNN baseline。
"""
import os
import re
import json
import csv
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
exp = _load("0616-1.py", "exp0616")

device     = exp.device
input_size = exp.input_size
HM_STRIDE  = exp.HM_STRIDE
HM_SIGMA   = exp.HM_SIGMA
batch_size = exp.batch_size
num_epochs = exp.num_epochs
LR         = exp.LR
HM_THRESH  = exp.HM_THRESH
nms_kernel = exp.nms_kernel
max_det    = exp.max_det
IOU_THR    = 0.5
SEEDS      = [0, 1, 2, 3, 4]
IMG_DIR    = exp.data_sources[0]["image_dir"]
HYP_JSON   = exp.data_sources[0]["annotation_json"]
RECT_JSON  = os.path.join(os.path.dirname(HYP_JSON), "annotations_rect.json")


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


class BBoxDataset(Dataset):
    """读矩形标注，生成 CenterNet GT（heatmap / wh / offset / peak）。"""
    def __init__(self, input_size=(640, 640), hm_stride=8, sigma=4.3):
        self.input_h, self.input_w = input_size
        self.gh, self.gw = self.input_h // hm_stride, self.input_w // hm_stride
        self.stride, self.sigma = hm_stride, sigma
        with open(RECT_JSON, "r", encoding="utf-8") as f:
            self.rect = json.load(f)
        # 图像列表用 annotations.json 的 keys（315 张存在的图），与 attn_cnn / HyperbolaNet 一致，
        # 保证 split 可比；bbox 标注从 rect json 取。
        with open(HYP_JSON, "r", encoding="utf-8") as f:
            self.hyp = json.load(f)
        self.names = sorted(self.hyp.keys(), key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0])

    def __len__(self):
        return len(self.names)

    def _gt_boxes(self, name, sx, sy):
        boxes = []
        for r in self.rect.get(name, []):
            if r.get("label", "") != "hyperbola":
                continue
            x1, y1 = r["x1"] * sx, r["y1"] * sy
            boxes.append([x1, y1, x1 + r["width"] * sx, y1 + r["height"] * sy])
        return boxes

    def __getitem__(self, idx):
        name = self.names[idx]
        path = os.path.join(IMG_DIR, name)
        img = Image.open(path).convert("L")
        ow, oh = img.size
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        sx, sy = self.input_w / ow, self.input_h / oh

        hm   = np.zeros((self.gh, self.gw), np.float32)
        wh   = np.zeros((2, self.gh, self.gw), np.float32)
        off  = np.zeros((2, self.gh, self.gw), np.float32)
        peak = np.zeros((self.gh, self.gw), np.float32)
        gt_boxes = self._gt_boxes(name, sx, sy)
        for (x1, y1, x2, y2) in gt_boxes:
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
            fx, fy = cx / self.stride, cy / self.stride
            xi = int(np.clip(round(fx), 0, self.gw - 1)); yi = int(np.clip(round(fy), 0, self.gh - 1))
            exp.render_gaussian(hm, fx, fy, self.sigma)
            wh[0, yi, xi] = bw / self.input_w; wh[1, yi, xi] = bh / self.input_h
            off[0, yi, xi] = fx - xi; off[1, yi, xi] = fy - yi
            peak[yi, xi] = 1.0

        meta = {"image_name": name, "image_path": path, "gt_boxes": gt_boxes}
        return (torch.from_numpy(img_np).unsqueeze(0).float(),
                torch.from_numpy(hm).unsqueeze(0).float(),
                torch.from_numpy(wh).float(),
                torch.from_numpy(off).float(),
                torch.from_numpy(peak).unsqueeze(0).float(),
                meta)


def collate(batch):
    imgs, hms, whs, offs, pks, metas = zip(*batch)
    return (torch.stack(imgs), torch.stack(hms), torch.stack(whs),
            torch.stack(offs), torch.stack(pks), list(metas))


class SimpleBBoxCNN(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.down1 = exp.DownBlock(in_ch, base_ch)
        self.down2 = exp.DownBlock(base_ch, base_ch * 2)
        self.down3 = exp.DownBlock(base_ch * 2, base_ch * 4)
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
        _, x = self.down1(x); _, x = self.down2(x); _, x = self.down3(x)
        feat = self.bottleneck(x)
        return self.heatmap_head(feat), torch.sigmoid(self.wh_head(feat)), self.offset_head(feat)


def masked_l1(pred, gt, peak):
    mask = peak.expand_as(pred); n = mask.sum()
    if n == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred * mask, gt * mask, reduction="sum") / (n / pred.shape[1] + 1e-6)


def compute_loss(model, img, hm, wh, off, peak):
    hm_logit, wh_p, off_p = model(img)
    return exp.focal_loss_heatmap(hm_logit, hm, peak) + masked_l1(wh_p, wh, peak) + masked_l1(off_p, off, peak)


def train_model(seed, full, tr, va, n_ep, work, tag):
    exp.set_seed(seed)
    tl = DataLoader(Subset(full, tr), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    vl = DataLoader(Subset(full, va), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model = SimpleBBoxCNN(in_ch=1, base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_ep + 1):
        model.train()
        for img, hm, wh, off, pk, _ in tl:
            img, hm, wh, off, pk = [t.to(device) for t in (img, hm, wh, off, pk)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = compute_loss(model, img, hm, wh, off, pk)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, hm, wh, off, pk, _ in vl:
                img, hm, wh, off, pk = [t.to(device) for t in (img, hm, wh, off, pk)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += compute_loss(model, img, hm, wh, off, pk).item()
        va_loss = tot / max(len(vl), 1)
        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} val={va_loss:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


@torch.no_grad()
def predict(model, img_tensor):
    H, W = input_size
    gh, gw = H // HM_STRIDE, W // HM_STRIDE
    hm_logit, wh_p, off_p = model(img_tensor.to(device))
    hm = torch.sigmoid(hm_logit[0, 0]).float().cpu().numpy()
    hm_nms = exp._heatmap_nms(hm, nms_kernel)
    ys, xs = np.where(hm_nms >= 1e-4)        # 收全部候选，score 留给 mAP 排序；F1 另用阈值过滤
    boxes, scores = [], []
    if len(ys) > 0:
        sc = hm_nms[ys, xs]; order = np.argsort(sc)[::-1][:max_det]
        ys, xs = ys[order], xs[order]
        whp, ofp = wh_p[0].float().cpu().numpy(), off_p[0].float().cpu().numpy()
        for yi, xi in zip(ys, xs):
            bw = float(whp[0, yi, xi]) * W; bh = float(whp[1, yi, xi]) * H
            cx = (xi + float(np.clip(ofp[0, yi, xi], -0.5, 0.5))) / gw * W
            cy = (yi + float(np.clip(ofp[1, yi, xi], -0.5, 0.5))) / gh * H
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            scores.append(float(hm_nms[yi, xi]))
    return boxes, scores


def compute_ap50(all_preds, all_gts):
    """VOC 全点插值 AP @IoU0.5（单类）。all_preds[i]=(boxes,scores), all_gts[i]=boxes。"""
    entries = []   # (score, is_tp)
    n_gt = 0
    for (boxes, scores), gts in zip(all_preds, all_gts):
        n_gt += len(gts)
        matched = [False] * len(gts)
        for b, s in sorted(zip(boxes, scores), key=lambda z: -z[1]):
            bi, bj = 0.0, -1
            for j, gb in enumerate(gts):
                if matched[j]:
                    continue
                iou = bbox_iou(b, gb)
                if iou > bi:
                    bi, bj = iou, j
            tp = bi >= IOU_THR and bj >= 0
            if tp:
                matched[bj] = True
            entries.append((s, tp))
    if n_gt == 0 or not entries:
        return 0.0
    entries.sort(key=lambda z: -z[0])
    tp = np.array([e[1] for e in entries], dtype=np.float32)
    fp = 1.0 - tp
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate(model, full, test_idx):
    TP = FP = FN = 0
    all_preds, all_gts = [], []
    for i in test_idx:
        img, _, _, _, _, meta = full[i]
        boxes, scores = predict(model, img.unsqueeze(0))
        gt = meta["gt_boxes"]
        all_preds.append((boxes, scores)); all_gts.append(gt)
        # 固定阈值下的 P/R/F1
        keep = [(b, s) for b, s in zip(boxes, scores) if s >= HM_THRESH]
        matched = [False] * len(gt)
        for b, _ in sorted(keep, key=lambda z: -z[1]):
            bi, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if matched[j]:
                    continue
                iou = bbox_iou(b, gb)
                if iou > bi:
                    bi, bj = iou, j
            if bi >= IOU_THR and bj >= 0:
                matched[bj] = True; TP += 1
            else:
                FP += 1
        FN += len(gt) - sum(matched)
    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
    return {"bbox_P": P, "bbox_R": R, "bbox_F1": 2 * P * R / max(P + R, 1e-9),
            "mAP50": compute_ap50(all_preds, all_gts)}


def main():
    work = os.path.join(os.getcwd(), f"simple_bbox_cnn_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = BBoxDataset(input_size=input_size, hm_stride=HM_STRIDE, sigma=HM_SIGMA)
    n_total = len(full)
    print(f"[简单 CNN bbox baseline]  n_total={n_total}  seeds={SEEDS}")

    rows = []
    for seed in SEEDS:
        tr, va, te = exp.make_split(n_total, seed, train_frac=0.70, val_frac=0.15)
        print(f"\n=== seed {seed}  train={len(tr)} val={len(va)} test={len(te)} ===")
        model = train_model(seed, full, tr, va, num_epochs, work, f"seed{seed}")
        m = evaluate(model, full, te)
        print(f"  seed{seed}: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} F1={m['bbox_F1']:.4f} mAP50={m['mAP50']:.4f}")
        row = {"seed": seed}; row.update(m); rows.append(row)

    keys = ["bbox_P", "bbox_R", "bbox_F1", "mAP50"]
    print("\n" + "=" * 60)
    print(f"{'metric':>14}{'mean':>12}{'std':>12}")
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"{k:>14}{np.mean(vals):>12.4f}{np.std(vals):>12.4f}")
    print("=" * 60)

    with open(os.path.join(work, "simple_bbox_cnn_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}")


if __name__ == "__main__":
    main()
