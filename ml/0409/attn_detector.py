"""
YOLO 式检测头 + 显式注意力监督的双曲线检测器。

- 检测头：YOLO 风格 dense grid（每个 cell 预测 objectness + 框 cx/cy/w/h），不是 CenterNet 稀疏 heatmap。
- 注意力：stride4 注意力图 A，用双曲线带 mask 强监督（LAM_ATT 加大），逼注意力落到双曲线带上而非背景。
- 两套标注：矩形(annotations_rect.json)监督检测；双曲线带(annotations.json rasterize)监督注意力。
- 消融开关 USE_ATTN：对比有/无注意力监督。
"""
import os
import re
import csv
import json
import importlib.util
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
exp = _load("0616-1.py", "exp0616")

device       = exp.device
input_size   = exp.input_size
HM_STRIDE    = exp.HM_STRIDE          # 检测 grid 步长 (stride8 -> 80x80)
batch_size   = exp.batch_size
num_epochs   = exp.num_epochs
LR           = exp.LR
nms_kernel   = exp.nms_kernel
max_det      = exp.max_det
SEED         = 0
ATTN_STRIDE  = 4                      # 注意力图 stride4 (160x160)，精细刻画薄带
LAM_BOX      = 5.0                    # 框回归权重
LAM_ATT      = 5.0                    # 注意力监督权重（加大，强制注意力落到带上）
CONF_THRESH  = 0.30                   # objectness 检出阈值
NMS_IOU      = 0.50
BBOX_IOU_THR = 0.50
IMG_DIR   = exp.data_sources[0]["image_dir"]
HYP_JSON  = exp.data_sources[0]["annotation_json"]
RECT_JSON = os.path.join(os.path.dirname(HYP_JSON), "annotations_rect.json")


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def hyperbola_to_bbox(o):
    hw = o["width"] / 2.0
    return [o["x_vertex"] - hw, o["y_vertex"] - o["thickness"] / 2.0,
            o["x_vertex"] + hw, o["y_vertex"] + o["height"] + o["thickness"] / 2.0]


def nms(boxes, scores, iou_thr=0.5, topk=5):
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = []
    while idx and len(keep) < topk:
        i = idx.pop(0); keep.append(i)
        idx = [j for j in idx if bbox_iou(boxes[i], boxes[j]) < iou_thr]
    return keep


# ── 数据集：YOLO 式检测 target + 双曲线带注意力 target ────────────────────────
class AttnDataset(Dataset):
    def __init__(self, input_size=(640, 640), hm_stride=8, **_):
        self.input_h, self.input_w = input_size
        self.gh, self.gw = self.input_h // hm_stride, self.input_w // hm_stride
        self.attn_h, self.attn_w = self.input_h // ATTN_STRIDE, self.input_w // ATTN_STRIDE
        self.stride = hm_stride
        with open(HYP_JSON, "r", encoding="utf-8") as f:
            self.hyp = json.load(f)
        with open(RECT_JSON, "r", encoding="utf-8") as f:
            self.rect = json.load(f)
        self.names = sorted(self.hyp.keys(), key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = os.path.join(IMG_DIR, name)
        img = Image.open(path).convert("L")
        ow, oh = img.size
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        sx, sy = self.input_w / ow, self.input_h / oh

        # YOLO 式检测 target：obj(1,gh,gw) + box(4,gh,gw)=[cx_off,cy_off,w_norm,h_norm]
        obj = np.zeros((1, self.gh, self.gw), np.float32)
        box = np.zeros((4, self.gh, self.gw), np.float32)
        for r in self.rect.get(name, []):
            if r.get("label", "") != "hyperbola":
                continue
            bw, bh = r["width"] * sx, r["height"] * sy
            cx, cy = r["x1"] * sx + bw / 2.0, r["y1"] * sy + bh / 2.0
            gx, gy = cx / self.stride, cy / self.stride
            xi = int(np.clip(int(gx), 0, self.gw - 1)); yi = int(np.clip(int(gy), 0, self.gh - 1))
            obj[0, yi, xi] = 1.0
            box[0, yi, xi] = gx - xi
            box[1, yi, xi] = gy - yi
            box[2, yi, xi] = bw / self.input_w
            box[3, yi, xi] = bh / self.input_h

        # 双曲线带注意力 target（stride4）
        band_full = np.zeros((self.input_h, self.input_w), np.float32)
        meta_objs = []
        for o in self.hyp.get(name, []):
            if o.get("label", "") != "hyperbola":
                continue
            mo = {"x_vertex": o["x_vertex"] * sx, "y_vertex": o["y_vertex"] * sy,
                  "width": o["width"] * sx, "height": o["height"] * sy, "thickness": o["thickness"] * sy}
            meta_objs.append(mo)
            band_full = np.maximum(band_full, exp.rasterize_hyperbola_band_mask(
                self.input_h, self.input_w, mo["x_vertex"], mo["y_vertex"],
                mo["width"], mo["height"], mo["thickness"]))
        band = cv2.resize(band_full, (self.attn_w, self.attn_h), interpolation=cv2.INTER_AREA)
        band = (band > 0.5).astype(np.float32)

        meta = {"image_name": name, "image_path": path, "objects": meta_objs, "orig_size": (oh, ow)}
        return (torch.from_numpy(img_np).unsqueeze(0).float(),
                torch.from_numpy(obj).float(),
                torch.from_numpy(box).float(),
                torch.from_numpy(band).unsqueeze(0).float(),
                meta)


def collate(batch):
    imgs, objs, boxes, bands, metas = zip(*batch)
    return torch.stack(imgs), torch.stack(objs), torch.stack(boxes), torch.stack(bands), list(metas)


# ── 模型：backbone + 注意力分支(带监督) + YOLO 式检测头 ───────────────────────
class AttnYoloNet(nn.Module):
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
                nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 2, 1, 1))
        self.det_head = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid // 2, 5, 1))   # obj, cx_off, cy_off, w, h

    def forward(self, x):
        _, x = self.down1(x); _, x = self.down2(x)
        f3, x = self.down3(x)
        feat = self.bottleneck(x)
        a_logit = None
        if self.use_attn:
            a_logit = self.attn_head(f3)
            gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)
            feat = feat * (1.0 + gate)
        return self.det_head(feat), a_logit


def obj_focal(logit, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - pt).pow(gamma) * ce).mean()


def attn_loss(a_logit, band):
    bce = F.binary_cross_entropy_with_logits(a_logit, band)
    p = torch.sigmoid(a_logit)
    dice = 1.0 - 2.0 * (p * band).sum() / (p.sum() + band.sum() + 1e-6)
    return bce + dice


def compute_loss(model, img, obj, box, band):
    det, a_logit = model(img)
    l_obj = obj_focal(det[:, 0:1], obj)
    box_pred = torch.sigmoid(det[:, 1:5])
    n = obj.sum()
    l_box = (F.l1_loss(box_pred * obj, box * obj, reduction="sum") / (n * 4 + 1e-6)) if n > 0 else box_pred.sum() * 0.0
    loss = l_obj + LAM_BOX * l_box
    if model.use_attn:
        loss = loss + LAM_ATT * attn_loss(a_logit, band)
    return loss


def train_one_epoch(model, loader, opt):
    model.train(); tot = 0.0
    for img, obj, box, band, _ in loader:
        img, obj, box, band = img.to(device), obj.to(device), box.to(device), band.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = compute_loss(model, img, obj, box, band)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    return tot / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader):
    model.eval(); tot = 0.0
    for img, obj, box, band, _ in loader:
        img, obj, box, band = img.to(device), obj.to(device), box.to(device), band.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            tot += compute_loss(model, img, obj, box, band).item()
    return tot / max(len(loader), 1)


@torch.no_grad()
def predict(model, img_tensor):
    input_h, input_w = input_size
    det, a_logit = model(img_tensor.to(device))
    obj = torch.sigmoid(det[0, 0]).float().cpu().numpy()
    bp = torch.sigmoid(det[0, 1:5]).float().cpu().numpy()
    ys, xs = np.where(obj >= CONF_THRESH)
    boxes, scores = [], []
    for yi, xi in zip(ys, xs):
        cxo, cyo, wn, hn = bp[:, yi, xi]
        cx = (xi + cxo) * HM_STRIDE; cy = (yi + cyo) * HM_STRIDE
        w = wn * input_w; h = hn * input_h
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]); scores.append(float(obj[yi, xi]))
    keep = nms(boxes, scores, NMS_IOU, max_det)
    boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
    A = torch.sigmoid(a_logit[0, 0]).float().cpu().numpy() if a_logit is not None else None
    return boxes, scores, A


@torch.no_grad()
def evaluate(model, dataset, test_idx):
    TP = FP = FN = 0
    attn_inter = attn_union = 0.0
    for i in test_idx:
        img, _, _, band, meta = dataset[i]
        boxes, scores, A = predict(model, img.unsqueeze(0))
        gt = [hyperbola_to_bbox(o) for o in meta["objects"]]
        matched = [False] * len(gt)
        for b, _ in sorted(zip(boxes, scores), key=lambda z: -z[1]):
            bi, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if matched[j]:
                    continue
                iou = bbox_iou(b, gb)
                if iou > bi:
                    bi, bj = iou, j
            if bi >= BBOX_IOU_THR and bj >= 0:
                matched[bj] = True; TP += 1
            else:
                FP += 1
        FN += len(gt) - sum(matched)
        if A is not None:
            ab = A > 0.5; bb = band[0].numpy() > 0.5
            attn_inter += float(np.logical_and(ab, bb).sum())
            attn_union += float(np.logical_or(ab, bb).sum())
    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
    return {"bbox_P": P, "bbox_R": R, "bbox_F1": 2 * P * R / max(P + R, 1e-9),
            "attn_band_iou": (attn_inter / attn_union) if attn_union > 0 else float("nan")}


def run(use_attn, full, train_idx, val_idx, test_idx, work, tag):
    exp.set_seed(SEED)
    tl = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    vl = DataLoader(Subset(full, val_idx), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model = AttnYoloNet(in_ch=1, base_ch=32, use_attn=use_attn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, num_epochs + 1):
        train_one_epoch(model, tl, opt)
        va = validate(model, vl)
        if va < best:
            best = va; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == num_epochs:
            print(f"  [{tag}] epoch {ep}/{num_epochs} val={va:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return evaluate(model, full, test_idx)


def main():
    now = datetime.now()
    work = os.path.join(os.getcwd(), f"attn_yolo_{now.strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = AttnDataset(input_size=input_size, hm_stride=HM_STRIDE)
    n_total = len(full)
    train_idx, val_idx, test_idx = exp.make_split(n_total, SEED)
    print(f"n_total={n_total}  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}  "
          f"LAM_ATT={LAM_ATT} LAM_BOX={LAM_BOX}")

    rows = []
    for use_attn, tag in [(True, "with_attn"), (False, "no_attn")]:
        print(f"\n=== {tag} (use_attn={use_attn}) ===")
        m = run(use_attn, full, train_idx, val_idx, test_idx, work, tag)
        print(f"  {tag}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        row = {"config": tag}; row.update(m); rows.append(row)

    print("\n" + "=" * 60)
    print(f"{'config':>12}{'bbox_P':>10}{'bbox_R':>10}{'bbox_F1':>10}{'attn_iou':>10}")
    for r in rows:
        print(f"{r['config']:>12}{r['bbox_P']:>10.4f}{r['bbox_R']:>10.4f}{r['bbox_F1']:>10.4f}{r['attn_band_iou']:>10.4f}")
    with open(os.path.join(work, "attn_ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}")


if __name__ == "__main__":
    main()
