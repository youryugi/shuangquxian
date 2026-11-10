"""
work 的方法模块：CenterNet 检测头 + stride4 显式注意力监督（双曲线带 mask 监督注意力图）。
供 visualize_attn.py / attn_multiseed.py 复用。

（这是之前验证有效的版本：with_attn 使 precision 0.55->0.83、F1 0.43->0.46。）
"""
import os
import re
import json
import importlib.util

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
exp = _load("0616-1.py", "exp0616")

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
SEED         = 0
ATTN_STRIDE  = 4
LAM_ATT      = 1.0
BBOX_IOU_THR = 0.5
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


class AttnDataset(Dataset):
    def __init__(self, input_size=(640, 640), hm_stride=8, sigma=4.3):
        self.input_h, self.input_w = input_size
        self.gh, self.gw = self.input_h // hm_stride, self.input_w // hm_stride
        self.attn_h, self.attn_w = self.input_h // ATTN_STRIDE, self.input_w // ATTN_STRIDE
        self.stride, self.sigma = hm_stride, sigma
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

        hm = np.zeros((self.gh, self.gw), np.float32)
        wh = np.zeros((2, self.gh, self.gw), np.float32)
        off = np.zeros((2, self.gh, self.gw), np.float32)
        peak = np.zeros((self.gh, self.gw), np.float32)
        for r in self.rect.get(name, []):
            if r.get("label", "") != "hyperbola":
                continue
            bw, bh = r["width"] * sx, r["height"] * sy
            cx, cy = r["x1"] * sx + bw / 2.0, r["y1"] * sy + bh / 2.0
            fx, fy = cx / self.stride, cy / self.stride
            xi = int(np.clip(round(fx), 0, self.gw - 1)); yi = int(np.clip(round(fy), 0, self.gh - 1))
            exp.render_gaussian(hm, fx, fy, self.sigma)
            wh[0, yi, xi] = bw / self.input_w; wh[1, yi, xi] = bh / self.input_h
            off[0, yi, xi] = fx - xi; off[1, yi, xi] = fy - yi
            peak[yi, xi] = 1.0

        band_full = np.zeros((self.input_h, self.input_w), np.float32)
        meta_objs = []
        for o in self.hyp.get(name, []):
            if o.get("label", "") != "hyperbola":
                continue
            mo = {"x_vertex": o["x_vertex"] * sx, "y_vertex": o["y_vertex"] * sy,
                  "width": o["width"] * sx, "height": o["height"] * sy, "thickness": o["thickness"] * sy}
            meta_objs.append(mo)
            band_full = np.maximum(band_full, exp.rasterize_hyperbola_band_mask(
                self.input_h, self.input_w, mo["x_vertex"], mo["y_vertex"], mo["width"], mo["height"], mo["thickness"]))
        band = cv2.resize(band_full, (self.attn_w, self.attn_h), interpolation=cv2.INTER_AREA)
        band = (band > 0.5).astype(np.float32)

        meta = {"image_name": name, "image_path": path, "objects": meta_objs, "orig_size": (oh, ow)}
        return (torch.from_numpy(img_np).unsqueeze(0).float(),
                torch.from_numpy(hm).unsqueeze(0).float(),
                torch.from_numpy(wh).float(),
                torch.from_numpy(off).float(),
                torch.from_numpy(peak).unsqueeze(0).float(),
                torch.from_numpy(band).unsqueeze(0).float(),
                meta)


def collate(batch):
    imgs, hms, whs, offs, pks, bands, metas = zip(*batch)
    return (torch.stack(imgs), torch.stack(hms), torch.stack(whs), torch.stack(offs),
            torch.stack(pks), torch.stack(bands), list(metas))


class AttnBBoxNet(nn.Module):
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
        _, x = self.down1(x); _, x = self.down2(x)
        f3, x = self.down3(x)
        feat = self.bottleneck(x)
        a_logit = None
        if self.use_attn:
            a_logit = self.attn_head(f3)
            gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)
            feat = feat * (1.0 + gate)   # 只增强双曲线带、不抑制背景（抑制背景实测有害，背景上下文对GPR双曲线检测有用）
        return self.heatmap_head(feat), torch.sigmoid(self.wh_head(feat)), self.offset_head(feat), a_logit


def masked_l1(pred, gt, peak):
    mask = peak.expand_as(pred); n = mask.sum()
    if n == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred * mask, gt * mask, reduction="sum") / (n / pred.shape[1] + 1e-6)


def attn_loss(a_logit, band):
    bce = F.binary_cross_entropy_with_logits(a_logit, band)
    p = torch.sigmoid(a_logit)
    dice = 1.0 - 2.0 * (p * band).sum() / (p.sum() + band.sum() + 1e-6)
    return bce + dice


def compute_loss(model, img, hm, wh, off, peak, band):
    hm_logit, wh_p, off_p, a_logit = model(img)
    loss = exp.focal_loss_heatmap(hm_logit, hm, peak) + masked_l1(wh_p, wh, peak) + masked_l1(off_p, off, peak)
    if model.use_attn:
        loss = loss + LAM_ATT * attn_loss(a_logit, band)
    return loss


def train_model(use_attn, full, train_idx, val_idx, n_epochs, work, tag):
    exp.set_seed(SEED)
    tl = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    vl = DataLoader(Subset(full, val_idx), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model = AttnBBoxNet(in_ch=1, base_ch=32, use_attn=use_attn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_epochs + 1):
        model.train()
        for img, hm, wh, off, pk, band, _ in tl:
            img, hm, wh, off, pk, band = [t.to(device) for t in (img, hm, wh, off, pk, band)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = compute_loss(model, img, hm, wh, off, pk, band)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, hm, wh, off, pk, band, _ in vl:
                img, hm, wh, off, pk, band = [t.to(device) for t in (img, hm, wh, off, pk, band)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += compute_loss(model, img, hm, wh, off, pk, band).item()
        va = tot / max(len(vl), 1)
        if va < best:
            best = va; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_epochs:
            print(f"  [{tag}] epoch {ep}/{n_epochs} val={va:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


@torch.no_grad()
def predict(model, img_tensor):
    input_h, input_w = input_size
    gh, gw = input_h // HM_STRIDE, input_w // HM_STRIDE
    hm_logit, wh_p, off_p, a_logit = model(img_tensor.to(device))
    hm = torch.sigmoid(hm_logit[0, 0]).float().cpu().numpy()
    hm_nms = exp._heatmap_nms(hm, nms_kernel)
    ys, xs = np.where(hm_nms >= HM_THRESH)
    boxes, scores = [], []
    if len(ys) > 0:
        sc = hm_nms[ys, xs]; order = np.argsort(sc)[::-1][:max_det]
        ys, xs, sc = ys[order], xs[order], sc[order]
        whp, ofp = wh_p[0].float().cpu().numpy(), off_p[0].float().cpu().numpy()
        for yi, xi in zip(ys, xs):
            bw = float(whp[0, yi, xi]) * input_w; bh = float(whp[1, yi, xi]) * input_h
            cx = (xi + float(np.clip(ofp[0, yi, xi], -0.5, 0.5))) / gw * input_w
            cy = (yi + float(np.clip(ofp[1, yi, xi], -0.5, 0.5))) / gh * input_h
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]); scores.append(float(hm_nms[yi, xi]))
    A = torch.sigmoid(a_logit[0, 0]).float().cpu().numpy() if a_logit is not None else None
    return boxes, scores, A


@torch.no_grad()
def evaluate(model, dataset, test_idx):
    TP = FP = FN = 0
    ai = au = 0.0
    for i in test_idx:
        img, _, _, _, _, band, meta = dataset[i]
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
            ai += float(np.logical_and(ab, bb).sum()); au += float(np.logical_or(ab, bb).sum())
    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
    return {"bbox_P": P, "bbox_R": R, "bbox_F1": 2 * P * R / max(P + R, 1e-9),
            "attn_band_iou": (ai / au) if au > 0 else float("nan")}


# ── 多 seed 消融（with_attn vs no_attn，70/15/15）──────────────────────────────
def main():
    import csv
    from datetime import datetime

    global SEED
    SEEDS = [0, 1, 2, 3, 4]
    work = os.path.join(os.getcwd(), f"attn_cnn_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = AttnDataset(input_size=input_size, hm_stride=HM_STRIDE, sigma=HM_SIGMA)
    n = len(full)
    print(f"[attn_cnn 多seed消融]  n_total={n}  70/15/15  seeds={SEEDS}")

    rows = []
    for seed in SEEDS:
        # 显式传 0.70/0.15（make_split 默认绑定 0.5/0.25，必须覆盖）
        tr, va, te = exp.make_split(n, seed, train_frac=0.70, val_frac=0.15)
        print(f"\n=== seed {seed}  train={len(tr)} val={len(va)} test={len(te)} ===")
        for use_attn, tag in [(True, "with_attn"), (False, "no_attn")]:
            SEED = seed   # train_model 内 set_seed 用当前 seed
            model = train_model(use_attn, full, tr, va, num_epochs, work, f"seed{seed}_{tag}")
            m = evaluate(model, full, te)
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

    with open(os.path.join(work, "attn_cnn_multiseed.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}")


if __name__ == "__main__":
    main()
