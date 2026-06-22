"""
经典 YOLO 风格 dense 检测器 + 双曲线带显式注意力监督。

吸取上次教训：YOLO 式 head 失败是正样本太稀疏(单中心 cell)。这次：
  - 正样本 = GT 框中心 cell 的 3×3 邻域（多正样本，objectness 训得起来）
  - objectness 用平衡 focal(alpha=0.5)，不压正样本
  - 每个正 cell 回归 [框中心相对 cell 偏移 dx,dy, 宽 w, 高 h]，推理 NMS 合并
注意力分支(stride4)用双曲线带 mask 监督，门控 stride8 检测特征。
带 with/no 消融 + 全部 test 可视化。
"""
import os
import re
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

device     = exp.device
input_size = exp.input_size
HM_STRIDE  = exp.HM_STRIDE
batch_size = exp.batch_size
num_epochs = exp.num_epochs
LR         = exp.LR
nms_kernel = exp.nms_kernel
max_det    = exp.max_det
SEED       = 0
ATTN_STRIDE = 4
LAM_ATT    = 1.0
LAM_BOX    = 5.0
CONF_THRESH = 0.30
NMS_IOU    = 0.50
BBOX_IOU_THR = 0.5
N_VIS      = 100000
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


class AttnYoloDataset(Dataset):
    def __init__(self, input_size=(640, 640), hm_stride=8):
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

        obj = np.zeros((1, self.gh, self.gw), np.float32)
        box = np.zeros((4, self.gh, self.gw), np.float32)
        for r in self.rect.get(name, []):
            if r.get("label", "") != "hyperbola":
                continue
            bw, bh = r["width"] * sx, r["height"] * sy
            cx, cy = r["x1"] * sx + bw / 2.0, r["y1"] * sy + bh / 2.0
            fx, fy = cx / self.stride, cy / self.stride
            xi, yi = int(fx), int(fy)
            for ddy in (-1, 0, 1):                     # 3×3 多正样本
                for ddx in (-1, 0, 1):
                    xj, yj = xi + ddx, yi + ddy
                    if 0 <= xj < self.gw and 0 <= yj < self.gh:
                        obj[0, yj, xj] = 1.0
                        box[0, yj, xj] = fx - xj       # 框中心相对该 cell 偏移（可负/>1）
                        box[1, yj, xj] = fy - yj
                        box[2, yj, xj] = bw / self.input_w
                        box[3, yj, xj] = bh / self.input_h

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
                torch.from_numpy(obj).float(),
                torch.from_numpy(box).float(),
                torch.from_numpy(band).unsqueeze(0).float(),
                meta)


def collate(batch):
    imgs, objs, boxes, bands, metas = zip(*batch)
    return torch.stack(imgs), torch.stack(objs), torch.stack(boxes), torch.stack(bands), list(metas)


class AttnYoloNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.last_a_logit = None
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
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, 5, 1))  # obj, dx, dy, w, h

    def forward(self, x):
        _, x = self.down1(x); _, x = self.down2(x)
        f3, x = self.down3(x)
        feat = self.bottleneck(x)
        if self.use_attn:
            a_logit = self.attn_head(f3)
            gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)
            feat = feat * (1.0 + gate)
            self.last_a_logit = a_logit
        else:
            self.last_a_logit = None
        out = self.det_head(feat)
        return out  # (B,5,gh,gw)


def obj_focal(logit, target, alpha=0.5, gamma=2.0):
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
    out = model(img)
    obj_logit = out[:, 0:1]
    dxdy = out[:, 1:3]                       # 偏移，无激活
    wh = torch.sigmoid(out[:, 3:5])          # 宽高 0~1
    l_obj = obj_focal(obj_logit, obj)
    pos = obj
    n = pos.sum()
    if n > 0:
        l_box = (F.l1_loss(dxdy * pos, box[:, 0:2] * pos, reduction="sum")
                 + F.l1_loss(wh * pos, box[:, 2:4] * pos, reduction="sum")) / (n * 4 + 1e-6)
    else:
        l_box = out.sum() * 0.0
    loss = l_obj + LAM_BOX * l_box
    if model.use_attn:
        loss = loss + LAM_ATT * attn_loss(model.last_a_logit, band)
    return loss


def train_model(use_attn, full, tr, va, n_ep, work, tag):
    exp.set_seed(SEED)
    tl = DataLoader(Subset(full, tr), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    vl = DataLoader(Subset(full, va), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model = AttnYoloNet(in_ch=1, base_ch=32, use_attn=use_attn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_ep + 1):
        model.train()
        for img, obj, box, band, _ in tl:
            img, obj, box, band = img.to(device), obj.to(device), box.to(device), band.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = compute_loss(model, img, obj, box, band)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, obj, box, band, _ in vl:
                img, obj, box, band = img.to(device), obj.to(device), box.to(device), band.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    tot += compute_loss(model, img, obj, box, band).item()
        va_loss = tot / max(len(vl), 1)
        if va_loss < best:
            best = va_loss; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} val={va_loss:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


@torch.no_grad()
def predict(model, img_tensor):
    input_h, input_w = input_size
    gh, gw = input_h // HM_STRIDE, input_w // HM_STRIDE
    out = model(img_tensor.to(device))
    obj = torch.sigmoid(out[0, 0]).float().cpu().numpy()
    dxdy = out[0, 1:3].float().cpu().numpy()
    wh = torch.sigmoid(out[0, 3:5]).float().cpu().numpy()
    ys, xs = np.where(obj >= CONF_THRESH)
    boxes, scores = [], []
    for yi, xi in zip(ys, xs):
        cx = (xi + dxdy[0, yi, xi]) * HM_STRIDE
        cy = (yi + dxdy[1, yi, xi]) * HM_STRIDE
        w = wh[0, yi, xi] * input_w; h = wh[1, yi, xi] * input_h
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]); scores.append(float(obj[yi, xi]))
    keep = nms(boxes, scores, NMS_IOU, max_det)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


@torch.no_grad()
def evaluate(model, dataset, test_idx):
    TP = FP = FN = 0
    ap_tp, ap_sc = [], []
    for i in test_idx:
        img = dataset[i][0]; meta = dataset[i][-1]
        boxes, scores = predict(model, img.unsqueeze(0))
        gt = [hyperbola_to_bbox(o) for o in meta["objects"]]
        matched = [False] * len(gt)
        for b, s in sorted(zip(boxes, scores), key=lambda z: -z[1]):
            bi, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if matched[j]:
                    continue
                iou = bbox_iou(b, gb)
                if iou > bi:
                    bi, bj = iou, j
            tp = bi >= BBOX_IOU_THR and bj >= 0
            if tp:
                matched[bj] = True; TP += 1
            else:
                FP += 1
            ap_tp.append(tp); ap_sc.append(s)
        FN += len(gt) - sum(matched)
    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
    return {"bbox_P": P, "bbox_R": R, "bbox_F1": 2 * P * R / max(P + R, 1e-9),
            "bbox_mAP50": exp.compute_ap50(ap_tp, ap_sc, TP + FN)}


def overlay(gray_u8, heat01):
    h, w = gray_u8.shape
    heat = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8), (w, h))
    hc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), 0.55, hc, 0.45, 0)


def visualize(model, full, te, work):
    H, W = input_size
    for k, i in enumerate(te[:N_VIS]):
        img_t = full[i][0]; meta = full[i][-1]
        boxes, scores = predict(model, img_t.unsqueeze(0))
        A = torch.sigmoid(model.last_a_logit[0, 0]).float().cpu().numpy() if model.last_a_logit is not None else None
        gray = (img_t[0].numpy() * 255).astype(np.uint8)
        panel_orig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        panel_attn = overlay(gray, A) if A is not None else panel_orig.copy()
        gt_band = np.zeros((H, W), np.float32)
        for o in meta["objects"]:
            gt_band = np.maximum(gt_band, exp.rasterize_hyperbola_band_mask(
                H, W, o["x_vertex"], o["y_vertex"], o["width"], o["height"], o["thickness"]))
        panel_gt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR); panel_gt[gt_band > 0.5] = (0, 200, 0)
        panel_pred = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for b in boxes:
            cv2.rectangle(panel_pred, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 220), 2)
        for p, t in [(panel_orig, "Original"), (panel_attn, "Attention"), (panel_pred, "Pred"), (panel_gt, "GT band")]:
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(work, f"{k:02d}_{os.path.splitext(meta['image_name'])[0]}.png"),
                    np.hstack([panel_orig, panel_attn, panel_pred, panel_gt]))


def main():
    work = os.path.join(os.getcwd(), f"attn_yolo_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    full = AttnYoloDataset(input_size=input_size, hm_stride=HM_STRIDE)
    tr, va, te = exp.make_split(len(full), SEED)
    print(f"YOLO-style + attention  train={len(tr)} val={len(va)} test={len(te)}")

    keys = ["bbox_P", "bbox_R", "bbox_F1", "bbox_mAP50"]
    results = {}
    vis_model = None
    for use_attn, tag in [(True, "with_attn"), (False, "no_attn")]:
        print(f"\n=== {tag} ===")
        model = train_model(use_attn, full, tr, va, num_epochs, work, tag)
        m = evaluate(model, full, te)
        print(f"  {tag}: " + "  ".join(f"{k}={m[k]:.4f}" for k in keys))
        results[tag] = m
        if use_attn:
            vis_model = model

    print("\n" + "=" * 56)
    print(f"{'config':>12}" + "".join(f"{k:>13}" for k in keys))
    for tag in ["with_attn", "no_attn"]:
        print(f"{tag:>12}" + "".join(f"{results[tag][k]:>13.4f}" for k in keys))
    print("=" * 56)
    visualize(vis_model, full, te, work)
    print(f"\nSaved -> {work}")


if __name__ == "__main__":
    main()
