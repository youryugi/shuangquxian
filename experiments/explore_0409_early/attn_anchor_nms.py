"""
work 的方法模块（anchor 版）：预设锚框(anchor) + 框回归 + IoU-NMS 检测头，
保留 stride4 显式注意力监督（双曲线带 mask 监督注意力图）。

与 attn_cnn.py 的区别：
  - attn_cnn.py  : CenterNet 检测头（热力图找中心点 + wh/offset），用 max-pool 找峰值去重。
  - 本文件        : RetinaNet 风格 anchor 检测头（每个网格点铺 K 个预设框，预测
                    objectness + 框回归残差），解码后用经典 IoU-NMS 去重。
  - 注意力监督     : 两者完全一致（attn_head 接在 stride4 的 f3 上，用 band mask 监督）。

骨干网络、注意力门控、band 监督全部复用 attn_cnn 的思路；仅检测头与解码不同，
方便与 attn_cnn.py 做同口径对比（evaluate 返回相同的 bbox_P/R/F1 + attn_band_iou）。
先是no attn 然后是 with attn
"""
import os
import re
import json
import math
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
HM_STRIDE    = exp.HM_STRIDE          # 检测头所在的下采样步长（stride8）
HM_SIGMA     = exp.HM_SIGMA
batch_size   = exp.batch_size
num_epochs   = 20
LR           = exp.LR
SEED         = 1

# ── stride4 注意力监督（与 attn_cnn.py 一致）──────────────────────────────────
ATTN_STRIDE  = 4
LAM_ATT      = 1.0

# ── anchor 检测头超参 ─────────────────────────────────────────────────────────
ANCHOR_SCALES = (32.0, 64.0, 128.0)   # 锚框基准边长（输入分辨率像素）
ANCHOR_RATIOS = (0.5, 1.0, 2.0)       # 宽高比 w/h（>1 更宽，适配横向双曲线包围框）
POS_IOU      = 0.5                    # IoU≥此值的 anchor 记为正样本
NEG_IOU      = 0.2                    # IoU<此值的 anchor 记为负样本（之间忽略）
LAM_REG      = 1.0                    # 框回归 smooth-L1 权重
FOCAL_ALPHA  = 0.25
FOCAL_GAMMA  = 2.0
SCORE_THRESH = 0.30                   # 推理 objectness 阈值
NMS_IOU      = 0.50                   # NMS IoU 阈值
MAX_DET      = 5
BBOX_IOU_THR = 0.5                    # 评估时判定 TP 的 IoU 阈值
_BBOX_CLAMP  = math.log(1000.0 / 16.0)  # 解码时对 dw/dh 限幅，防 exp 溢出

IMG_DIR   = exp.data_sources[0]["image_dir"]
HYP_JSON  = exp.data_sources[0]["annotation_json"]


# ── 几何工具 ──────────────────────────────────────────────────────────────────
def bbox_iou(a, b):
    """单对框 IoU（评估用，xyxy）。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def hyperbola_to_bbox(o):
    """双曲线参数 → 包围框（输入分辨率 xyxy），与 attn_cnn.py 完全一致。"""
    hw = o["width"] / 2.0
    return [o["x_vertex"] - hw, o["y_vertex"] - o["thickness"] / 2.0,
            o["x_vertex"] + hw, o["y_vertex"] + o["height"] + o["thickness"] / 2.0]


def box_iou_matrix(a, b):
    """成对 IoU，a:(M,4) b:(N,4) → (M,N)，torch。"""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


# ── anchor 生成（顺序：gy 外、gx 中、k 内，需与 head 输出 reshape 顺序一致）────
def make_anchors(gh, gw, stride, scales, ratios):
    base = []
    for s in scales:
        for r in ratios:
            base.append((s * math.sqrt(r), s / math.sqrt(r)))   # (w, h)
    base = np.asarray(base, np.float32)                          # (K, 2)
    K = len(base)
    anchors = np.zeros((gh, gw, K, 4), np.float32)
    for gy in range(gh):
        cy = (gy + 0.5) * stride
        for gx in range(gw):
            cx = (gx + 0.5) * stride
            anchors[gy, gx, :, 0] = cx - base[:, 0] / 2.0
            anchors[gy, gx, :, 1] = cy - base[:, 1] / 2.0
            anchors[gy, gx, :, 2] = cx + base[:, 0] / 2.0
            anchors[gy, gx, :, 3] = cy + base[:, 1] / 2.0
    return anchors.reshape(-1, 4), K


GRID_H = input_size[0] // HM_STRIDE
GRID_W = input_size[1] // HM_STRIDE
ANCHORS_NP, NUM_ANCHORS = make_anchors(GRID_H, GRID_W, HM_STRIDE, ANCHOR_SCALES, ANCHOR_RATIOS)
_ANCHOR_CACHE = {}


def get_anchors(dev):
    if dev not in _ANCHOR_CACHE:
        _ANCHOR_CACHE[dev] = torch.from_numpy(ANCHORS_NP).to(dev)
    return _ANCHOR_CACHE[dev]


def encode_boxes(anchors, gt):
    """框 → 回归目标 (dx,dy,dw,dh)，标准 Faster R-CNN 参数化。anchors/gt:(P,4)。"""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw / 2.0
    acy = anchors[:, 1] + ah / 2.0
    gw = gt[:, 2] - gt[:, 0]
    gh = gt[:, 3] - gt[:, 1]
    gcx = gt[:, 0] + gw / 2.0
    gcy = gt[:, 1] + gh / 2.0
    tx = (gcx - acx) / aw.clamp(min=1e-6)
    ty = (gcy - acy) / ah.clamp(min=1e-6)
    tw = torch.log((gw / aw.clamp(min=1e-6)).clamp(min=1e-6))
    th = torch.log((gh / ah.clamp(min=1e-6)).clamp(min=1e-6))
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(anchors, deltas):
    """回归残差 → 框 xyxy。anchors/deltas:(M,4)。"""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw / 2.0
    acy = anchors[:, 1] + ah / 2.0
    dx, dy = deltas[:, 0], deltas[:, 1]
    dw = deltas[:, 2].clamp(max=_BBOX_CLAMP)
    dh = deltas[:, 3].clamp(max=_BBOX_CLAMP)
    cx = acx + dx * aw
    cy = acy + dy * ah
    w = aw * torch.exp(dw)
    h = ah * torch.exp(dh)
    return torch.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=1)


def nms_numpy(boxes, scores, iou_thr):
    """经典贪心 IoU-NMS（CenterNet 不需要这步，anchor 法靠它去重）。"""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        iw = np.maximum(0.0, xx2 - xx1); ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou < iou_thr]
    return keep


# ── 数据集：返回 图像 / stride4 band / GT 框列表 / meta ───────────────────────
class AnchorDataset(Dataset):
    def __init__(self, input_size=(640, 640), hm_stride=8, sigma=4.3):
        self.input_h, self.input_w = input_size
        self.attn_h, self.attn_w = self.input_h // ATTN_STRIDE, self.input_w // ATTN_STRIDE
        self.stride, self.sigma = hm_stride, sigma
        with open(HYP_JSON, "r", encoding="utf-8") as f:
            self.hyp = json.load(f)
        self.names = sorted(self.hyp.keys(),
                            key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0])

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

        band_full = np.zeros((self.input_h, self.input_w), np.float32)
        meta_objs, gt_boxes = [], []
        for o in self.hyp.get(name, []):
            if o.get("label", "") != "hyperbola":
                continue
            mo = {"x_vertex": o["x_vertex"] * sx, "y_vertex": o["y_vertex"] * sy,
                  "width": o["width"] * sx, "height": o["height"] * sy,
                  "thickness": o["thickness"] * sy}
            meta_objs.append(mo)
            gt_boxes.append(hyperbola_to_bbox(mo))
            band_full = np.maximum(band_full, exp.rasterize_hyperbola_band_mask(
                self.input_h, self.input_w, mo["x_vertex"], mo["y_vertex"],
                mo["width"], mo["height"], mo["thickness"]))
        band = cv2.resize(band_full, (self.attn_w, self.attn_h), interpolation=cv2.INTER_AREA)
        band = (band > 0.5).astype(np.float32)

        gt_t = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4))
        meta = {"image_name": name, "image_path": path, "objects": meta_objs, "orig_size": (oh, ow)}
        return (torch.from_numpy(img_np).unsqueeze(0).float(),
                torch.from_numpy(band).unsqueeze(0).float(),
                gt_t, meta)


def collate(batch):
    imgs, bands, gts, metas = zip(*batch)
    return torch.stack(imgs), torch.stack(bands), list(gts), list(metas)


# ── 网络：复用 attn_cnn 骨干 + stride4 注意力门控；检测头换成 anchor cls/reg ──
class AnchorAttnNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, num_anchors=NUM_ANCHORS, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.K = num_anchors
        self.down1 = exp.DownBlock(in_ch, base_ch)
        self.down2 = exp.DownBlock(base_ch, base_ch * 2)
        self.down3 = exp.DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = exp.ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
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
        # objectness 先验偏置：正样本稀疏，初始化为低概率，稳定 focal 训练
        nn.init.constant_(self.cls_head[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        _, x = self.down1(x); _, x = self.down2(x)
        f3, x = self.down3(x)               # f3: stride4，注意力监督在此
        feat = self.bottleneck(x)           # stride8，检测头在此
        a_logit = None
        if self.use_attn:
            a_logit = self.attn_head(f3)
            gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)
            feat = feat * (1.0 + gate)      # 只增强双曲线带、不抑制背景（与 attn_cnn 一致）
        return self.cls_head(feat), self.reg_head(feat), a_logit


# ── 损失 ──────────────────────────────────────────────────────────────────────
def sigmoid_focal_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).sum()


def assign_targets(anchors, gt):
    """返回 labels(M,) in {-1 忽略,0 负,1 正} 与 matched_gt(M,)。"""
    M = anchors.shape[0]
    labels = torch.zeros(M, device=anchors.device)
    matched = torch.zeros(M, dtype=torch.long, device=anchors.device)
    if gt.shape[0] == 0:
        return labels, matched                       # 无 GT → 全负样本
    ious = box_iou_matrix(anchors, gt)               # (M,N)
    max_iou, matched = ious.max(dim=1)
    labels = torch.full((M,), -1.0, device=anchors.device)
    labels[max_iou < NEG_IOU] = 0.0
    labels[max_iou >= POS_IOU] = 1.0
    _, gt_best_anchor = ious.max(dim=0)              # 每个 GT 至少配一个正 anchor
    labels[gt_best_anchor] = 1.0
    return labels, matched


def attn_loss(a_logit, band):
    bce = F.binary_cross_entropy_with_logits(a_logit, band)
    p = torch.sigmoid(a_logit)
    dice = 1.0 - 2.0 * (p * band).sum() / (p.sum() + band.sum() + 1e-6)
    return bce + dice


def compute_loss(model, img, gt_list, band, anchors):
    cls, reg, a_logit = model(img)
    B = img.shape[0]
    cls_flat = cls.permute(0, 2, 3, 1).reshape(B, -1)          # (B, M)
    reg_flat = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)       # (B, M, 4)

    cls_loss = reg_flat.sum() * 0.0
    reg_loss = reg_flat.sum() * 0.0
    n_pos_total = 0
    for b in range(B):
        gt = gt_list[b].to(img.device)
        labels, matched = assign_targets(anchors, gt)
        valid = labels >= 0
        cls_loss = cls_loss + sigmoid_focal_loss(cls_flat[b][valid], labels[valid].clamp(min=0))
        pos = labels == 1
        n_pos = int(pos.sum())
        n_pos_total += n_pos
        if n_pos > 0:
            tgt = encode_boxes(anchors[pos], gt[matched[pos]])
            reg_loss = reg_loss + F.smooth_l1_loss(reg_flat[b][pos], tgt, reduction="sum")
    norm = max(n_pos_total, 1)
    loss = cls_loss / norm + LAM_REG * reg_loss / norm
    if model.use_attn:
        loss = loss + LAM_ATT * attn_loss(a_logit, band)
    return loss


# ── 训练 ──────────────────────────────────────────────────────────────────────
def train_model(use_attn, full, train_idx, val_idx, n_epochs, work, tag):
    exp.set_seed(SEED)
    anchors = get_anchors(device)
    tl = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True,
                    num_workers=0, collate_fn=collate)
    vl = DataLoader(Subset(full, val_idx), batch_size=batch_size, shuffle=False,
                    num_workers=0, collate_fn=collate)
    model = AnchorAttnNet(in_ch=1, base_ch=32, use_attn=use_attn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, bp = float("inf"), os.path.join(work, f"{tag}_best.pth")
    for ep in range(1, n_epochs + 1):
        model.train()
        for img, band, gts, _ in tl:
            img, band = img.to(device), band.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                loss = compute_loss(model, img, gts, band, anchors)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, band, gts, _ in vl:
                img, band = img.to(device), band.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    tot += compute_loss(model, img, gts, band, anchors).item()
        va = tot / max(len(vl), 1)
        if va < best:
            best = va; torch.save(model.state_dict(), bp)
        if ep % 20 == 0 or ep == n_epochs:
            print(f"  [{tag}] epoch {ep}/{n_epochs} val={va:.4f}", flush=True)
    model.load_state_dict(torch.load(bp, map_location=device)); model.eval()
    return model


# ── 推理：解码 anchor + NMS ───────────────────────────────────────────────────
@torch.no_grad()
def predict(model, img_tensor):
    anchors = get_anchors(device)
    cls, reg, a_logit = model(img_tensor.to(device))
    scores = torch.sigmoid(cls.permute(0, 2, 3, 1).reshape(-1)).float()   # (M,)
    deltas = reg.permute(0, 2, 3, 1).reshape(-1, 4).float()               # (M,4)
    keep_mask = scores >= SCORE_THRESH
    boxes, out_scores = [], []
    if keep_mask.any():
        sc = scores[keep_mask]
        bx = decode_boxes(anchors[keep_mask], deltas[keep_mask])
        bx[:, 0::2] = bx[:, 0::2].clamp(0, input_size[1] - 1)
        bx[:, 1::2] = bx[:, 1::2].clamp(0, input_size[0] - 1)
        bx_np = bx.cpu().numpy(); sc_np = sc.cpu().numpy()
        keep = nms_numpy(bx_np, sc_np, NMS_IOU)[:MAX_DET]
        boxes = [bx_np[i].tolist() for i in keep]
        out_scores = [float(sc_np[i]) for i in keep]
    A = torch.sigmoid(a_logit[0, 0]).float().cpu().numpy() if a_logit is not None else None
    return boxes, out_scores, A


# ── 评估：与 attn_cnn.py 同口径（bbox P/R/F1 + 注意力带 IoU）──────────────────
@torch.no_grad()
def evaluate(model, dataset, test_idx):
    TP = FP = FN = 0
    ai = au = 0.0
    for i in test_idx:
        img, band, _, meta = dataset[i]
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


# ── 自检 / 同口径对比 with_attn vs no_attn ────────────────────────────────────
def main():
    exp.set_seed(SEED)
    full = AnchorDataset(input_size=input_size, hm_stride=HM_STRIDE, sigma=HM_SIGMA)
    n = len(full)
    train_idx, val_idx, test_idx = exp.make_split(n, SEED, train_frac=0.70, val_frac=0.15)
    work = os.path.join(os.getcwd(), "anchor_nms_out"); os.makedirs(work, exist_ok=True)
    print(f"device={device}  total={n}  anchors/grid={NUM_ANCHORS}  "
          f"total_anchors={ANCHORS_NP.shape[0]}", flush=True)
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
    rows = []
    for use_attn in (False, True):
        tag = "with_attn" if use_attn else "no_attn"
        model = train_model(use_attn, full, train_idx, val_idx, num_epochs, work, tag)
        m = evaluate(model, full, test_idx)
        print(f"[{tag}] P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
              f"F1={m['bbox_F1']:.4f} attn_band_iou={m['attn_band_iou']:.4f}", flush=True)
        row = {"config": tag}; row.update(m); rows.append(row)

    import csv
    csv_path = os.path.join(work, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Saved metrics -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
