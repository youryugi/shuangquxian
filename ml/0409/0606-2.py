import itertools
import os
import csv
from datetime import datetime
import json
import math
import random
from typing import List, Dict, Tuple

# =========================================================
# Change Notes (2026-06-06)
# Grid search over key hyperparameters (based on 0606-1.py):
#   Training params : hm_sigma  × lambda_size × lr
#   Inference params: hm_thresh (swept post-training on saved best model)
#   Total runs      : len(hm_sigma_list) × len(lambda_list) × len(lr_list)
#                     × len(thresh_list) evaluations
#   num_epochs reduced to 30 for grid-search speed.
# =========================================================

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split

# =========================================================
# 固定超参数
# =========================================================

data_sources = [
    {
        "image_dir": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhumore/Utilities",
        "annotation_json": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhumore/Utilities/annotations.json",
    },
]

SEED        = 1
input_size  = (256, 256)
HM_STRIDE   = 8          # heatmap size = 32 × 32
batch_size  = 8
num_epochs  = 100          # shortened for grid search
train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15
nms_kernel  = 3
mask_min_area = 20
max_det     = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 搜索网格
# =========================================================

hm_sigma_list  = [1.5, 2.0, 2.5]          # Gaussian sigma (heatmap pixel units)
lambda_list    = [0.5, 1.0, 2.0]           # param-loss weight
lr_list        = [5e-4, 1e-3]              # learning rate
thresh_list    = [0.10, 0.20, 0.30]        # hm_thresh (inference only, post-train)

# =========================================================
# Constants
# =========================================================

PARAM_CH = ("width", "height", "thickness")
N_PARAM  = len(PARAM_CH)


# =========================================================
# 1. 物理公式
# =========================================================

def height_from_physics(y_v_norm: float, width_norm: float, v: float) -> float:
    y_safe = max(float(y_v_norm), 1e-3)
    v_safe = max(float(v), 1e-3)
    return float(np.clip(float(width_norm) ** 2 / (2.0 * v_safe ** 2 * y_safe), 1e-4, 2.0))


def v_from_annotation(y_v_norm: float, width_norm: float, height_norm: float) -> float:
    y_safe = max(float(y_v_norm), 1e-3)
    h_safe = max(float(height_norm), 1e-3)
    w_safe = max(float(width_norm), 1e-3)
    return float(np.clip(math.sqrt(w_safe ** 2 / (2.0 * h_safe * y_safe)), 0.05, 50.0))


# =========================================================
# 2. 基础工具
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rasterize_hyperbola_band_mask(
    h: int, w: int,
    x_v: float, y_v: float,
    width: float, height: float, thickness: float,
) -> np.ndarray:
    width     = max(float(width),     2.0)
    height    = max(float(height),    1.0)
    thickness = max(float(thickness), 1.0)
    half_w    = width / 2.0
    n_pts     = max(40, int(round(width)))
    upper_pts, lower_pts = [], []
    for i in range(n_pts + 1):
        t  = i / max(n_pts, 1)
        x  = (x_v - half_w) + width * t
        dx = (x - x_v) / (half_w + 1e-6)
        yc = y_v + height * dx ** 2
        upper_pts.append((x, yc - thickness / 2.0))
        lower_pts.append((x, yc + thickness / 2.0))
    poly = np.array(upper_pts + list(reversed(lower_pts)), dtype=np.float32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask.astype(np.float32)


def overlay_mask_on_image(img_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    red   = np.array([0.0, 0.0, 255.0], dtype=np.float32)
    alpha = 0.45 * np.clip(mask, 0, 1)[..., None]
    return np.clip(img_color * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def detection_to_mask(det: Dict, input_size: Tuple[int, int]) -> np.ndarray:
    h, w = input_size
    return rasterize_hyperbola_band_mask(
        h, w,
        float(det["x_vertex"]), float(det["y_vertex"]),
        float(det["width"]),    float(det["height"]),
        float(det["thickness"]),
    )


def remove_small_mask_components(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    h, w    = mask_u8.shape
    min_eff = max(int(min_area), int(round(0.0005 * h * w)))
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for lid in range(1, n_lbl):
        if int(stats[lid, cv2.CC_STAT_AREA]) >= min_eff:
            out[lbl == lid] = 1
    return out.astype(np.float32)


# =========================================================
# 3. 编码 / 解码
# =========================================================

def encode_hyperbola_params(
    obj: Dict, orig_size: Tuple[int, int], input_size: Tuple[int, int],
) -> Dict[str, float]:
    orig_h, orig_w = orig_size
    x_v       = float(obj["x_vertex"])  / orig_w
    y_v       = float(obj["y_vertex"])  / orig_h
    width     = float(obj["width"])     / orig_w
    height    = float(obj["height"])    / orig_h
    thickness = float(obj["thickness"]) / orig_h
    return {
        "x_vertex":  float(np.clip(x_v,       0.0, 1.0)),
        "y_vertex":  float(np.clip(y_v,       0.0, 1.0)),
        "width":     float(np.clip(width,     0.0, 1.0)),
        "height":    float(np.clip(height,    0.0, 1.0)),
        "thickness": float(np.clip(thickness, 0.0, 1.0)),
    }


def decode_param_at_peak(
    hm_xi: int, hm_yi: int,
    param_map:  np.ndarray,   # [3, hm_h, hm_w]
    offset_map: np.ndarray,   # [2, hm_h, hm_w]  sub-pixel (dx, dy)
    hm_size: Tuple[int, int],
    input_size: Tuple[int, int],
    score: float,
) -> Dict[str, float]:
    hm_h, hm_w      = hm_size
    input_h, input_w = input_size

    # Sub-pixel correction: recover continuous heatmap coordinate
    dx = float(np.clip(offset_map[0, hm_yi, hm_xi], -0.5, 0.5))
    dy = float(np.clip(offset_map[1, hm_yi, hm_xi], -0.5, 0.5))
    x_v_norm = (hm_xi + dx) / hm_w
    y_v_norm = (hm_yi + dy) / hm_h

    width_norm     = float(np.clip(param_map[0, hm_yi, hm_xi], 0.0, 1.0))
    height_norm    = float(np.clip(param_map[1, hm_yi, hm_xi], 0.0, 1.0))
    thickness_norm = float(np.clip(param_map[2, hm_yi, hm_xi], 0.0, 1.0))
    x_v       = float(np.clip(x_v_norm * input_w, 0.0, input_w - 1))
    y_v       = float(np.clip(y_v_norm * input_h, 0.0, input_h - 1))
    width     = max(width_norm * input_w, 2.0)
    height    = max(height_norm * input_h, 1.0)
    thickness = max(thickness_norm * input_h, 1.0)
    return {
        "score":     float(score),
        "x_vertex":  x_v,
        "y_vertex":  y_v,
        "width":     width,
        "height":    height,
        "thickness": thickness,
    }


# =========================================================
# 4. 数据集
# =========================================================

def render_gaussian(heatmap: np.ndarray, cx: float, cy: float, sigma: float):
    hm_h, hm_w = heatmap.shape
    ys = np.arange(hm_h, dtype=np.float32)[:, None]
    xs = np.arange(hm_w, dtype=np.float32)[None, :]
    g  = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    np.maximum(heatmap, g, out=heatmap)


class HyperbolaDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_json: str,
        input_size: Tuple[int, int] = (256, 256),
        hm_stride: int = 8,
        sigma: float = 2.0,
    ):
        super().__init__()
        self.image_dir   = image_dir
        self.input_h, self.input_w = input_size
        self.hm_h      = self.input_h // hm_stride
        self.hm_w      = self.input_w // hm_stride
        self.hm_stride = hm_stride
        self.sigma     = sigma
        with open(annotation_json, "r", encoding="utf-8") as f:
            self.ann_dict = json.load(f)
        self.image_names = sorted(list(self.ann_dict.keys()))

    def __len__(self):
        return len(self.image_names)

    def _load_image(self, path: str):
        img = Image.open(path).convert("L")
        orig_w, orig_h = img.size
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0, orig_w, orig_h

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.image_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img, orig_w, orig_h = self._load_image(path)
        objs = self.ann_dict[name]

        heatmap    = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)
        param_map  = np.zeros((N_PARAM, self.hm_h, self.hm_w), dtype=np.float32)
        offset_map = np.zeros((2, self.hm_h, self.hm_w), dtype=np.float32)
        peak_mask  = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)
        meta_objs  = []

        for obj in objs:
            if obj.get("label", "") != "hyperbola":
                continue
            params = encode_hyperbola_params(obj, (orig_h, orig_w), (self.input_h, self.input_w))
            hm_cx = params["x_vertex"] * self.hm_w
            hm_cy = params["y_vertex"] * self.hm_h
            hm_xi = int(np.clip(round(hm_cx), 0, self.hm_w - 1))
            hm_yi = int(np.clip(round(hm_cy), 0, self.hm_h - 1))
            render_gaussian(heatmap, hm_cx, hm_cy, self.sigma)
            param_map[0, hm_yi, hm_xi]  = params["width"]
            param_map[1, hm_yi, hm_xi]  = params["height"]
            param_map[2, hm_yi, hm_xi]  = params["thickness"]
            offset_map[0, hm_yi, hm_xi] = hm_cx - hm_xi   # dx ∈ [-0.5, 0.5]
            offset_map[1, hm_yi, hm_xi] = hm_cy - hm_yi   # dy ∈ [-0.5, 0.5]
            peak_mask[hm_yi, hm_xi]     = 1.0
            meta_objs.append({
                "x_vertex":  float(obj["x_vertex"])  / orig_w * self.input_w,
                "y_vertex":  float(obj["y_vertex"])  / orig_h * self.input_h,
                "width":     float(obj["width"])     / orig_w * self.input_w,
                "height":    float(obj["height"])    / orig_h * self.input_h,
                "thickness": float(obj["thickness"]) / orig_h * self.input_h,
            })

        meta = {
            "image_name":   name,
            "image_path":   path,
            "objects":      meta_objs,
            "orig_size":    (orig_h, orig_w),
            "resized_size": (self.input_h, self.input_w),
        }
        return (
            torch.from_numpy(img).unsqueeze(0).float(),              # [1, H, W]
            torch.from_numpy(heatmap).unsqueeze(0).float(),          # [1, hm_h, hm_w]
            torch.from_numpy(param_map).float(),                     # [3, hm_h, hm_w]
            torch.from_numpy(offset_map).float(),                    # [2, hm_h, hm_w]
            torch.from_numpy(peak_mask).unsqueeze(0).float(),        # [1, hm_h, hm_w]
            meta,
        )


def build_dataset(sigma: float) -> Dataset:
    raw = [
        HyperbolaDataset(
            image_dir=src["image_dir"],
            annotation_json=src["annotation_json"],
            input_size=input_size,
            hm_stride=HM_STRIDE,
            sigma=sigma,
        )
        for src in data_sources
    ]
    return raw[0] if len(raw) == 1 else ConcatDataset(raw)


# =========================================================
# 5. 模型
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)


class HyperbolaNet(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.down1      = DownBlock(in_ch,       base_ch)
        self.down2      = DownBlock(base_ch,     base_ch * 2)
        self.down3      = DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid // 2, 1, 1),
        )
        self.param_head = nn.Sequential(
            nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid // 2, N_PARAM, 1),
        )
        # Sub-pixel offset head: predicts (dx, dy) ∈ [-0.5, 0.5] per heatmap cell
        self.offset_head = nn.Sequential(
            nn.Conv2d(mid, mid // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 4), nn.ReLU(inplace=True),
            nn.Conv2d(mid // 4, 2, 1),  # 2 channels: dx, dy — no activation, raw L1
        )

    def forward(self, x):
        _, x = self.down1(x)
        _, x = self.down2(x)
        _, x = self.down3(x)
        x    = self.bottleneck(x)
        hm_logit   = self.heatmap_head(x)
        raw        = self.param_head(x)
        offset_out = self.offset_head(x)   # [B, 2, hm_h, hm_w]  raw, no activation
        param_out  = torch.cat([
            torch.sigmoid(raw[:, 0:1]),    # width
            torch.sigmoid(raw[:, 1:2]),    # height
            torch.sigmoid(raw[:, 2:3]),    # thickness
        ], dim=1)
        return hm_logit, param_out, offset_out


# =========================================================
# 6. Loss
# =========================================================

def focal_loss_heatmap(
    pred_logit: torch.Tensor,
    target_hm:  torch.Tensor,
    peak_mask:  torch.Tensor,
    alpha: float = 2.0,
    beta:  float = 4.0,
) -> torch.Tensor:
    pred     = torch.sigmoid(pred_logit)
    pos_mask = peak_mask
    neg_w    = (1.0 - target_hm) ** beta
    pos_loss = pos_mask * (1.0 - pred) ** alpha * torch.log(pred.clamp(min=1e-6))
    neg_loss = neg_w * (1.0 - pos_mask) * pred ** alpha * torch.log((1.0 - pred).clamp(min=1e-6))
    n_pos    = pos_mask.sum().clamp(min=1.0)
    return -(pos_loss + neg_loss).sum() / n_pos


def masked_param_loss_spatial(
    pred_param: torch.Tensor,
    gt_param:   torch.Tensor,
    peak_mask:  torch.Tensor,
) -> torch.Tensor:
    mask = peak_mask.expand_as(pred_param)
    n    = mask.sum()
    if n == 0:
        return pred_param.sum() * 0.0
    loss = F.smooth_l1_loss(pred_param * mask, gt_param * mask, reduction="sum")
    return loss / (n / N_PARAM + 1e-6)


def masked_offset_loss(
    pred_offset: torch.Tensor,
    gt_offset:   torch.Tensor,
    peak_mask:   torch.Tensor,
) -> torch.Tensor:
    """L1 loss on (dx, dy) at GT peak locations only."""
    mask = peak_mask.expand_as(pred_offset)   # [B, 2, hm_h, hm_w]
    n    = mask.sum()
    if n == 0:
        return pred_offset.sum() * 0.0
    loss = F.l1_loss(pred_offset * mask, gt_offset * mask, reduction="sum")
    return loss / (n / 2 + 1e-6)


def hyperbola_collate_fn(batch):
    images, heatmaps, param_maps, offset_maps, peak_masks, metas = zip(*batch)
    return (
        torch.stack(images,      dim=0),
        torch.stack(heatmaps,    dim=0),
        torch.stack(param_maps,  dim=0),
        torch.stack(offset_maps, dim=0),
        torch.stack(peak_masks,  dim=0),
        list(metas),
    )


# =========================================================
# 7. 训练 / 验证
# =========================================================

def train_one_epoch(model, loader, optimizer, device, lambda_size):
    model.train()
    total = 0.0
    for images, gt_hm, gt_param, gt_offset, peak_mask, _ in loader:
        images    = images.to(device)
        gt_hm     = gt_hm.to(device)
        gt_param  = gt_param.to(device)
        gt_offset = gt_offset.to(device)
        peak_mask = peak_mask.to(device)
        pred_hm, pred_param, pred_offset = model(images)
        loss = (focal_loss_heatmap(pred_hm, gt_hm, peak_mask) +
                lambda_size * masked_param_loss_spatial(pred_param, gt_param, peak_mask) +
                masked_offset_loss(pred_offset, gt_offset, peak_mask))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device, lambda_size):
    model.eval()
    total = 0.0
    for images, gt_hm, gt_param, gt_offset, peak_mask, _ in loader:
        images    = images.to(device)
        gt_hm     = gt_hm.to(device)
        gt_param  = gt_param.to(device)
        gt_offset = gt_offset.to(device)
        peak_mask = peak_mask.to(device)
        pred_hm, pred_param, pred_offset = model(images)
        loss = (focal_loss_heatmap(pred_hm, gt_hm, peak_mask) +
                lambda_size * masked_param_loss_spatial(pred_param, gt_param, peak_mask) +
                masked_offset_loss(pred_offset, gt_offset, peak_mask))
        total += loss.item()
    return total / max(len(loader), 1)


# =========================================================
# 8. 推理
# =========================================================

def _heatmap_nms(hm: np.ndarray, kernel: int) -> np.ndarray:
    t    = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    tmax = F.max_pool2d(t, kernel, stride=1, padding=kernel // 2)
    keep = (t == tmax).squeeze(0).squeeze(0).numpy()
    return hm * keep


def predict_single_image(
    model, image_path: str, input_size: Tuple[int, int], device,
    obj_thresh: float = 0.2, nms_k: int = 3, max_det: int = 20,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    input_h, input_w = input_size
    hm_h = input_h // HM_STRIDE
    hm_w = input_w // HM_STRIDE

    img_pil     = Image.open(image_path).convert("L")
    img_resized = img_pil.resize((input_w, input_h), Image.BILINEAR)
    img_np      = np.array(img_resized, dtype=np.float32) / 255.0

    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        pred_hm_logit, pred_param, pred_offset = model(x)

    hm         = torch.sigmoid(pred_hm_logit[0, 0]).cpu().numpy()
    param_np   = pred_param[0].cpu().numpy()
    offset_np  = pred_offset[0].cpu().numpy()   # [2, hm_h, hm_w]
    hm_nms     = _heatmap_nms(hm, nms_k)

    ys, xs = np.where(hm_nms >= obj_thresh)
    if len(ys) == 0:
        return np.array(img_resized), np.zeros((input_h, input_w), np.float32), []

    scores = hm_nms[ys, xs]
    order  = np.argsort(scores)[::-1][:max_det]
    ys, xs, scores = ys[order], xs[order], scores[order]

    detections    = []
    combined_mask = np.zeros((input_h, input_w), dtype=np.float32)
    for yi, xi, score in zip(ys, xs, scores):
        det = decode_param_at_peak(
            int(xi), int(yi), param_np, offset_np,
            hm_size=(hm_h, hm_w),
            input_size=input_size,
            score=float(score),
        )
        detections.append(det)
        combined_mask = np.maximum(combined_mask, detection_to_mask(det, input_size))

    return np.array(img_resized), combined_mask, detections


# =========================================================
# 9. 评估
# =========================================================

def build_gt_mask_from_meta(meta, input_size):
    h, w    = input_size
    gt_mask = np.zeros((h, w), dtype=np.float32)
    for obj in meta["objects"]:
        gt_mask = np.maximum(gt_mask, detection_to_mask(obj, input_size))
    return gt_mask


@torch.no_grad()
def evaluate(model, dataset, device, input_size, obj_thresh):
    total_inter = total_union = total_pred = total_gt = 0.0
    per_iou = []

    for idx in range(len(dataset)):
        _, _, _, _, _, meta = dataset[idx]
        _, pred_mask, _  = predict_single_image(
            model, meta["image_path"], input_size, device,
            obj_thresh=obj_thresh, nms_k=nms_kernel, max_det=max_det,
        )
        pred_mask = remove_small_mask_components(pred_mask, min_area=mask_min_area)
        gt_mask   = build_gt_mask_from_meta(meta, input_size)

        pred_bin = pred_mask > 0
        gt_bin   = gt_mask   > 0
        inter    = float(np.logical_and(pred_bin, gt_bin).sum())
        union    = float(np.logical_or(pred_bin,  gt_bin).sum())
        total_inter += inter
        total_union += union
        total_pred  += float(pred_bin.sum())
        total_gt    += float(gt_bin.sum())
        per_iou.append(inter / max(union, 1e-6))

    pp = total_inter / max(total_pred, 1e-6)
    pr = total_inter / max(total_gt,   1e-6)
    f1 = 2 * pp * pr / max(pp + pr, 1e-6)
    return {
        "global_iou":      total_inter / max(total_union, 1e-6),
        "pixel_precision": pp,
        "pixel_recall":    pr,
        "pixel_f1":        f1,
        "mean_image_iou":  float(np.mean(per_iou)) if per_iou else 0.0,
    }


# =========================================================
# 10. 结果输出
# =========================================================

def save_results_csv(rows: List[Dict], csv_path: str):
    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_results_table(rows: List[Dict]):
    if not rows:
        return
    keys   = ["hm_sigma", "lambda_size", "lr", "hm_thresh",
               "global_iou", "pixel_f1", "pixel_precision", "pixel_recall", "mean_image_iou"]
    header = " | ".join(f"{k:>16}" for k in keys)
    sep    = "-" * len(header)
    print("\n" + "=" * len(header))
    print("Grid Search Results (sorted by pixel_f1 desc)")
    print("=" * len(header))
    print(header)
    print(sep)
    sorted_rows = sorted(rows, key=lambda r: r.get("pixel_f1", 0), reverse=True)
    for r in sorted_rows:
        vals = [
            f"{r.get('hm_sigma', ''):.2f}",
            f"{r.get('lambda_size', ''):.1f}",
            f"{r.get('lr', 0):.5f}",
            f"{r.get('hm_thresh', ''):.2f}",
            f"{r.get('global_iou', 0):.4f}",
            f"{r.get('pixel_f1', 0):.4f}",
            f"{r.get('pixel_precision', 0):.4f}",
            f"{r.get('pixel_recall', 0):.4f}",
            f"{r.get('mean_image_iou', 0):.4f}",
        ]
        print(" | ".join(f"{v:>16}" for v in vals))
    print(sep)

    best = sorted_rows[0]
    print(
        f"\n★ Best combination:\n"
        f"  hm_sigma={best['hm_sigma']}  lambda_size={best['lambda_size']}  "
        f"lr={best['lr']}  hm_thresh={best['hm_thresh']}\n"
        f"  pixel_f1={best['pixel_f1']:.4f}  global_iou={best['global_iou']:.4f}"
    )


# =========================================================
# 11. 主程序
# =========================================================

def main():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    now      = datetime.now()
    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{script_name}_{now.strftime('%m%d_%H%M')}",
    )
    ensure_dir(work_dir)
    print("Using device:", device)
    print(f"Grid: hm_sigma={hm_sigma_list}  lambda={lambda_list}  lr={lr_list}  thresh={thresh_list}")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # ── Build splits once per sigma value ────────────────────────────────
    # sigma affects dataset label generation, so datasets are rebuilt per sigma.
    # Splits use the same seed so image assignment is identical across configs.

    all_results: List[Dict] = []
    training_configs = list(itertools.product(hm_sigma_list, lambda_list, lr_list))
    n_total_train = len(training_configs)

    for run_idx, (sigma, lam, lr_val) in enumerate(training_configs, start=1):
        tag = f"sigma{sigma:.2f}_lam{lam:.1f}_lr{lr_val:.5f}".replace(".", "p")
        exp_dir  = os.path.join(work_dir, tag)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        ensure_dir(ckpt_dir)

        print(f"\n{'='*70}")
        print(f"[{run_idx}/{n_total_train}] hm_sigma={sigma}  lambda={lam}  lr={lr_val}")
        print(f"{'='*70}")

        # Build dataset with this sigma
        set_seed(SEED)
        full_dataset = build_dataset(sigma)
        n_total  = len(full_dataset)
        n_train  = int(n_total * train_ratio)
        n_val    = int(n_total * val_ratio)
        n_test   = n_total - n_train - n_val

        gen = torch.Generator().manual_seed(SEED)
        train_set, val_set, test_set = random_split(
            full_dataset, [n_train, n_val, n_test], generator=gen,
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=0, collate_fn=hyperbola_collate_fn)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=0, collate_fn=hyperbola_collate_fn)

        # Train
        model     = HyperbolaNet(in_ch=1, base_ch=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_val)
        best_val  = float("inf")
        best_path = os.path.join(ckpt_dir, "best_model.pth")

        for epoch in range(1, num_epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, lam)
            va = validate_one_epoch(model, val_loader, device, lam)
            if epoch % 5 == 0 or epoch == num_epochs:
                print(f"  [Epoch {epoch:03d}/{num_epochs}] train={tr:.5f}  val={va:.5f}")
            if va < best_val:
                best_val = va
                torch.save(model.state_dict(), best_path)

        # Load best and evaluate across all hm_thresh values
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        print(f"  Best val loss: {best_val:.5f}")
        print(f"  Evaluating hm_thresh in {thresh_list} ...")

        for thresh in thresh_list:
            metrics = evaluate(model, test_set, device, input_size, obj_thresh=thresh)
            row = {
                "hm_sigma":   sigma,
                "lambda_size": lam,
                "lr":          lr_val,
                "hm_thresh":   thresh,
                "best_val_loss": round(best_val, 6),
            }
            row.update(metrics)
            all_results.append(row)
            print(
                f"    thresh={thresh:.2f}  "
                f"IoU={metrics['global_iou']:.4f}  "
                f"F1={metrics['pixel_f1']:.4f}  "
                f"P={metrics['pixel_precision']:.4f}  "
                f"R={metrics['pixel_recall']:.4f}"
            )

    # Save and print
    csv_path = os.path.join(work_dir, "grid_search_results.csv")
    save_results_csv(all_results, csv_path)
    print_results_table(all_results)
    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
