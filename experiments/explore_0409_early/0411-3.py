import os
import csv
from datetime import datetime
import json
import math
import random
from typing import List, Dict, Tuple

# =========================================================
# Change Notes (2026-04-11)
# 1) Upgraded mask->hyperbola regression in decode stage:
#    - Extract centerline from connected component columns.
#    - Fit weighted quadratic curve for robust vertex/shape estimation.
#    - Fuse geometric estimates with network size regression.
# 2) Added configurable heatmap thresholding:
#    - fixed mode: use user-defined hm_thresh.
#    - adaptive mode: per-image quantile threshold with min/max clamp.
# =========================================================
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split

#输出的是mask，但是没有回归成漂亮的双曲线
#增加回归成双曲线
#可惜目前只是val的
#加入多一个数据集并且划分train val test
#调整评价的方式 改成mask和 标签的重叠？ 要记得mask要是去除小地方的 
#最后可视化双曲线的时候，双曲线的参数也要标记出来
# =========================================================
# 1. 一些基础工具
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], sigma: float):
    """
    在 heatmap 上画一个二维高斯
    heatmap: [H, W]
    center: (x, y)
    """
    x0, y0 = center
    H, W = heatmap.shape

    radius = int(3 * sigma)
    left = max(0, x0 - radius)
    right = min(W - 1, x0 + radius)
    top = max(0, y0 - radius)
    bottom = min(H - 1, y0 + radius)

    if left > right or top > bottom:
        return

    xs = np.arange(left, right + 1)
    ys = np.arange(top, bottom + 1)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')

    gaussian = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma * sigma))
    heatmap[top:bottom + 1, left:right + 1] = np.maximum(
        heatmap[top:bottom + 1, left:right + 1], gaussian
    )


def rasterize_hyperbola_band_mask(
    h: int,
    w: int,
    x_v: float,
    y_v: float,
    width: float,
    height: float,
    thickness: float,
) -> np.ndarray:
    """Rasterize a hyperbola-like band (same geometry as visualization) into a binary mask."""
    width = max(float(width), 2.0)
    height = max(float(height), 1.0)
    thickness = max(float(thickness), 1.0)

    half_w = width / 2.0
    x_left = x_v - half_w
    x_right = x_v + half_w

    upper_pts = []
    lower_pts = []
    n_points = max(40, int(round(width)))

    for i in range(n_points + 1):
        t = i / max(n_points, 1)
        x = x_left + (x_right - x_left) * t
        dx = (x - x_v) / (half_w + 1e-6)
        y_center = y_v + height * (dx ** 2)

        y_up = y_center - thickness / 2.0
        y_dn = y_center + thickness / 2.0

        upper_pts.append((x, y_up))
        lower_pts.append((x, y_dn))

    poly = upper_pts + list(reversed(lower_pts))
    poly_np = np.array(poly, dtype=np.float32)
    poly_np[:, 0] = np.clip(poly_np[:, 0], 0, w - 1)
    poly_np[:, 1] = np.clip(poly_np[:, 1], 0, h - 1)
    poly_int = np.round(poly_np).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 1)
    return mask.astype(np.float32)


def overlay_heatmap_on_image(img_gray: np.ndarray, heatmap: np.ndarray):
    """
    img_gray: [H, W], uint8
    heatmap: [H, W], float in [0,1]
    """
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    hm = np.clip(heatmap, 0, 1)
    hm = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_color, 0.6, hm_color, 0.4, 0)
    return overlay


def resolve_heatmap_threshold(
    pred_hm: np.ndarray,
    hm_thresh: float,
    hm_thresh_mode: str = "fixed",
    hm_thresh_quantile: float = 0.85,
    hm_thresh_min: float = 0.20,
    hm_thresh_max: float = 0.70,
) -> float:
    """Resolve threshold for binarizing predicted heatmap."""
    if hm_thresh_mode == "adaptive":
        q = float(np.clip(hm_thresh_quantile, 0.01, 0.99))
        thr = float(np.quantile(pred_hm, q))
        thr = float(np.clip(thr, hm_thresh_min, hm_thresh_max))
        return thr

    # default/fallback to fixed threshold
    return float(np.clip(hm_thresh, 0.01, 0.99))


def decode_mask_to_hyperbola_detections(
    pred_mask: np.ndarray,
    pred_hm: np.ndarray,
    pred_size: np.ndarray,
    input_size: Tuple[int, int],
    normalize_param: bool = True,
    min_area: int = 20,
    max_det: int = 20,
    min_component_score: float = 0.20,
    merge_detections: bool = True,
):
    """Convert a predicted curve mask back into parametric hyperbola detections."""

    def robust_median(values: np.ndarray, default: float) -> float:
        if values.size == 0:
            return float(default)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float(default)
        return float(np.median(values))

    def fit_quadratic_centerline(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray):
        """Fit y = ax^2 + bx + c from centerline samples. Return (ok, a, b, c)."""
        if len(xs) < 6:
            return False, 0.0, 0.0, 0.0

        # Use centered x for numerical stability.
        x0 = float(np.mean(xs))
        x_shift = xs - x0

        if weights is None or len(weights) != len(xs):
            w = np.ones_like(xs, dtype=np.float32)
        else:
            w = np.clip(weights.astype(np.float32), 1e-3, None)

        try:
            coef = np.polyfit(x_shift, ys, deg=2, w=w)
        except Exception:
            return False, 0.0, 0.0, 0.0

        a_s, b_s, c_s = [float(v) for v in coef]

        # Convert back to original x coordinates:
        # y = a_s(x-x0)^2 + b_s(x-x0) + c_s
        #   = a x^2 + b x + c
        a = a_s
        b = -2.0 * a_s * x0 + b_s
        c = a_s * x0 * x0 - b_s * x0 + c_s
        return True, a, b, c

    def maybe_merge_detections(dets: List[Dict]) -> List[Dict]:
        """Merge disconnected fragments that likely belong to the same hyperbola."""
        if not dets:
            return dets

        dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
        merged = []

        for det in dets_sorted:
            merged_to_existing = False
            for m in merged:
                w_avg = 0.5 * (det["width"] + m["width"])
                t_avg = 0.5 * (det["thickness"] + m["thickness"])
                h_avg = max(0.5 * (det["height"] + m["height"]), 1.0)

                # Parameter consistency gates.
                x_close = abs(det["x_vertex"] - m["x_vertex"]) <= max(0.20 * w_avg, 10.0)
                y_close = abs(det["y_vertex"] - m["y_vertex"]) <= max(2.0 * t_avg, 6.0)
                h_close = abs(det["height"] - m["height"]) / h_avg <= 0.6

                # Range/gap gate on x span.
                d_left = det["x_vertex"] - det["width"] / 2.0
                d_right = det["x_vertex"] + det["width"] / 2.0
                m_left = m["x_vertex"] - m["width"] / 2.0
                m_right = m["x_vertex"] + m["width"] / 2.0
                x_gap = max(max(d_left, m_left) - min(d_right, m_right), 0.0)
                gap_ok = x_gap <= max(0.20 * w_avg, 8.0)

                if x_close and y_close and h_close and gap_ok:
                    # Confidence-weighted fusion.
                    w1 = float(max(m["score"], 1e-3))
                    w2 = float(max(det["score"], 1e-3))
                    s = w1 + w2

                    m["x_vertex"] = (m["x_vertex"] * w1 + det["x_vertex"] * w2) / s
                    m["y_vertex"] = (m["y_vertex"] * w1 + det["y_vertex"] * w2) / s
                    m["width"] = (m["width"] * w1 + det["width"] * w2) / s
                    m["height"] = (m["height"] * w1 + det["height"] * w2) / s
                    m["thickness"] = (m["thickness"] * w1 + det["thickness"] * w2) / s
                    m["score"] = max(m["score"], det["score"])
                    merged_to_existing = True
                    break

            if not merged_to_existing:
                merged.append({k: float(v) for k, v in det.items()})

        return merged

    input_h, input_w = input_size
    mask_u8 = (pred_mask > 0).astype(np.uint8)

    # Connect nearby broken segments while removing tiny holes.
    close_kw = max(3, int(round(input_w * 0.02)))
    if close_kw % 2 == 0:
        close_kw += 1
    close_kh = 3
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kw, close_kh))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    # Remove tiny islands that should not be forced into a hyperbola.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    detections = []
    min_area_eff = max(min_area, int(round(0.0005 * input_h * input_w)))
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area_eff:
            continue

        comp_mask = labels == label_id
        ys, xs = np.where(comp_mask)
        if len(xs) == 0:
            continue

        raw_w = robust_median(pred_size[0][comp_mask], default=0.0)
        raw_h = robust_median(pred_size[1][comp_mask], default=0.0)
        raw_t = robust_median(pred_size[2][comp_mask], default=0.0)

        if normalize_param:
            width = abs(raw_w) * input_w
            height = abs(raw_h) * input_h
            thickness = abs(raw_t) * input_h
        else:
            width = abs(raw_w)
            height = abs(raw_h)
            thickness = abs(raw_t)

        width = max(width, 2.0)
        height = max(height, 1.0)
        thickness = max(thickness, 1.0)

        # Build centerline samples from each x-column of the connected component.
        x_min = int(xs.min())
        x_max = int(xs.max())
        center_xs = []
        center_ys = []
        center_ws = []
        local_thickness = []

        for x_col in range(x_min, x_max + 1):
            y_col = ys[xs == x_col]
            if y_col.size == 0:
                continue

            y_top = float(np.min(y_col))
            y_bottom = float(np.max(y_col))
            y_center = 0.5 * (y_top + y_bottom)
            t_col = max(y_bottom - y_top + 1.0, 1.0)

            center_xs.append(float(x_col))
            center_ys.append(float(y_center))
            local_thickness.append(float(t_col))

            # Column confidence: combine heatmap confidence and vertical support.
            hm_vals = pred_hm[y_col, np.full_like(y_col, x_col)]
            col_conf = float(np.mean(hm_vals)) if hm_vals.size > 0 else 0.0
            center_ws.append(max(col_conf * np.sqrt(t_col), 1e-3))

        center_xs = np.asarray(center_xs, dtype=np.float32)
        center_ys = np.asarray(center_ys, dtype=np.float32)
        center_ws = np.asarray(center_ws, dtype=np.float32)

        # Robust geometry priors from the component itself.
        geo_width = float(x_max - x_min + 1)
        geo_thickness = robust_median(np.asarray(local_thickness, dtype=np.float32), default=thickness)

        ok_fit, a, b, c = fit_quadratic_centerline(center_xs, center_ys, center_ws)

        if ok_fit and a > 1e-6:
            x_vertex_fit = float(np.clip(-b / (2.0 * a), 0, input_w - 1))
            y_vertex_fit = float(a * x_vertex_fit * x_vertex_fit + b * x_vertex_fit + c)

            # Use the visible x-span to infer curve height from fitted curvature.
            x_span = max(float(center_xs.max() - center_xs.min()), 2.0)
            height_fit = float(max(a * (x_span / 2.0) ** 2, 1.0))
        else:
            # Fallback to previous heuristic when fitting is unstable.
            top_y = ys.min()
            top_band = ys <= top_y + max(1, int(round(thickness)))
            if np.any(top_band):
                x_vertex_fit = float(np.median(xs[top_band]))
            else:
                x_vertex_fit = float(np.median(xs))
            y_vertex_fit = float(top_y + thickness / 2.0)
            height_fit = float(height)

        # Fuse network regression with geometry to improve robustness.
        width = max(0.6 * width + 0.4 * geo_width, 2.0)
        thickness = max(0.6 * thickness + 0.4 * geo_thickness, 1.0)
        height = max(0.6 * height + 0.4 * height_fit, 1.0)

        x_vertex = float(np.clip(x_vertex_fit, 0, input_w - 1))
        y_vertex = float(np.clip(y_vertex_fit - 0.5 * thickness, 0, input_h - 1))
        score = float(np.mean(pred_hm[comp_mask]))

        # Drop weak/noisy components: tiny x-span or very low confidence.
        x_span = float(x_max - x_min + 1)
        if score < min_component_score:
            continue
        if x_span < max(6.0, 1.5 * thickness):
            continue

        detections.append({
            "score": score,
            "x_vertex": x_vertex,
            "y_vertex": y_vertex,
            "width": float(width),
            "height": float(height),
            "thickness": float(thickness),
        })

    if merge_detections:
        detections = maybe_merge_detections(detections)

    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections[:max_det]


def local_peak_extract(heatmap: torch.Tensor, thresh: float = 0.3, max_det: int = 50):
    """
    从 heatmap 中找局部峰值
    heatmap: [H, W] torch tensor
    return: list of (score, x, y)
    """
    H, W = heatmap.shape
    hm = heatmap.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    pooled = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hm == pooled) & (hm > thresh)
    ys, xs = torch.where(keep[0, 0])

    results = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        score = heatmap[y, x].item()
        results.append((score, x, y))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:max_det]


# =========================================================
# 2. 数据集
# =========================================================

class HyperbolaDataset(Dataset):
    """
    输入：
      - 图片文件夹
      - JSON 标注文件

    输出：
      - image tensor: [1, H, W]
            - heatmap: [1, H, W]  -> 整体双曲线区域监督
      - size map: [3, H, W]  -> width, height, thickness
            - reg mask: [1, H, W]  -> 双曲线区域为1，其余为0
      - meta
    """

    def __init__(
        self,
        image_dir: str,
        annotation_json: str,
        input_size: Tuple[int, int] = (256, 256),
        sigma: float = 3.0,
        normalize_param: bool = True,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_json = annotation_json
        self.input_h, self.input_w = input_size
        self.sigma = sigma
        self.normalize_param = normalize_param

        with open(annotation_json, "r", encoding="utf-8") as f:
            self.ann_dict = json.load(f)

        self.image_names = sorted(list(self.ann_dict.keys()))

    def __len__(self):
        return len(self.image_names)

    def _load_image(self, image_path: str):
        img = Image.open(image_path).convert("L")
        orig_w, orig_h = img.size
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        return img, orig_w, orig_h

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img, orig_w, orig_h = self._load_image(image_path)

        objs = self.ann_dict[image_name]

        heatmap = np.zeros((self.input_h, self.input_w), dtype=np.float32)
        size_map = np.zeros((3, self.input_h, self.input_w), dtype=np.float32)  # w,h,t
        reg_mask = np.zeros((1, self.input_h, self.input_w), dtype=np.float32)

        meta_objs = []

        for obj in objs:
            if obj.get("label", "") != "hyperbola":
                continue

            # 原始坐标 -> resize 后坐标
            x_v = obj["x_vertex"] / orig_w * self.input_w
            y_v = obj["y_vertex"] / orig_h * self.input_h
            width = obj["width"] / orig_w * self.input_w
            height = obj["height"] / orig_h * self.input_h
            thickness = obj["thickness"] / orig_h * self.input_h

            band_mask = rasterize_hyperbola_band_mask(
                self.input_h, self.input_w, x_v, y_v, width, height, thickness
            )

            heatmap = np.maximum(heatmap, band_mask)
            reg_mask[0] = np.maximum(reg_mask[0], band_mask)

            if self.normalize_param:
                w_val = width / self.input_w
                h_val = height / self.input_h
                t_val = thickness / self.input_h
            else:
                w_val = width
                h_val = height
                t_val = thickness

            pos = band_mask > 0
            size_map[0, pos] = w_val
            size_map[1, pos] = h_val
            size_map[2, pos] = t_val

            meta_objs.append({
                "x_vertex": x_v,
                "y_vertex": y_v,
                "width": width,
                "height": height,
                "thickness": thickness
            })

        image_tensor = torch.from_numpy(img).unsqueeze(0).float()  # [1,H,W]
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()
        size_map_tensor = torch.from_numpy(size_map).float()
        reg_mask_tensor = torch.from_numpy(reg_mask).float()

        meta = {
            "image_name": image_name,
            "image_path": image_path,
            "objects": meta_objs,
            "orig_size": (orig_h, orig_w),
            "resized_size": (self.input_h, self.input_w),
        }

        return image_tensor, heatmap_tensor, size_map_tensor, reg_mask_tensor, meta


# =========================================================
# 3. 模型：简单 ResUNet 风格
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class HyperbolaNet(nn.Module):
    """
    输出：
      heatmap: [B,1,H,W]
      size_map: [B,3,H,W] -> width, height, thickness
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1)
        )

        self.size_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 3, 1)
        )

    def forward(self, x):
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)

        x = self.bottleneck(x)

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        heatmap = torch.sigmoid(self.heatmap_head(x))
        size_map = self.size_head(x)  # 回归值，不做 sigmoid
        return heatmap, size_map


# =========================================================
# 4. Loss
# =========================================================

class FocalHeatmapLoss(nn.Module):
    """
    CenterNet 风格的简化 heatmap focal loss
    pred/gt: [B,1,H,W], pred in [0,1]
    """
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)

        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return neg_loss
        else:
            return (pos_loss + neg_loss) / num_pos


def masked_l1_loss(pred, target, mask):
    """
    pred: [B,C,H,W]
    target: [B,C,H,W]
    mask: [B,1,H,W]
    只在 mask==1 的位置监督
    """
    mask = mask.expand_as(pred)
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    denom = mask.sum() + 1e-6
    return loss / denom


def hyperbola_collate_fn(batch):
    """
    自定义 DataLoader 拼接逻辑。
    - 张量字段按 batch 维堆叠
    - meta 保持为 list，避免其中变长 objects 触发默认 collate 报错
    """
    images, gt_hm, gt_size, reg_mask, metas = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.stack(gt_hm, dim=0),
        torch.stack(gt_size, dim=0),
        torch.stack(reg_mask, dim=0),
        list(metas),
    )


# =========================================================
# 5. 训练与验证
# =========================================================

def train_one_epoch(model, loader, optimizer, device, hm_loss_fn, lambda_size=5.0):
    model.train()
    total_loss = 0.0

    for images, gt_hm, gt_size, reg_mask, _ in loader:
        images = images.to(device)
        gt_hm = gt_hm.to(device)
        gt_size = gt_size.to(device)
        reg_mask = reg_mask.to(device)

        pred_hm, pred_size = model(images)

        loss_hm = hm_loss_fn(pred_hm, gt_hm)
        loss_size = masked_l1_loss(pred_size, gt_size, reg_mask)

        loss = loss_hm + lambda_size * loss_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device, hm_loss_fn, lambda_size=5.0):
    model.eval()
    total_loss = 0.0

    for images, gt_hm, gt_size, reg_mask, _ in loader:
        images = images.to(device)
        gt_hm = gt_hm.to(device)
        gt_size = gt_size.to(device)
        reg_mask = reg_mask.to(device)

        pred_hm, pred_size = model(images)

        loss_hm = hm_loss_fn(pred_hm, gt_hm)
        loss_size = masked_l1_loss(pred_size, gt_size, reg_mask)

        loss = loss_hm + lambda_size * loss_size
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# =========================================================
# 6. 推理与可视化
# =========================================================

def predict_single_image(model, image_path, input_size, device, normalize_param=True,
                         hm_thresh=0.35, max_det=20,
                         hm_thresh_mode="fixed", hm_thresh_quantile=0.85,
                         hm_thresh_min=0.20, hm_thresh_max=0.70,
                         min_component_score=0.20, merge_detections=True):
    input_h, input_w = input_size

    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    img_resized = img.resize((input_w, input_h), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        pred_hm, pred_size = model(x)

    pred_hm = pred_hm[0, 0].cpu().numpy()
    pred_size = pred_size[0].cpu().numpy()
    used_thresh = resolve_heatmap_threshold(
        pred_hm=pred_hm,
        hm_thresh=hm_thresh,
        hm_thresh_mode=hm_thresh_mode,
        hm_thresh_quantile=hm_thresh_quantile,
        hm_thresh_min=hm_thresh_min,
        hm_thresh_max=hm_thresh_max,
    )
    pred_mask = (pred_hm >= used_thresh).astype(np.float32)
    detections = decode_mask_to_hyperbola_detections(
        pred_mask=pred_mask,
        pred_hm=pred_hm,
        pred_size=pred_size,
        input_size=input_size,
        normalize_param=normalize_param,
        max_det=max_det,
        min_component_score=min_component_score,
        merge_detections=merge_detections,
    )
    return np.array(img_resized), pred_hm, pred_mask, detections, used_thresh


def draw_hyperbola_on_image(img_gray, detections, color=(0, 255, 0)):
    """
    在图上画出预测的双曲线近似形状
    这里用一个简单抛物线近似：
      y = y_v + height * ((x - x_v)/(width/2))^2
    只在区间 [x_v-width/2, x_v+width/2] 内绘制
    """
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    H, W = img_gray.shape

    for det in detections:
        x_v = float(det["x_vertex"])
        y_v = float(det["y_vertex"])
        width = max(float(det["width"]), 2.0)
        height = max(float(det["height"]), 1.0)
        thickness = max(float(det["thickness"]), 1.0)
        score = det.get("score", None)

        half_w = width / 2.0
        x_left = x_v - half_w
        x_right = x_v + half_w

        upper_pts = []
        lower_pts = []
        center_pts = []
        n_points = max(40, int(round(width)))

        for i in range(n_points + 1):
            t = i / max(n_points, 1)
            x = x_left + (x_right - x_left) * t
            dx = (x - x_v) / (half_w + 1e-6)
            y_center = y_v + height * (dx ** 2)

            y_up = y_center - thickness / 2.0
            y_dn = y_center + thickness / 2.0

            upper_pts.append((x, y_up))
            lower_pts.append((x, y_dn))
            center_pts.append((x, y_center))

        poly = upper_pts + list(reversed(lower_pts))
        poly_np = np.array(poly, dtype=np.float32)
        center_np = np.array(center_pts, dtype=np.float32)

        poly_np[:, 0] = np.clip(poly_np[:, 0], 0, W - 1)
        poly_np[:, 1] = np.clip(poly_np[:, 1], 0, H - 1)
        center_np[:, 0] = np.clip(center_np[:, 0], 0, W - 1)
        center_np[:, 1] = np.clip(center_np[:, 1], 0, H - 1)

        poly_int = np.round(poly_np).astype(np.int32)
        center_int = np.round(center_np).astype(np.int32)

        # 与标注工具一致：先半透明填充带状区域，再画中心线
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly_int], color)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.polylines(img, [center_int], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        cv2.circle(img, (int(round(x_v)), int(round(y_v))), 3, (0, 0, 255), -1)

        if score is not None:
            cv2.putText(
                img,
                f"{score:.2f}",
                (int(round(x_v)) + 3, int(round(y_v)) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

    return img


def save_prediction_visualization(model, dataset, indices, out_dir, device,
                                  input_size, normalize_param,
                                  hm_thresh=0.35, max_det=20,
                                  hm_thresh_mode="fixed", hm_thresh_quantile=0.85,
                                  hm_thresh_min=0.20, hm_thresh_max=0.70,
                                  min_component_score=0.20, merge_detections=True):
    ensure_dir(out_dir)

    def format_params_lines(tag: str, detections: List[Dict], max_show: int = 4) -> List[str]:
        if not detections:
            return [f"{tag}: none"]

        lines = []
        n_show = min(len(detections), max_show)
        for i in range(n_show):
            d = detections[i]
            line = (
                f"{tag}[{i}] "
                f"x={float(d['x_vertex']):.1f}, y={float(d['y_vertex']):.1f}, "
                f"w={float(d['width']):.1f}, h={float(d['height']):.1f}, t={float(d['thickness']):.1f}"
            )
            if "score" in d:
                line += f", s={float(d['score']):.2f}"
            lines.append(line)

        if len(detections) > max_show:
            lines.append(f"{tag}: ... and {len(detections) - max_show} more")

        return lines

    for idx in indices:
        image_tensor, gt_hm, gt_size, reg_mask, meta = dataset[idx]
        image_name = meta["image_name"]
        image_path = meta["image_path"]

        img_gray, pred_hm, pred_mask, pred_detections, used_thresh = predict_single_image(
            model=model,
            image_path=image_path,
            input_size=input_size,
            device=device,
            normalize_param=normalize_param,
            hm_thresh=hm_thresh,
            max_det=max_det,
            hm_thresh_mode=hm_thresh_mode,
            hm_thresh_quantile=hm_thresh_quantile,
            hm_thresh_min=hm_thresh_min,
            hm_thresh_max=hm_thresh_max,
            min_component_score=min_component_score,
            merge_detections=merge_detections,
        )

        gt_detections = []
        for obj in meta["objects"]:
            gt_detections.append({
                "x_vertex": float(obj["x_vertex"]),
                "y_vertex": float(obj["y_vertex"]),
                "width": float(obj["width"]),
                "height": float(obj["height"]),
                "thickness": float(obj["thickness"]),
            })

        gt_draw_img = draw_hyperbola_on_image(img_gray, gt_detections, color=(0, 255, 255))
        pred_draw_img = overlay_heatmap_on_image(img_gray, pred_mask)
        pred_curve_img = draw_hyperbola_on_image(img_gray, pred_detections, color=(0, 255, 0))

        fig = plt.figure(figsize=(20, 7.2))
        plt.subplot(1, 4, 1)
        plt.imshow(img_gray, cmap="gray")
        plt.title(f"Input: {image_name}")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(gt_draw_img, cv2.COLOR_BGR2RGB))
        plt.title("GT Hyperbola")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(cv2.cvtColor(pred_draw_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Pred Curve Mask (thr={used_thresh:.2f})")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(pred_curve_img, cv2.COLOR_BGR2RGB))
        plt.title("Regressed Hyperbola")
        plt.axis("off")

        gt_lines = format_params_lines("GT", gt_detections)
        pred_lines = format_params_lines("Pred", pred_detections)
        bottom_text = "\n".join(gt_lines + pred_lines)
        fig.text(0.01, 0.01, bottom_text, ha="left", va="bottom", fontsize=9, family="monospace")

        save_path = os.path.join(out_dir, image_name.replace(".jpg", "_pred.png"))
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


def detection_to_mask(det: Dict, input_size: Tuple[int, int]) -> np.ndarray:
    input_h, input_w = input_size
    return rasterize_hyperbola_band_mask(
        input_h,
        input_w,
        float(det["x_vertex"]),
        float(det["y_vertex"]),
        float(det["width"]),
        float(det["height"]),
        float(det["thickness"]),
    )


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a > 0
    mask_b = mask_b > 0
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def remove_small_mask_components(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Remove tiny scattered connected components from a binary mask."""
    mask_u8 = (mask > 0).astype(np.uint8)

    # Mild opening first to suppress isolated single-pixel noise.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)

    h, w = mask_u8.shape
    min_area_eff = max(int(min_area), int(round(0.0005 * h * w)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area_eff:
            cleaned[labels == label_id] = 1

    return cleaned.astype(np.float32)


def build_gt_mask_from_meta(meta: Dict, input_size: Tuple[int, int]) -> np.ndarray:
    input_h, input_w = input_size
    gt_mask = np.zeros((input_h, input_w), dtype=np.float32)
    for obj in meta["objects"]:
        obj_mask = detection_to_mask(obj, input_size)
        gt_mask = np.maximum(gt_mask, obj_mask)
    return gt_mask


@torch.no_grad()
def evaluate_mask_overlap(model, dataset, device, input_size, normalize_param,
                          hm_thresh=0.35, max_det=20,
                          hm_thresh_mode="fixed", hm_thresh_quantile=0.85,
                          hm_thresh_min=0.20, hm_thresh_max=0.70,
                          min_component_score=0.20, merge_detections=True,
                          mask_min_area=20):
    total_intersection = 0.0
    total_union = 0.0
    total_pred_pixels = 0.0
    total_gt_pixels = 0.0

    per_image_iou = []
    per_image_overlap = []

    for idx in range(len(dataset)):
        _, _, _, _, meta = dataset[idx]
        image_path = meta["image_path"]

        _, _, pred_mask, _, _ = predict_single_image(
            model=model,
            image_path=image_path,
            input_size=input_size,
            device=device,
            normalize_param=normalize_param,
            hm_thresh=hm_thresh,
            max_det=max_det,
            hm_thresh_mode=hm_thresh_mode,
            hm_thresh_quantile=hm_thresh_quantile,
            hm_thresh_min=hm_thresh_min,
            hm_thresh_max=hm_thresh_max,
            min_component_score=min_component_score,
            merge_detections=merge_detections,
        )

        pred_mask = remove_small_mask_components(pred_mask, min_area=mask_min_area)
        gt_mask = build_gt_mask_from_meta(meta, input_size)

        pred_bin = pred_mask > 0
        gt_bin = gt_mask > 0

        intersection = float(np.logical_and(pred_bin, gt_bin).sum())
        union = float(np.logical_or(pred_bin, gt_bin).sum())
        pred_pixels = float(pred_bin.sum())
        gt_pixels = float(gt_bin.sum())

        total_intersection += intersection
        total_union += union
        total_pred_pixels += pred_pixels
        total_gt_pixels += gt_pixels

        image_iou = intersection / max(union, 1e-6)
        image_overlap = intersection / max(gt_pixels, 1e-6)
        per_image_iou.append(image_iou)
        per_image_overlap.append(image_overlap)

    pixel_precision = total_intersection / max(total_pred_pixels, 1e-6)
    pixel_recall = total_intersection / max(total_gt_pixels, 1e-6)
    pixel_f1 = 2 * pixel_precision * pixel_recall / max(pixel_precision + pixel_recall, 1e-6)
    global_iou = total_intersection / max(total_union, 1e-6)
    global_overlap = total_intersection / max(total_gt_pixels, 1e-6)

    return {
        "num_test_images": len(dataset),
        "gt_pixels": int(total_gt_pixels),
        "pred_pixels": int(total_pred_pixels),
        "intersection_pixels": int(total_intersection),
        "union_pixels": int(total_union),
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_f1": pixel_f1,
        "global_iou": global_iou,
        "global_overlap": global_overlap,
        "mean_image_iou": float(np.mean(per_image_iou)) if per_image_iou else 0.0,
        "mean_image_overlap": float(np.mean(per_image_overlap)) if per_image_overlap else 0.0,
    }


def print_results_table(rows: List[Dict]):
    if not rows:
        return

    headers = [
        "lambda",
        "images",
        "GT_px",
        "Pred_px",
        "Inter_px",
        "Union_px",
        "P(px)",
        "R(px)",
        "F1(px)",
        "IoU(global)",
        "Overlap(global)",
    ]
    table_rows = []
    for row in rows:
        table_rows.append([
            str(row["lambda_size"]),
            str(row["num_test_images"]),
            str(row["gt_pixels"]),
            str(row["pred_pixels"]),
            str(row["intersection_pixels"]),
            str(row["union_pixels"]),
            f"{row['pixel_precision']:.4f}",
            f"{row['pixel_recall']:.4f}",
            f"{row['pixel_f1']:.4f}",
            f"{row['global_iou']:.4f}",
            f"{row['global_overlap']:.4f}",
        ])

    col_widths = []
    for col_idx, header in enumerate(headers):
        max_width = len(header)
        for row in table_rows:
            max_width = max(max_width, len(row[col_idx]))
        col_widths.append(max_width)

    def format_row(values):
        return " | ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

    print("\nTest Metrics")
    print(format_row(headers))
    print("-+-".join("-" * width for width in col_widths))
    for row in table_rows:
        print(format_row(row))


def save_results_table(rows: List[Dict], csv_path: str):
    fieldnames = [
        "lambda_size",
        "num_test_images",
        "gt_pixels",
        "pred_pixels",
        "intersection_pixels",
        "union_pixels",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "global_iou",
        "global_overlap",
        "mean_image_iou",
        "mean_image_overlap",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Add one explanation row so metric meanings are visible in the same CSV.
        writer.writerow({
            "lambda_size": "metric_meaning / 指标含义",
            "num_test_images": "number of test images / 测试图像数量",
            "gt_pixels": "total GT mask pixels / 标签mask总像素",
            "pred_pixels": "total predicted mask pixels after small-component removal / 去除小连通域后预测mask总像素",
            "intersection_pixels": "overlapped pixels between prediction and GT / 预测与标签重叠像素",
            "union_pixels": "union pixels between prediction and GT / 预测与标签并集像素",
            "pixel_precision": "intersection / pred_pixels / 像素精确率",
            "pixel_recall": "intersection / gt_pixels (same as global_overlap) / 像素召回率（同global_overlap）",
            "pixel_f1": "2*P*R/(P+R) / 像素级F1",
            "global_iou": "intersection / union_pixels / 全局IoU",
            "global_overlap": "intersection / gt_pixels / 全局重叠率",
            "mean_image_iou": "average per-image IoU / 每张图IoU均值",
            "mean_image_overlap": "average per-image overlap (intersection/GT) / 每张图重叠率均值",
        })

        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


# =========================================================
# 7. 主程序
# =========================================================

def main():
    # -----------------------
    # 路径设置
    # -----------------------
    data_sources = [
        {
            "image_dir": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhu/Utilities",
            "annotation_json": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhu/Utilities/annotations.json",
        },
        {
            "image_dir": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhu/cavities",
            "annotation_json": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhu/cavities/annotations.json",
        },
    ]

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    now = datetime.now()
    time_tag = now.strftime("%m%d_%H%M")
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{script_name}_{time_tag}")
    ensure_dir(work_dir)

    # -----------------------
    # 超参数
    # -----------------------
    set_seed(1)

    input_size = (256, 256)
    sigma = 3.0

    batch_size = 8
    num_epochs = 50
    lr = 1e-3
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 多组 lambda_size 对比实验
    lambda_size_list = [0.5, 1.0, 2, 3, 5, 10]
    lambda_size_list = [1.0]
    normalize_param = True

    # Heatmap threshold configuration:
    # - fixed: use hm_thresh directly
    # - adaptive: per-image quantile threshold, clamped by [hm_thresh_min, hm_thresh_max]
    hm_thresh_mode = "adaptive"  # "fixed" or "adaptive"
    hm_thresh = 0.35
    hm_thresh_quantile = 0.85
    hm_thresh_min = 0.20
    hm_thresh_max = 0.70

    # Decode robustness configuration:
    # - min_component_score: suppress weak tiny mask fragments.
    # - merge_detections: merge disconnected fragments of the same curve.
    min_component_score = 0.20
    merge_detections = True
    mask_min_area = 20
    max_det = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test ratio sum must be 1.0, but got {ratio_sum:.6f}"
        )

    # -----------------------
    # 数据
    # -----------------------
    datasets = []
    for source in data_sources:
        datasets.append(
            HyperbolaDataset(
                image_dir=source["image_dir"],
                annotation_json=source["annotation_json"],
                input_size=input_size,
                sigma=sigma,
                normalize_param=normalize_param,
            )
        )

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    split_generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=split_generator
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=hyperbola_collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=hyperbola_collate_fn
    )

    print(
        f"Total samples: {n_total}, Train: {n_train}, Val: {n_val}, Test: {n_test}"
    )

    test_results = []

    # -----------------------
    # 多 lambda 训练 + 可视化
    # -----------------------
    for lambda_size in lambda_size_list:
        lambda_tag = str(lambda_size).replace(".", "p")
        exp_dir = os.path.join(work_dir, f"lambda_{lambda_tag}")
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        vis_dir = os.path.join(exp_dir, "visuals")
        ensure_dir(exp_dir)
        ensure_dir(ckpt_dir)
        ensure_dir(vis_dir)

        print("\n" + "=" * 60)
        print(f"Start training with lambda_size={lambda_size}")
        print("=" * 60)

        model = HyperbolaNet(in_ch=1, base_ch=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        hm_loss_fn = FocalHeatmapLoss()

        best_val = 1e9

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, hm_loss_fn, lambda_size=lambda_size
            )
            val_loss = validate_one_epoch(
                model, val_loader, device, hm_loss_fn, lambda_size=lambda_size
            )

            print(
                f"[lambda={lambda_size}][Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)

            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(ckpt_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  -> Saved best model to {best_path}")

        best_path = os.path.join(ckpt_dir, "best_model.pth")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model for lambda={lambda_size}.")

        sample_indices = list(range(min(10, len(test_set))))
        sample_indices = list(range(len(test_set)))
        save_prediction_visualization(
            model=model,
            dataset=test_set,
            indices=sample_indices,
            out_dir=vis_dir,
            device=device,
            input_size=input_size,
            normalize_param=normalize_param,
            hm_thresh=hm_thresh,
            max_det=max_det,
            hm_thresh_mode=hm_thresh_mode,
            hm_thresh_quantile=hm_thresh_quantile,
            hm_thresh_min=hm_thresh_min,
            hm_thresh_max=hm_thresh_max,
            min_component_score=min_component_score,
            merge_detections=merge_detections,
        )

        metrics = evaluate_mask_overlap(
            model=model,
            dataset=test_set,
            device=device,
            input_size=input_size,
            normalize_param=normalize_param,
            hm_thresh=hm_thresh,
            max_det=max_det,
            hm_thresh_mode=hm_thresh_mode,
            hm_thresh_quantile=hm_thresh_quantile,
            hm_thresh_min=hm_thresh_min,
            hm_thresh_max=hm_thresh_max,
            min_component_score=min_component_score,
            merge_detections=merge_detections,
            mask_min_area=mask_min_area,
        )
        metrics["lambda_size"] = lambda_size
        test_results.append(metrics)

        print(f"Visualization saved to: {vis_dir}")
        print(
            f"Test Overlap(global)={metrics['global_overlap']:.4f}, "
            f"IoU(global)={metrics['global_iou']:.4f}, "
            f"P(px)={metrics['pixel_precision']:.4f}, "
            f"R(px)={metrics['pixel_recall']:.4f}, "
            f"F1(px)={metrics['pixel_f1']:.4f}"
        )

    print_results_table(test_results)
    results_csv_path = os.path.join(work_dir, "test_metrics_summary.csv")
    save_results_table(test_results, results_csv_path)
    print(f"Saved test metrics table to: {results_csv_path}")


if __name__ == "__main__":
    main()