import os
import re
import csv
from datetime import datetime
import json
import random
from typing import Dict, Tuple, List

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ── 超参数 ──────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_sources = [
    {
        "image_dir":       os.path.join(_REPO_ROOT, "dataset3", "images-selected"),
        "annotation_json": os.path.join(_REPO_ROOT, "dataset3", "images-selected", "annotations.json"),
    },
]

SEED          = 2
N_FOLDS       = 7
input_size    = (640, 640)
HM_STRIDE     = 8
batch_size    = 8
num_epochs    = 100
nms_kernel    = 3
mask_min_area = 163
max_det       = 5

HM_SIGMA      = 4.3
LAM           = 1.0     # 粗参数 smooth-L1 权重
LAM_REFINE    = 1.0     # 精修参数 smooth-L1 权重
LAM_BAND      = 1.0     # Band IoU/Dice 权重（作用在精修后的参数上）
LAM_SEG       = 0.5     # 分割 head 监督权重
BAND_TEMP     = 1.0     # Band loss 软 mask 温度
ATT_TEMP      = 2.0     # 注意力池化软 mask 温度（略大更平滑）
BAND_LOSS_FN  = 'iou'   # 'iou' 或 'dice'
LR            = 5e-4
HM_THRESH     = 0.30
VERTEX_THRESH = 57.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Constants ────────────────────────────────────────────────────────────────
PARAM_CH = ("width", "height", "thickness")
N_PARAM  = len(PARAM_CH)


# ── 1. 基础工具 ───────────────────────────────────────────────────────────────
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
    # 与标注工具一致：thickness 为竖直厚度（沿 y 方向偏移 ±thickness/2）。
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


# ── 2. 编码 / 解码 ────────────────────────────────────────────────────────────
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


# ── 3. 数据集 ─────────────────────────────────────────────────────────────────
def render_gaussian(heatmap: np.ndarray, cx: float, cy: float, sigma: float):
    hm_h, hm_w = heatmap.shape
    ys = np.arange(hm_h, dtype=np.float32)[:, None]
    xs = np.arange(hm_w, dtype=np.float32)[None, :]
    g  = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    np.maximum(heatmap, g, out=heatmap)


class HyperbolaDataset(Dataset):
    def __init__(self, image_dir, annotation_json, input_size=(256, 256), hm_stride=4, sigma=2.0):
        super().__init__()
        self.image_dir   = image_dir
        self.input_h, self.input_w = input_size
        self.hm_h      = self.input_h // hm_stride
        self.hm_w      = self.input_w // hm_stride
        self.hm_stride = hm_stride
        self.sigma     = sigma
        with open(annotation_json, "r", encoding="utf-8") as f:
            self.ann_dict = json.load(f)
        self.image_names = sorted(
            list(self.ann_dict.keys()),
            key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0],
        )

    def __len__(self):
        return len(self.image_names)

    def _load_image(self, path):
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
        gt_seg     = np.zeros((self.input_h, self.input_w), dtype=np.float32)  # 全分辨率带 mask
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
            offset_map[0, hm_yi, hm_xi] = hm_cx - hm_xi
            offset_map[1, hm_yi, hm_xi] = hm_cy - hm_yi
            peak_mask[hm_yi, hm_xi]     = 1.0

            mo = {
                "x_vertex":  float(obj["x_vertex"])  / orig_w * self.input_w,
                "y_vertex":  float(obj["y_vertex"])  / orig_h * self.input_h,
                "width":     float(obj["width"])     / orig_w * self.input_w,
                "height":    float(obj["height"])    / orig_h * self.input_h,
                "thickness": float(obj["thickness"]) / orig_h * self.input_h,
            }
            meta_objs.append(mo)
            # 累加该实例的带到全分辨率分割 GT
            band = rasterize_hyperbola_band_mask(
                self.input_h, self.input_w,
                mo["x_vertex"], mo["y_vertex"], mo["width"], mo["height"], mo["thickness"],
            )
            gt_seg = np.maximum(gt_seg, band)

        meta = {
            "image_name":   name,
            "image_path":   path,
            "objects":      meta_objs,
            "orig_size":    (orig_h, orig_w),
            "resized_size": (self.input_h, self.input_w),
        }
        return (
            torch.from_numpy(img).unsqueeze(0).float(),
            torch.from_numpy(heatmap).unsqueeze(0).float(),
            torch.from_numpy(param_map).float(),
            torch.from_numpy(offset_map).float(),
            torch.from_numpy(peak_mask).unsqueeze(0).float(),
            torch.from_numpy(gt_seg).unsqueeze(0).float(),
            meta,
        )


# ── 4. 数据增强 ───────────────────────────────────────────────────────────────
def augment_sample(img_np, hm_np, pm_np, om_np, pk_np, seg_np):
    if random.random() < 0.5:
        img_np = img_np[:, ::-1].copy()
        hm_np  = hm_np[:, ::-1].copy()
        pm_np  = pm_np[:, :, ::-1].copy()
        om_flip = om_np[:, :, ::-1].copy()
        om_flip[0] = -om_flip[0]
        om_np  = om_flip
        pk_np  = pk_np[:, ::-1].copy()
        seg_np = seg_np[:, ::-1].copy()
    factor = random.uniform(0.7, 1.3)
    img_np = np.clip(img_np * factor, 0.0, 1.0)
    return img_np, hm_np, pm_np, om_np, pk_np, seg_np


class AugWrapper(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_t, hm_t, pm_t, om_t, pk_t, seg_t, meta = self.subset[idx]
        img_np, hm_np, pm_np, om_np, pk_np, seg_np = augment_sample(
            img_t.numpy()[0], hm_t.numpy()[0], pm_t.numpy(), om_t.numpy(),
            pk_t.numpy()[0], seg_t.numpy()[0],
        )
        return (
            torch.from_numpy(img_np).unsqueeze(0).float(),
            torch.from_numpy(hm_np).unsqueeze(0).float(),
            torch.from_numpy(pm_np).float(),
            torch.from_numpy(om_np).float(),
            torch.from_numpy(pk_np).unsqueeze(0).float(),
            torch.from_numpy(seg_np).unsqueeze(0).float(),
            meta,
        )


# ── 5. 模型 ───────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # 尺寸对齐（防奇偶不齐）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# 可微分软 band mask（竖直厚度，与标注一致）——既用于注意力池化，也用于 Band loss
def soft_band_mask(
    h: int, w: int,
    x_v: torch.Tensor, y_v: torch.Tensor,
    width: torch.Tensor, height: torch.Tensor, thickness: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    device = x_v.device
    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)
    py, px = torch.meshgrid(ys, xs, indexing='ij')

    half_w = width / 2.0
    dx     = (px - x_v) / (half_w + 1e-6)
    yc     = y_v + height * dx.pow(2)

    dist = (py - yc).abs()
    band = torch.sigmoid((thickness / 2.0 - dist) / temperature)

    x_in = (
        torch.sigmoid((px - (x_v - half_w)) / temperature) *
        torch.sigmoid(((x_v + half_w) - px) / temperature)
    )
    return band * x_in


class HyperbolaNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.down1      = DownBlock(in_ch,       base_ch)        # f1: stride1, base_ch
        self.down2      = DownBlock(base_ch,     base_ch * 2)    # f2: stride2, base_ch*2
        self.down3      = DownBlock(base_ch * 2, base_ch * 4)    # f3: stride4, base_ch*4
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)    # stride8, base_ch*8
        mid = base_ch * 8

        # 检测三个 head（在 stride8 上）
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
        self.offset_head = nn.Sequential(
            nn.Conv2d(mid, mid // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid // 4), nn.ReLU(inplace=True),
            nn.Conv2d(mid // 4, 2, 1),
        )

        # 想法①：U-Net 解码器 + 分割 head（用回 skip 特征）
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)   # stride8→4
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)   # stride4→2
        self.up1 = UpBlock(base_ch * 2, base_ch,     base_ch)       # stride2→1
        self.seg_head = nn.Conv2d(base_ch, 1, 1)                    # 全分辨率带 mask logit

        # 想法②：参数精修 MLP（沿带注意力池化后回归 width/height/thickness 残差）
        self.refine_mlp = nn.Sequential(
            nn.Linear(mid, mid // 2), nn.ReLU(inplace=True),
            nn.Linear(mid // 2, N_PARAM),
        )

    def forward(self, x):
        f1, x = self.down1(x)
        f2, x = self.down2(x)
        f3, x = self.down3(x)
        x     = self.bottleneck(x)
        feat  = x                      # bottleneck 特征，供注意力池化

        hm_logit   = self.heatmap_head(x)
        raw        = self.param_head(x)
        offset_out = self.offset_head(x)
        param_out  = torch.sigmoid(raw)            # 粗参数，(B,3,hm,hm) ∈ [0,1]

        d = self.up3(x, f3)
        d = self.up2(d, f2)
        d = self.up1(d, f1)
        seg_logit = self.seg_head(d)               # (B,1,H,W)

        return hm_logit, param_out, offset_out, seg_logit, feat

    def refine_one(self, feat_b, w_n, h_n, t_n, xv_hm, yv_hm, hm_h, hm_w, att_temp):
        """
        feat_b: (C, hm_h, hm_w) 单样本 bottleneck 特征
        w_n/h_n/t_n: 该峰值处的粗参数（归一化标量 Tensor）
        xv_hm/yv_hm: 顶点在 heatmap 网格中的坐标（标量 Tensor）
        返回精修后的归一化 (3,) 参数。
        """
        # 在 bottleneck 分辨率上用粗参数渲染软注意力
        w_hm = w_n * hm_w
        h_hm = h_n * hm_h
        t_hm = t_n * hm_h
        A = soft_band_mask(hm_h, hm_w, xv_hm, yv_hm, w_hm, h_hm, t_hm, att_temp)  # (hm_h, hm_w)
        A = A + 1e-4
        pooled = (feat_b * A.unsqueeze(0)).sum(dim=(1, 2)) / A.sum()              # (C,)
        residual = self.refine_mlp(pooled)                                        # (3,)
        refined = torch.stack([w_n, h_n, t_n]) + residual
        return refined.clamp(0.0, 1.0)


# ── 6. Loss 函数 ──────────────────────────────────────────────────────────────
def focal_loss_heatmap(pred_logit, target_hm, peak_mask, alpha=2.0, beta=4.0):
    pred     = torch.sigmoid(pred_logit)
    pos_mask = peak_mask
    neg_w    = (1.0 - target_hm) ** beta
    pos_loss = pos_mask * (1.0 - pred) ** alpha * torch.log(pred.clamp(min=1e-6))
    neg_loss = neg_w * (1.0 - pos_mask) * pred ** alpha * torch.log((1.0 - pred).clamp(min=1e-6))
    n_pos    = pos_mask.sum().clamp(min=1.0)
    return -(pos_loss + neg_loss).sum() / n_pos


def masked_param_loss_spatial(pred_param, gt_param, peak_mask):
    mask = peak_mask.expand_as(pred_param)
    n    = mask.sum()
    if n == 0:
        return pred_param.sum() * 0.0
    loss = F.smooth_l1_loss(pred_param * mask, gt_param * mask, reduction="sum")
    return loss / (n / N_PARAM + 1e-6)


def masked_offset_loss(pred_offset, gt_offset, peak_mask):
    mask = peak_mask.expand_as(pred_offset)
    n    = mask.sum()
    if n == 0:
        return pred_offset.sum() * 0.0
    loss = F.l1_loss(pred_offset * mask, gt_offset * mask, reduction="sum")
    return loss / (n / 2 + 1e-6)


def seg_loss_fn(seg_logit, gt_seg):
    bce  = F.binary_cross_entropy_with_logits(seg_logit, gt_seg)
    prob = torch.sigmoid(seg_logit)
    inter = (prob * gt_seg).sum()
    dice  = 1.0 - 2.0 * inter / (prob.sum() + gt_seg.sum() + 1e-6)
    return bce + dice


def band_iou_loss(h, w, p_xv, p_yv, p_w, p_h, p_t, g_xv, g_yv, g_w, g_h, g_t, temperature=2.0):
    pred_mask = soft_band_mask(h, w, p_xv, p_yv, p_w, p_h, p_t, temperature)
    gt_mask   = soft_band_mask(h, w, g_xv, g_yv, g_w, g_h, g_t, temperature)
    inter = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - inter
    return 1.0 - inter / (union + 1e-6)


def band_dice_loss(h, w, p_xv, p_yv, p_w, p_h, p_t, g_xv, g_yv, g_w, g_h, g_t, temperature=2.0):
    pred_mask = soft_band_mask(h, w, p_xv, p_yv, p_w, p_h, p_t, temperature)
    gt_mask   = soft_band_mask(h, w, g_xv, g_yv, g_w, g_h, g_t, temperature)
    inter = (pred_mask * gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    return 1.0 - 2.0 * inter / (denom + 1e-6)


def compute_refinement_losses(
    model, feat, param_coarse, offset_pred, gt_param, gt_offset, peak_mask,
    input_h, input_w, hm_h, hm_w, att_temp, band_temp, loss_fn='iou',
):
    """对每个 GT 峰值做注意力池化精修，返回 (精修参数 smooth-L1, Band loss)。"""
    B = param_coarse.shape[0]
    p_losses, b_losses = [], []

    for b in range(B):
        peak_yx = peak_mask[b, 0].nonzero(as_tuple=False)
        for yx in peak_yx:
            yi, xi = int(yx[0]), int(yx[1])

            w_n = param_coarse[b, 0, yi, xi]
            h_n = param_coarse[b, 1, yi, xi]
            t_n = param_coarse[b, 2, yi, xi]
            pd_dx = offset_pred[b, 0, yi, xi]
            pd_dy = offset_pred[b, 1, yi, xi]
            xv_hm = xi + pd_dx
            yv_hm = yi + pd_dy

            refined = model.refine_one(feat[b], w_n, h_n, t_n, xv_hm, yv_hm, hm_h, hm_w, att_temp)

            gw = gt_param[b, 0, yi, xi]
            gh = gt_param[b, 1, yi, xi]
            gtk = gt_param[b, 2, yi, xi]
            p_losses.append(F.smooth_l1_loss(refined, torch.stack([gw, gh, gtk])))

            # Band loss（像素单位，作用在精修参数上）
            gt_dx = gt_offset[b, 0, yi, xi]
            gt_dy = gt_offset[b, 1, yi, xi]
            g_xv = (xi + gt_dx) / hm_w * input_w
            g_yv = (yi + gt_dy) / hm_h * input_h
            g_w  = gw * input_w
            g_h  = gh * input_h
            g_t  = gtk * input_h

            p_xv = xv_hm / hm_w * input_w
            p_yv = yv_hm / hm_h * input_h
            p_w  = refined[0] * input_w
            p_h  = refined[1] * input_h
            p_t  = refined[2] * input_h

            fn = band_dice_loss if loss_fn == 'dice' else band_iou_loss
            b_losses.append(fn(input_h, input_w, p_xv, p_yv, p_w, p_h, p_t,
                               g_xv, g_yv, g_w, g_h, g_t, band_temp))

    zero = param_coarse.sum() * 0.0
    pl = torch.stack(p_losses).mean() if p_losses else zero
    bl = torch.stack(b_losses).mean() if b_losses else zero
    return pl, bl


def hyperbola_collate_fn(batch):
    images, heatmaps, param_maps, offset_maps, peak_masks, gt_segs, metas = zip(*batch)
    return (
        torch.stack(images,      dim=0),
        torch.stack(heatmaps,    dim=0),
        torch.stack(param_maps,  dim=0),
        torch.stack(offset_maps, dim=0),
        torch.stack(peak_masks,  dim=0),
        torch.stack(gt_segs,     dim=0),
        list(metas),
    )


# ── 7. 训练 / 验证 ────────────────────────────────────────────────────────────
def _compute_total_loss(model, images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg):
    hm_h = input_size[0] // HM_STRIDE
    hm_w = input_size[1] // HM_STRIDE

    pred_hm, pred_param, pred_offset, seg_logit, feat = model(images)

    l_focal  = focal_loss_heatmap(pred_hm, gt_hm, peak_mask)
    l_coarse = masked_param_loss_spatial(pred_param, gt_param, peak_mask)
    l_offset = masked_offset_loss(pred_offset, gt_offset, peak_mask)
    l_seg    = seg_loss_fn(seg_logit, gt_seg)
    l_refine, l_band = compute_refinement_losses(
        model, feat, pred_param, pred_offset, gt_param, gt_offset, peak_mask,
        input_h=input_size[0], input_w=input_size[1], hm_h=hm_h, hm_w=hm_w,
        att_temp=ATT_TEMP, band_temp=BAND_TEMP, loss_fn=BAND_LOSS_FN,
    )

    return (
        l_focal
        + LAM        * l_coarse
        + l_offset
        + LAM_SEG    * l_seg
        + LAM_REFINE * l_refine
        + LAM_BAND   * l_band
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg, _ in loader:
        images    = images.to(device)
        gt_hm     = gt_hm.to(device)
        gt_param  = gt_param.to(device)
        gt_offset = gt_offset.to(device)
        peak_mask = peak_mask.to(device)
        gt_seg    = gt_seg.to(device)

        loss = _compute_total_loss(model, images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg, _ in loader:
        images    = images.to(device)
        gt_hm     = gt_hm.to(device)
        gt_param  = gt_param.to(device)
        gt_offset = gt_offset.to(device)
        peak_mask = peak_mask.to(device)
        gt_seg    = gt_seg.to(device)

        loss = _compute_total_loss(model, images, gt_hm, gt_param, gt_offset, peak_mask, gt_seg)
        total += loss.item()
    return total / max(len(loader), 1)


# ── 8. 推理 ───────────────────────────────────────────────────────────────────
def _heatmap_nms(hm: np.ndarray, kernel: int) -> np.ndarray:
    t    = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    tmax = F.max_pool2d(t, kernel, stride=1, padding=kernel // 2)
    keep = (t == tmax).squeeze(0).squeeze(0).numpy()
    return hm * keep


@torch.no_grad()
def predict_single_image(model, image_path, input_size, device, obj_thresh=0.2, nms_k=3, max_det=20):
    input_h, input_w = input_size
    hm_h = input_h // HM_STRIDE
    hm_w = input_w // HM_STRIDE

    img_pil     = Image.open(image_path).convert("L")
    img_resized = img_pil.resize((input_w, input_h), Image.BILINEAR)
    img_np      = np.array(img_resized, dtype=np.float32) / 255.0

    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    model.eval()
    pred_hm_logit, pred_param, pred_offset, seg_logit, feat = model(x)

    hm     = torch.sigmoid(pred_hm_logit[0, 0]).cpu().numpy()
    hm_nms = _heatmap_nms(hm, nms_k)

    ys, xs = np.where(hm_nms >= obj_thresh)
    if len(ys) == 0:
        return np.array(img_resized), np.zeros((input_h, input_w), np.float32), []

    scores = hm_nms[ys, xs]
    order  = np.argsort(scores)[::-1][:max_det]
    ys, xs, scores = ys[order], xs[order], scores[order]

    detections    = []
    combined_mask = np.zeros((input_h, input_w), dtype=np.float32)
    for yi, xi, score in zip(ys, xs, scores):
        w_n = pred_param[0, 0, yi, xi]
        h_n = pred_param[0, 1, yi, xi]
        t_n = pred_param[0, 2, yi, xi]
        dx  = torch.clamp(pred_offset[0, 0, yi, xi], -0.5, 0.5)
        dy  = torch.clamp(pred_offset[0, 1, yi, xi], -0.5, 0.5)
        xv_hm = xi + dx
        yv_hm = yi + dy

        refined = model.refine_one(feat[0], w_n, h_n, t_n, xv_hm, yv_hm, hm_h, hm_w, ATT_TEMP)

        x_v       = float(np.clip(float(xv_hm) / hm_w * input_w, 0.0, input_w - 1))
        y_v       = float(np.clip(float(yv_hm) / hm_h * input_h, 0.0, input_h - 1))
        width     = max(float(refined[0]) * input_w, 2.0)
        height    = max(float(refined[1]) * input_h, 1.0)
        thickness = max(float(refined[2]) * input_h, 1.0)
        det = {
            "score": float(score), "x_vertex": x_v, "y_vertex": y_v,
            "width": width, "height": height, "thickness": thickness,
        }
        detections.append(det)
        combined_mask = np.maximum(combined_mask, detection_to_mask(det, input_size))

    return np.array(img_resized), combined_mask, detections


# ── 9. 评估 ──────────────────────────────────────────────────────────────────
def build_gt_mask_from_meta(meta, input_size):
    h, w    = input_size
    gt_mask = np.zeros((h, w), dtype=np.float32)
    for obj in meta["objects"]:
        gt_mask = np.maximum(gt_mask, detection_to_mask(obj, input_size))
    return gt_mask


def _instance_mask_iou(det, gt_obj, input_size):
    pred_bin = detection_to_mask(det,    input_size) > 0
    gt_bin   = detection_to_mask(gt_obj, input_size) > 0
    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    union = float(np.logical_or(pred_bin,  gt_bin).sum())
    return inter / max(union, 1e-6)


def _vertex_dist(det, gt_obj):
    dx = det["x_vertex"] - gt_obj["x_vertex"]
    dy = det["y_vertex"] - gt_obj["y_vertex"]
    return float(np.sqrt(dx * dx + dy * dy))


def compute_ap50(all_tp_flags, all_scores, n_gt):
    if n_gt == 0:
        return 0.0
    pairs = sorted(zip(all_scores, all_tp_flags), key=lambda x: -x[0])
    tp = 0; fp = 0
    prec, rec = [], []
    for _, is_tp in pairs:
        if is_tp: tp += 1
        else:     fp += 1
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p_at_t = [p for p, r in zip(prec, rec) if r >= t]
        ap += (max(p_at_t) if p_at_t else 0.0)
    return ap / 11.0


@torch.no_grad()
def evaluate(model, dataset, device, input_size, obj_thresh):
    total_inter = total_union = total_pred = total_gt_px = 0.0
    per_iou = []
    ap_scores, ap_tp_flags = [], []
    n_gt_instances    = 0
    total_gt_detected = 0
    vertex_dists      = []

    for idx in range(len(dataset)):
        meta = dataset[idx][-1]
        _, pred_mask, detections = predict_single_image(
            model, meta["image_path"], input_size, device,
            obj_thresh=obj_thresh, nms_k=nms_kernel, max_det=max_det,
        )
        pred_mask = remove_small_mask_components(pred_mask, min_area=mask_min_area)
        gt_mask   = build_gt_mask_from_meta(meta, input_size)

        pred_bin = pred_mask > 0
        gt_bin   = gt_mask   > 0
        inter    = float(np.logical_and(pred_bin, gt_bin).sum())
        union    = float(np.logical_or(pred_bin,  gt_bin).sum())
        total_inter  += inter
        total_union  += union
        total_pred   += float(pred_bin.sum())
        total_gt_px  += float(gt_bin.sum())
        per_iou.append(1.0 if union == 0 else inter / union)

        gt_objs   = meta["objects"]
        n_gt_instances += len(gt_objs)

        gt_matched_iou = [False] * len(gt_objs)
        for det in sorted(detections, key=lambda d: d["score"], reverse=True):
            best_iou, best_j = 0.0, -1
            for j, gt_obj in enumerate(gt_objs):
                if gt_matched_iou[j]: continue
                iou = _instance_mask_iou(det, gt_obj, input_size)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                ap_tp_flags.append(True)
                gt_matched_iou[best_j] = True
            else:
                ap_tp_flags.append(False)
            ap_scores.append(float(det["score"]))

        gt_matched_vtx = [False] * len(gt_objs)
        for gt_idx, gt_obj in enumerate(gt_objs):
            best_dist = float("inf")
            for det in detections:
                d = _vertex_dist(det, gt_obj)
                if d < best_dist:
                    best_dist = d
            if best_dist <= VERTEX_THRESH:
                gt_matched_vtx[gt_idx] = True
                vertex_dists.append(best_dist)

        total_gt_detected += sum(gt_matched_vtx)

    pp = total_inter / max(total_pred,  1e-6)
    pr = total_inter / max(total_gt_px, 1e-6)
    f1 = 2 * pp * pr / max(pp + pr, 1e-6)
    return {
        "global_iou":       total_inter / max(total_union, 1e-6),
        "pixel_precision":  pp,
        "pixel_recall":     pr,
        "pixel_f1":         f1,
        "mean_image_iou":   float(np.mean(per_iou)) if per_iou else 0.0,
        "mAP50":            compute_ap50(ap_tp_flags, ap_scores, n_gt_instances),
        "instance_recall":  total_gt_detected / max(n_gt_instances, 1),
        "mean_vertex_dist": float(np.mean(vertex_dists)) if vertex_dists else float("nan"),
    }


# ── 10. 可视化保存 ────────────────────────────────────────────────────────────
def save_test_visuals(model, dataset, device, input_size, obj_thresh, save_dir):
    ensure_dir(save_dir)

    def _color_overlay(img_u8, mask, bgr):
        panel = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
        alpha = 0.3 * np.clip(mask, 0.0, 1.0)
        for c, v in enumerate(bgr):
            panel[:, :, c] = np.clip(panel[:, :, c] * (1.0 - alpha) + v * alpha, 0, 255)
        return panel.astype(np.uint8)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    white = (255, 255, 255)

    for idx in range(len(dataset)):
        meta = dataset[idx][-1]
        img_u8, pred_mask, _ = predict_single_image(
            model, meta["image_path"], input_size, device,
            obj_thresh=obj_thresh, nms_k=nms_kernel, max_det=max_det,
        )
        pred_mask = remove_small_mask_components(pred_mask, min_area=mask_min_area)
        gt_mask   = build_gt_mask_from_meta(meta, input_size)

        inter = float(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
        union = float(np.logical_or(pred_mask > 0,  gt_mask > 0).sum())
        iou   = 1.0 if union == 0 else inter / union

        panel_orig = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        panel_gt   = _color_overlay(img_u8, gt_mask,   (0, 200, 0))
        panel_pred = _color_overlay(img_u8, pred_mask, (0, 0, 220))

        cv2.putText(panel_orig, "Original",            (4, 16), font, scale, white, thick, cv2.LINE_AA)
        cv2.putText(panel_gt,   "GT",                  (4, 16), font, scale, white, thick, cv2.LINE_AA)
        cv2.putText(panel_pred, f"Pred IoU={iou:.3f}", (4, 16), font, scale, white, thick, cv2.LINE_AA)

        vis  = np.hstack([panel_orig, panel_gt, panel_pred])
        stem = os.path.splitext(meta["image_name"])[0]
        cv2.imwrite(os.path.join(save_dir, f"{stem}.png"), vis)


# ── 11. 结果输出 ──────────────────────────────────────────────────────────────
def save_results_csv(rows, csv_path):
    if not rows: return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_cv_summary(fold_results):
    keys = ["global_iou", "pixel_f1", "mAP50", "instance_recall", "mean_vertex_dist"]
    col_w = 18
    header = f"{'fold':>{col_w}}" + "".join(f"{k:>{col_w}}" for k in keys)
    sep    = "-" * len(header)
    print("\n" + "=" * len(header))
    print(f"7-Fold CV  sigma={HM_SIGMA}  lam={LAM}  lam_refine={LAM_REFINE}  lam_band={LAM_BAND}  lam_seg={LAM_SEG}  lr={LR}")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in fold_results:
        print(f"{r['fold']:>{col_w}}" + "".join(f"{r[k]:>{col_w}.4f}" for k in keys))
    print(sep)
    means = {k: float(np.nanmean([r[k] for r in fold_results])) for k in keys}
    stds  = {k: float(np.nanstd( [r[k] for r in fold_results])) for k in keys}
    print(f"{'mean':>{col_w}}" + "".join(f"{means[k]:>{col_w}.4f}" for k in keys))
    print(f"{'std':>{col_w}}"  + "".join(f"{stds[k]:>{col_w}.4f}"  for k in keys))
    print("=" * len(header))
    print(
        f"\nSummary:\n"
        f"  instance_recall  = {means['instance_recall']:.4f} ± {stds['instance_recall']:.4f}"
        f"  (顶点距离 ≤ {VERTEX_THRESH:.0f}px 算检出)\n"
        f"  mean_vertex_dist = {means['mean_vertex_dist']:.2f} ± {stds['mean_vertex_dist']:.2f} px\n"
        f"  mAP50            = {means['mAP50']:.4f} ± {stds['mAP50']:.4f}\n"
        f"  global_iou       = {means['global_iou']:.4f} ± {stds['global_iou']:.4f}"
    )


# ── 12. 主程序 ────────────────────────────────────────────────────────────────
def make_folds(n_total, n_folds):
    fold_size = n_total // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end   = start + fold_size if i < n_folds - 1 else n_total
        folds.append(list(range(start, end)))
    return folds


def main():
    now      = datetime.now()
    work_dir = os.path.join(os.getcwd(), f"0615-1_{now.strftime('%m%d_%H%M')}")
    ensure_dir(work_dir)
    print("Using device:", device)
    print(f"7-Fold CV  sigma={HM_SIGMA}  lam={LAM}  lam_refine={LAM_REFINE}  lam_band={LAM_BAND}  "
          f"lam_seg={LAM_SEG}  band_temp={BAND_TEMP}  att_temp={ATT_TEMP}  band_loss={BAND_LOSS_FN}  lr={LR}")

    set_seed(SEED)
    full_dataset = HyperbolaDataset(
        image_dir=data_sources[0]["image_dir"],
        annotation_json=data_sources[0]["annotation_json"],
        input_size=input_size,
        hm_stride=HM_STRIDE,
        sigma=HM_SIGMA,
    )
    n_total = len(full_dataset)
    print(f"Total samples: {n_total}")

    folds = make_folds(n_total, N_FOLDS)
    for i, f in enumerate(folds):
        print(f"  fold {i}: {len(f)} samples")

    fold_results, all_rows = [], []

    for fold_idx in range(N_FOLDS):
        test_idx  = folds[fold_idx]
        remaining = [i for i in range(n_total) if i not in set(test_idx)]
        rng  = random.Random(SEED + fold_idx)
        perm = remaining.copy()
        rng.shuffle(perm)
        n_val     = len(folds[0])
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}  test={len(test_idx)}  val={len(val_idx)}  train={len(train_idx)}")
        print(f"{'='*70}")

        test_set   = Subset(full_dataset, test_idx)
        val_set    = Subset(full_dataset, val_idx)
        train_set  = AugWrapper(Subset(full_dataset, train_idx))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=0, collate_fn=hyperbola_collate_fn)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=0, collate_fn=hyperbola_collate_fn)

        fold_dir = os.path.join(work_dir, f"fold{fold_idx + 1:02d}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        ensure_dir(ckpt_dir)

        model     = HyperbolaNet(in_ch=1, base_ch=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_val  = float("inf")
        best_path = os.path.join(ckpt_dir, "best_model.pth")

        for epoch in range(1, num_epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device)
            va = validate_one_epoch(model, val_loader, device)
            if epoch % 10 == 0 or epoch == num_epochs:
                print(f"  [Epoch {epoch:03d}/{num_epochs}] train={tr:.5f}  val={va:.5f}")
            if va < best_val:
                best_val = va
                torch.save(model.state_dict(), best_path)

        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        print(f"  Best val loss: {best_val:.5f}")

        metrics = evaluate(model, test_set, device, input_size, obj_thresh=HM_THRESH)
        print(
            f"  [Fold {fold_idx + 1}] "
            f"IoU={metrics['global_iou']:.4f}  F1={metrics['pixel_f1']:.4f}  "
            f"mAP50={metrics['mAP50']:.4f}  "
            f"inst_recall={metrics['instance_recall']:.4f}  "
            f"vtx_dist={metrics['mean_vertex_dist']:.2f}px"
        )

        vis_dir = os.path.join(fold_dir, "visuals")
        save_test_visuals(model, test_set, device, input_size, HM_THRESH, vis_dir)
        print(f"  Visuals saved -> {vis_dir}")

        row = {"fold": fold_idx + 1, "best_val_loss": round(best_val, 6)}
        row.update(metrics)
        fold_results.append(row)
        all_rows.append(row)

    csv_path = os.path.join(work_dir, "cv_results.csv")
    save_results_csv(all_rows, csv_path)
    print_cv_summary(fold_results)
    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()