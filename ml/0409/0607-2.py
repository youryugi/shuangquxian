import os
import re
import csv
from datetime import datetime
import json
import random
from typing import List, Dict, Tuple

# =========================================================
# Change Notes (2026-06-09)
# Based on 0607-1.py + ResNet-18 backbone:
#   - 用 torchvision 预训练 ResNet-18 替换自定义 ConvNet
#   - 取 layer2 输出（stride=8, 128ch），与 HM_STRIDE=8 匹配
#   - 第一层 conv 改为单通道输入，权重从 RGB 均值初始化
#   - 其余数据/损失/评估逻辑与 0607-1.py 完全一致
# =========================================================

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as tvm

# =========================================================
# 固定超参数
# =========================================================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_sources = [
    {
        "image_dir":       os.path.join(_REPO_ROOT, "biaozhumore", "Utilities_more"),
        "annotation_json": os.path.join(_REPO_ROOT, "biaozhumore", "Utilities_more", "annotations.json"),
    },
]

SEED          = 2
N_FOLDS       = 7
input_size    = (224, 224)
HM_STRIDE     = 8
batch_size    = 8
num_epochs    = 100
nms_kernel    = 3
mask_min_area = 20
max_det       = 5

HM_SIGMA  = 1.5
LAM       = 1.0
LR        = 5e-4
HM_THRESH = 0.30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Constants
# =========================================================

PARAM_CH = ("width", "height", "thickness")
N_PARAM  = len(PARAM_CH)


# =========================================================
# 1. 基础工具
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
    param_map:  np.ndarray,
    offset_map: np.ndarray,
    hm_size: Tuple[int, int],
    input_size: Tuple[int, int],
    score: float,
) -> Dict[str, float]:
    hm_h, hm_w      = hm_size
    input_h, input_w = input_size

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
        hm_stride: int = 4,
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
        self.image_names = sorted(
            list(self.ann_dict.keys()),
            key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0],
        )

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
            offset_map[0, hm_yi, hm_xi] = hm_cx - hm_xi
            offset_map[1, hm_yi, hm_xi] = hm_cy - hm_yi
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
            torch.from_numpy(img).unsqueeze(0).float(),
            torch.from_numpy(heatmap).unsqueeze(0).float(),
            torch.from_numpy(param_map).float(),
            torch.from_numpy(offset_map).float(),
            torch.from_numpy(peak_mask).unsqueeze(0).float(),
            meta,
        )


# =========================================================
# 4b. 数据增强
# =========================================================

def augment_sample(
    img_np: np.ndarray,
    hm_np:  np.ndarray,
    pm_np:  np.ndarray,
    om_np:  np.ndarray,
    pk_np:  np.ndarray,
):
    if random.random() < 0.5:
        img_np = img_np[:, ::-1].copy()
        hm_np  = hm_np[:, ::-1].copy()
        pm_np  = pm_np[:, :, ::-1].copy()
        om_flip = om_np[:, :, ::-1].copy()
        om_flip[0] = -om_flip[0]
        om_np  = om_flip
        pk_np  = pk_np[:, ::-1].copy()

    factor = random.uniform(0.7, 1.3)
    img_np = np.clip(img_np * factor, 0.0, 1.0)

    return img_np, hm_np, pm_np, om_np, pk_np


class AugWrapper(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_t, hm_t, pm_t, om_t, pk_t, meta = self.subset[idx]

        img_np = img_t.numpy()[0]
        hm_np  = hm_t.numpy()[0]
        pm_np  = pm_t.numpy()
        om_np  = om_t.numpy()
        pk_np  = pk_t.numpy()[0]

        img_np, hm_np, pm_np, om_np, pk_np = augment_sample(
            img_np, hm_np, pm_np, om_np, pk_np
        )

        return (
            torch.from_numpy(img_np).unsqueeze(0).float(),
            torch.from_numpy(hm_np).unsqueeze(0).float(),
            torch.from_numpy(pm_np).float(),
            torch.from_numpy(om_np).float(),
            torch.from_numpy(pk_np).unsqueeze(0).float(),
            meta,
        )


# =========================================================
# 5. 模型 — ResNet-18 backbone
# =========================================================
# ResNet-18 各层 stride（输入 224×224）:
#   conv1      → stride 2  → 112×112, 64ch
#   maxpool    → stride 2  →  56×56,  64ch
#   layer1     → stride 1  →  56×56,  64ch
#   layer2     → stride 2  →  28×28, 128ch   ← 取到这里，总 stride=8
#   layer3     → stride 2  →  14×14, 256ch
#   layer4     → stride 2  →   7×7,  512ch

class HyperbolaNetV2(nn.Module):
    def __init__(self, in_ch: int = 1, pretrained: bool = True):
        super().__init__()

        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        resnet  = tvm.resnet18(weights=weights)

        # 把第一层 conv 从 3ch 改为 in_ch，用 RGB 权重均值初始化
        old_w = resnet.conv1.weight.data          # (64, 3, 7, 7)
        new_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_w.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
        resnet.conv1 = new_conv

        # 编码器：取到 layer2，总 stride=8，输出 128ch
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        feat_ch = 128

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch // 2), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 2, 1, 1),
        )
        self.param_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch // 2), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 2, N_PARAM, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch // 4), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 4, 2, 1),
        )

    def forward(self, x):
        feat       = self.encoder(x)
        hm_logit   = self.heatmap_head(feat)
        raw        = self.param_head(feat)
        offset_out = self.offset_head(feat)
        param_out  = torch.cat([
            torch.sigmoid(raw[:, 0:1]),
            torch.sigmoid(raw[:, 1:2]),
            torch.sigmoid(raw[:, 2:3]),
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
    mask = peak_mask.expand_as(pred_offset)
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
    offset_np  = pred_offset[0].cpu().numpy()
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


def _instance_mask_iou(det: Dict, gt_obj: Dict, input_size: Tuple[int, int]) -> float:
    pred_bin = detection_to_mask(det,    input_size) > 0
    gt_bin   = detection_to_mask(gt_obj, input_size) > 0
    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    union = float(np.logical_or(pred_bin,  gt_bin).sum())
    return inter / max(union, 1e-6)


def compute_ap50(all_tp_flags: List[bool], all_scores: List[float], n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    pairs = sorted(zip(all_scores, all_tp_flags), key=lambda x: -x[0])
    tp = 0; fp = 0
    prec, rec = [], []
    for _, is_tp in pairs:
        if is_tp:
            tp += 1
        else:
            fp += 1
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

    ap_scores:   List[float] = []
    ap_tp_flags: List[bool]  = []
    n_gt_instances = 0

    for idx in range(len(dataset)):
        _, _, _, _, _, meta = dataset[idx]
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
        gt_matched = [False] * len(gt_objs)

        dets_sorted = sorted(detections, key=lambda d: d["score"], reverse=True)
        for det in dets_sorted:
            best_iou = 0.0
            best_j   = -1
            for j, gt_obj in enumerate(gt_objs):
                if gt_matched[j]:
                    continue
                iou = _instance_mask_iou(det, gt_obj, input_size)
                if iou > best_iou:
                    best_iou = iou
                    best_j   = j
            if best_iou >= 0.5 and best_j >= 0:
                ap_tp_flags.append(True)
                gt_matched[best_j] = True
            else:
                ap_tp_flags.append(False)
            ap_scores.append(float(det["score"]))

    pp = total_inter / max(total_pred,  1e-6)
    pr = total_inter / max(total_gt_px, 1e-6)
    f1 = 2 * pp * pr / max(pp + pr, 1e-6)
    return {
        "global_iou":      total_inter / max(total_union, 1e-6),
        "pixel_precision": pp,
        "pixel_recall":    pr,
        "pixel_f1":        f1,
        "mean_image_iou":  float(np.mean(per_iou)) if per_iou else 0.0,
        "mAP50":           compute_ap50(ap_tp_flags, ap_scores, n_gt_instances),
    }


# =========================================================
# 9b. 可视化保存
# =========================================================

def save_test_visuals(model, dataset, device, input_size, obj_thresh, save_dir):
    ensure_dir(save_dir)

    def _color_overlay(img_u8, mask, bgr):
        panel = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
        alpha = 0.3 * np.clip(mask, 0.0, 1.0)
        for c, v in enumerate(bgr):
            panel[:, :, c] = np.clip(
                panel[:, :, c] * (1.0 - alpha) + v * alpha, 0, 255
            )
        return panel.astype(np.uint8)

    for idx in range(len(dataset)):
        _, _, _, _, _, meta = dataset[idx]
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

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        white = (255, 255, 255)
        cv2.putText(panel_orig, "Original",            (4, 16), font, scale, white, thick, cv2.LINE_AA)
        cv2.putText(panel_gt,   "GT",                  (4, 16), font, scale, white, thick, cv2.LINE_AA)
        cv2.putText(panel_pred, f"Pred IoU={iou:.3f}", (4, 16), font, scale, white, thick, cv2.LINE_AA)

        vis  = np.hstack([panel_orig, panel_gt, panel_pred])
        stem = os.path.splitext(meta["image_name"])[0]
        cv2.imwrite(os.path.join(save_dir, f"{stem}.png"), vis)


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


def print_cv_summary(fold_results: List[Dict]):
    keys = ["global_iou", "pixel_f1", "pixel_precision", "pixel_recall", "mean_image_iou", "mAP50"]
    col_w = 18

    header = f"{'fold':>{col_w}}" + "".join(f"{k:>{col_w}}" for k in keys)
    sep    = "-" * len(header)
    print("\n" + "=" * len(header))
    print(f"7-Fold Cross-Validation Results  ResNet-18 backbone  (sigma={HM_SIGMA}, lam={LAM}, lr={LR}, thresh={HM_THRESH})")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in fold_results:
        row = f"{r['fold']:>{col_w}}" + "".join(f"{r[k]:>{col_w}.4f}" for k in keys)
        print(row)

    print(sep)
    means = {k: float(np.mean([r[k] for r in fold_results])) for k in keys}
    stds  = {k: float(np.std( [r[k] for r in fold_results])) for k in keys}
    mean_row = f"{'mean':>{col_w}}" + "".join(f"{means[k]:>{col_w}.4f}" for k in keys)
    std_row  = f"{'std':>{col_w}}"  + "".join(f"{stds[k]:>{col_w}.4f}" for k in keys)
    print(mean_row)
    print(std_row)
    print("=" * len(header))

    print(
        f"\n★ Summary:\n"
        f"  global_iou  = {means['global_iou']:.4f} ± {stds['global_iou']:.4f}\n"
        f"  pixel_f1    = {means['pixel_f1']:.4f} ± {stds['pixel_f1']:.4f}\n"
        f"  mAP50       = {means['mAP50']:.4f} ± {stds['mAP50']:.4f}"
    )


# =========================================================
# 11. 主程序
# =========================================================

def make_folds(n_total: int, n_folds: int) -> List[List[int]]:
    fold_size = n_total // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end   = start + fold_size if i < n_folds - 1 else n_total
        folds.append(list(range(start, end)))
    return folds


def main():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    now      = datetime.now()
    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{script_name}_{now.strftime('%m%d_%H%M')}",
    )
    ensure_dir(work_dir)
    print("Using device:", device)
    print(f"Backbone: ResNet-18 (pretrained, stride=8, 128ch)")
    print(f"7-Fold CV  sigma={HM_SIGMA}  lam={LAM}  lr={LR}  thresh={HM_THRESH}")

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

    fold_results: List[Dict] = []
    all_rows:     List[Dict] = []

    for fold_idx in range(N_FOLDS):
        test_idx  = folds[fold_idx]
        remaining = [i for i in range(n_total) if i not in set(test_idx)]

        rng = random.Random(SEED + fold_idx)
        perm = remaining.copy()
        rng.shuffle(perm)
        n_val     = len(folds[0])
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}  "
              f"test={len(test_idx)}  "
              f"val={len(val_idx)}  "
              f"train={len(train_idx)}")
        print(f"{'='*70}")

        test_set  = Subset(full_dataset, test_idx)
        val_set   = Subset(full_dataset, val_idx)
        train_set = AugWrapper(Subset(full_dataset, train_idx))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=0, collate_fn=hyperbola_collate_fn)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=0, collate_fn=hyperbola_collate_fn)

        fold_dir = os.path.join(work_dir, f"fold{fold_idx + 1:02d}")
        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        ensure_dir(ckpt_dir)

        model     = HyperbolaNetV2(in_ch=1, pretrained=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_val  = float("inf")
        best_path = os.path.join(ckpt_dir, "best_model.pth")

        for epoch in range(1, num_epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, LAM)
            va = validate_one_epoch(model, val_loader, device, LAM)
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
            f"IoU={metrics['global_iou']:.4f}  "
            f"F1={metrics['pixel_f1']:.4f}  "
            f"P={metrics['pixel_precision']:.4f}  "
            f"R={metrics['pixel_recall']:.4f}  "
            f"mAP50={metrics['mAP50']:.4f}"
        )

        vis_dir = os.path.join(fold_dir, "visuals")
        save_test_visuals(model, test_set, device, input_size, HM_THRESH, vis_dir)
        print(f"  Visuals saved → {vis_dir}")

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
