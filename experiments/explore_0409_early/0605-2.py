import os
import csv
from datetime import datetime
import json
import math
import random
from typing import List, Dict, Tuple

# =========================================================
# Change Notes (2026-06-05)
# Physics-constrained hyperbola regression (vs 0527-1.py):
#   - Network predicts velocity proxy `v` instead of `height`.
#   - height is DERIVED from the GPR parabola physics equation:
#       height = width^2 / (2 * v^2 * y_vertex)   [normalized, square image]
#   - This enforces that predicted hyperbola shape is physically consistent.
#   - v corresponds to the image-space propagation velocity (pixel_x / pixel_y).
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
# 超参数配置
# =========================================================

data_sources = [
    {
        "image_dir": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhumore/Utilities",
        "annotation_json": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhumore/Utilities/annotations.json",
    },
    # {
    #     "image_dir": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhumore/cavities",
    #     "annotation_json": r"C:/Users/79152/Desktop/github/shuangquxian/biaozhumore/cavities/annotations.json",
    # },
]

seed_list = list(range(1, 11))
seed_list = [1]

input_size = (256, 256)
sigma = 3.0

batch_size = 8
num_epochs = 50
lr = 1e-3
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

lambda_size_list = [1]
normalize_param = True  # must stay True for physics formula to apply

hm_thresh = 0.5          # objectness score threshold
mask_min_area = 20
max_det = 20
match_iou_threshold = 0.10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Constants
# =========================================================

RANDOM_SEEDS = list(range(1, 2))

# v (velocity proxy) replaces height as a predicted parameter.
# height is always computed from physics during decode.
PARAM_KEYS = ("x_vertex", "y_vertex", "width", "v", "thickness")


# =========================================================
# 1. 物理公式
# =========================================================

def height_from_physics(y_v_norm: float, width_norm: float, v: float) -> float:
    """
    GPR parabola approximation in normalized square-image coordinates:

        y(x) = y_v + a * (x - x_v)^2
        a    = 2 / (v^2 * y_v)          [image-pixel curvature]
        height = a * (width/2)^2
               = width^2 / (2 * v^2 * y_v)

    Parameters
    ----------
    y_v_norm  : y_vertex normalized by image height  (∈ (0, 1])
    width_norm: width    normalized by image width   (∈ (0, 1])
    v         : image-space velocity proxy           (> 0)

    Returns
    -------
    height_norm : height normalized by image height  (clipped to (0, 2])
    """
    y_safe = max(float(y_v_norm), 1e-3)
    v_safe = max(float(v), 1e-3)
    return float(np.clip(
        float(width_norm) ** 2 / (2.0 * v_safe ** 2 * y_safe),
        1e-4, 2.0,
    ))


def v_from_annotation(y_v_norm: float, width_norm: float, height_norm: float) -> float:
    """
    Inverse of height_from_physics: compute ground-truth v from annotation.

        v = sqrt(width^2 / (2 * height * y_vertex))
    """
    y_safe = max(float(y_v_norm), 1e-3)
    h_safe = max(float(height_norm), 1e-3)
    w_safe = max(float(width_norm), 1e-3)
    return float(np.clip(
        math.sqrt(w_safe ** 2 / (2.0 * h_safe * y_safe)),
        0.05, 50.0,
    ))


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
    width = max(float(width), 2.0)
    height = max(float(height), 1.0)
    thickness = max(float(thickness), 1.0)

    half_w = width / 2.0
    x_left = x_v - half_w
    x_right = x_v + half_w

    upper_pts, lower_pts = [], []
    n_points = max(40, int(round(width)))

    for i in range(n_points + 1):
        t = i / max(n_points, 1)
        x = x_left + (x_right - x_left) * t
        dx = (x - x_v) / (half_w + 1e-6)
        y_center = y_v + height * (dx ** 2)
        upper_pts.append((x, y_center - thickness / 2.0))
        lower_pts.append((x, y_center + thickness / 2.0))

    poly = upper_pts + list(reversed(lower_pts))
    poly_np = np.array(poly, dtype=np.float32)
    poly_np[:, 0] = np.clip(poly_np[:, 0], 0, w - 1)
    poly_np[:, 1] = np.clip(poly_np[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly_np).astype(np.int32)], 1)
    return mask.astype(np.float32)


def overlay_mask_on_image(img_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    red = np.array([0.0, 0.0, 255.0], dtype=np.float32)
    alpha = 0.45 * np.clip(mask, 0, 1)[..., None]
    overlay = img_color * (1.0 - alpha) + red * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def detection_to_mask(det: Dict, input_size: Tuple[int, int]) -> np.ndarray:
    input_h, input_w = input_size
    return rasterize_hyperbola_band_mask(
        input_h, input_w,
        float(det["x_vertex"]), float(det["y_vertex"]),
        float(det["width"]),    float(det["height"]),
        float(det["thickness"]),
    )


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def remove_small_mask_components(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    h, w = mask_u8.shape
    min_area_eff = max(int(min_area), int(round(0.0005 * h * w)))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for label_id in range(1, num_labels):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area_eff:
            cleaned[labels == label_id] = 1
    return cleaned.astype(np.float32)


# =========================================================
# 3. 编码 / 解码参数
# =========================================================

def encode_hyperbola_params(
    obj: Dict,
    orig_size: Tuple[int, int],
    input_size: Tuple[int, int],
) -> Dict[str, float]:
    """
    Convert raw annotation to normalized (x_v, y_v, width, v, thickness).
    height is NOT stored; v is computed via v_from_annotation.
    normalize_param is always True here.
    """
    orig_h, orig_w = orig_size
    input_h, input_w = input_size

    x_v     = float(obj["x_vertex"])  / orig_w * input_w / input_w
    y_v     = float(obj["y_vertex"])  / orig_h * input_h / input_h
    width   = float(obj["width"])     / orig_w * input_w / input_w
    height  = float(obj["height"])    / orig_h * input_h / input_h
    thickness = float(obj["thickness"]) / orig_h * input_h / input_h

    v = v_from_annotation(y_v, width, height)

    return {
        "x_vertex":  float(np.clip(x_v, 0.0, 1.0)),
        "y_vertex":  float(np.clip(y_v, 0.0, 1.0)),
        "width":     float(np.clip(width, 0.0, 1.0)),
        "v":         v,
        "thickness": float(np.clip(thickness, 0.0, 1.0)),
    }


def decode_param_vector(
    param_vector: np.ndarray,
    input_size: Tuple[int, int],
    score: float = 1.0,
) -> Dict[str, float]:
    """
    Convert network output to a detection dict with pixel-space coordinates.
    Height is derived from physics (not directly predicted).
    """
    input_h, input_w = input_size
    values = np.asarray(param_vector, dtype=np.float32).reshape(-1)

    x_v_norm, y_v_norm, width_norm, v, thickness_norm = [float(val) for val in values]

    # De-normalize spatial params
    x_v     = float(np.clip(x_v_norm * input_w, 0.0, input_w - 1))
    y_v     = float(np.clip(y_v_norm * input_h, 0.0, input_h - 1))
    width   = float(max(width_norm * input_w, 2.0))
    thickness = float(max(thickness_norm * input_h, 1.0))
    v       = float(max(v, 1e-3))

    # Physics: derive height
    height_norm = height_from_physics(y_v_norm, width_norm, v)
    height = float(max(height_norm * input_h, 1.0))

    return {
        "score":     float(score),
        "x_vertex":  x_v,
        "y_vertex":  y_v,
        "width":     width,
        "height":    height,
        "thickness": thickness,
        "v":         v,
    }


def select_primary_hyperbola(
    objs: List[Dict],
    orig_size: Tuple[int, int],
    input_size: Tuple[int, int],
):
    """Select the largest hyperbola (by mask area) as training target."""
    primary_params = None
    primary_area = -1.0

    for obj in objs:
        if obj.get("label", "") != "hyperbola":
            continue

        # Pixel-space params for area computation
        orig_h, orig_w = orig_size
        input_h, input_w = input_size
        px = {
            "x_vertex":  float(obj["x_vertex"])  / orig_w * input_w,
            "y_vertex":  float(obj["y_vertex"])  / orig_h * input_h,
            "width":     float(obj["width"])     / orig_w * input_w,
            "height":    float(obj["height"])    / orig_h * input_h,
            "thickness": float(obj["thickness"]) / orig_h * input_h,
        }
        band_mask = rasterize_hyperbola_band_mask(
            input_h, input_w,
            px["x_vertex"], px["y_vertex"],
            px["width"], px["height"], px["thickness"],
        )
        area = float(band_mask.sum())
        if area > primary_area:
            primary_area = area
            primary_params = encode_hyperbola_params(obj, orig_size, input_size)

    return primary_params


# =========================================================
# 4. 数据集
# =========================================================

class HyperbolaDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_json: str,
        input_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.image_dir = image_dir
        self.input_h, self.input_w = input_size

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
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img, orig_w, orig_h = self._load_image(image_path)
        objs = self.ann_dict[image_name]

        # Pixel-space objects for meta
        meta_objs = []
        for obj in objs:
            if obj.get("label", "") != "hyperbola":
                continue
            px_x = float(obj["x_vertex"]) / orig_w * self.input_w
            px_y = float(obj["y_vertex"]) / orig_h * self.input_h
            px_w = float(obj["width"])    / orig_w * self.input_w
            px_h = float(obj["height"])   / orig_h * self.input_h
            px_t = float(obj["thickness"])/ orig_h * self.input_h
            meta_objs.append({
                "x_vertex": px_x, "y_vertex": px_y,
                "width": px_w, "height": px_h, "thickness": px_t,
            })

        primary = select_primary_hyperbola(
            objs,
            orig_size=(orig_h, orig_w),
            input_size=(self.input_h, self.input_w),
        )

        if primary is None:
            param_target = np.zeros(len(PARAM_KEYS), dtype=np.float32)
            has_hyperbola = np.array([0.0], dtype=np.float32)
        else:
            param_target = np.array([primary[k] for k in PARAM_KEYS], dtype=np.float32)
            has_hyperbola = np.array([1.0], dtype=np.float32)

        meta = {
            "image_name": image_name,
            "image_path": image_path,
            "objects": meta_objs,
            "orig_size": (orig_h, orig_w),
            "resized_size": (self.input_h, self.input_w),
        }

        return (
            torch.from_numpy(img).unsqueeze(0).float(),
            torch.from_numpy(param_target).float(),
            torch.from_numpy(has_hyperbola).float(),
            meta,
        )


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
    """
    Outputs:
      objectness_logit : [B, 1]
      param_raw        : [B, 5]  ->  (x_v, y_v, w, log_v, t)
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.objectness_head = nn.Linear(base_ch * 4, 1)
        self.param_head = nn.Linear(base_ch * 4, len(PARAM_KEYS))

    def forward(self, x):
        _, x = self.down1(x)
        _, x = self.down2(x)
        _, x = self.down3(x)
        x = self.bottleneck(x)
        feat = self.mlp(self.pool(x))

        objectness = self.objectness_head(feat)
        raw = self.param_head(feat)                  # [B, 5]

        # x_v, y_v, width, thickness -> sigmoid -> [0, 1]
        # v (index 3)                -> softplus -> (0, +inf)
        spatial = torch.sigmoid(torch.cat([raw[:, :3], raw[:, 4:5]], dim=1))  # [B, 4]
        v = F.softplus(raw[:, 3:4])                                           # [B, 1]

        # Reassemble in PARAM_KEYS order: (x_v, y_v, w, v, t)
        params = torch.cat([spatial[:, :3], v, spatial[:, 3:4]], dim=1)       # [B, 5]
        return objectness, params


# =========================================================
# 6. Loss
# =========================================================

def masked_param_loss(pred, target, has_hyperbola):
    """Smooth L1 on (x_v, y_v, w, v, t), only for images with a hyperbola."""
    valid = has_hyperbola.view(-1, 1)
    if valid.sum().item() == 0:
        return pred.sum() * 0.0
    loss = F.smooth_l1_loss(pred, target, reduction="none") * valid
    return loss.sum() / (valid.sum() * pred.shape[1] + 1e-6)


def hyperbola_collate_fn(batch):
    images, gt_params, has_hyp, metas = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.stack(gt_params, dim=0),
        torch.stack(has_hyp, dim=0),
        list(metas),
    )


# =========================================================
# 7. 训练 / 验证
# =========================================================

def train_one_epoch(model, loader, optimizer, device, obj_loss_fn, lambda_size=1.0):
    model.train()
    total = 0.0
    for images, gt_params, has_hyp, _ in loader:
        images    = images.to(device)
        gt_params = gt_params.to(device)
        has_hyp   = has_hyp.to(device)

        pred_obj, pred_params = model(images)
        loss_obj   = obj_loss_fn(pred_obj, has_hyp)
        loss_param = masked_param_loss(pred_params, gt_params, has_hyp)
        loss = loss_obj + lambda_size * loss_param

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device, obj_loss_fn, lambda_size=1.0):
    model.eval()
    total = 0.0
    for images, gt_params, has_hyp, _ in loader:
        images    = images.to(device)
        gt_params = gt_params.to(device)
        has_hyp   = has_hyp.to(device)

        pred_obj, pred_params = model(images)
        loss_obj   = obj_loss_fn(pred_obj, has_hyp)
        loss_param = masked_param_loss(pred_params, gt_params, has_hyp)
        total += (loss_obj + lambda_size * loss_param).item()
    return total / max(len(loader), 1)


# =========================================================
# 8. 推理
# =========================================================

def predict_single_image(model, image_path, input_size, device, obj_thresh=0.5):
    input_h, input_w = input_size

    img = Image.open(image_path).convert("L")
    img_resized = img.resize((input_w, input_h), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        pred_obj, pred_params = model(x)

    score = float(torch.sigmoid(pred_obj[0, 0]).item())
    detections = []
    pred_mask  = np.zeros((input_h, input_w), dtype=np.float32)

    if score >= obj_thresh:
        det = decode_param_vector(
            pred_params[0].cpu().numpy(),
            input_size=input_size,
            score=score,
        )
        detections = [det]
        pred_mask  = detection_to_mask(det, input_size)

    return np.array(img_resized), pred_mask, detections


# =========================================================
# 9. 可视化
# =========================================================

def draw_hyperbola_on_image(img_gray, detections, color=(0, 255, 0)):
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    H, W = img_gray.shape

    for det in detections:
        x_v   = float(det["x_vertex"])
        y_v   = float(det["y_vertex"])
        width = max(float(det["width"]), 2.0)
        height= max(float(det["height"]), 1.0)
        thickness = max(float(det["thickness"]), 1.0)
        score = det.get("score", None)
        v     = det.get("v", None)

        half_w = width / 2.0
        upper_pts, lower_pts, center_pts = [], [], []
        n_points = max(40, int(round(width)))
        for i in range(n_points + 1):
            t = i / max(n_points, 1)
            x = (x_v - half_w) + 2 * half_w * t
            dx = (x - x_v) / (half_w + 1e-6)
            y_c = y_v + height * dx ** 2
            upper_pts.append((x, y_c - thickness / 2.0))
            lower_pts.append((x, y_c + thickness / 2.0))
            center_pts.append((x, y_c))

        poly_np = np.array(upper_pts + list(reversed(lower_pts)), dtype=np.float32)
        center_np = np.array(center_pts, dtype=np.float32)
        poly_np[:, 0] = np.clip(poly_np[:, 0], 0, W - 1)
        poly_np[:, 1] = np.clip(poly_np[:, 1], 0, H - 1)
        center_np[:, 0] = np.clip(center_np[:, 0], 0, W - 1)
        center_np[:, 1] = np.clip(center_np[:, 1], 0, H - 1)

        overlay = img.copy()
        cv2.fillPoly(overlay, [np.round(poly_np).astype(np.int32)], color)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.polylines(img, [np.round(center_np).astype(np.int32)],
                      isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(round(x_v)), int(round(y_v))), 3, (0, 0, 255), -1)

        label_parts = []
        if score is not None:
            label_parts.append(f"s={score:.2f}")
        if v is not None:
            label_parts.append(f"v={v:.2f}")
        if label_parts:
            cv2.putText(
                img, " ".join(label_parts),
                (int(round(x_v)) + 3, int(round(y_v)) - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA,
            )
    return img


def save_prediction_visualization(model, dataset, indices, out_dir, device,
                                   input_size, obj_thresh=0.5):
    ensure_dir(out_dir)
    for idx in indices:
        _, _, _, meta = dataset[idx]
        image_name = meta["image_name"]
        image_path = meta["image_path"]

        img_gray, pred_mask, pred_dets = predict_single_image(
            model, image_path, input_size, device, obj_thresh=obj_thresh,
        )

        gt_dets = meta["objects"]  # already pixel-space, include height

        gt_img   = draw_hyperbola_on_image(img_gray, gt_dets,   color=(0, 255, 255))
        pred_img = draw_hyperbola_on_image(img_gray, pred_dets, color=(0, 255, 0))
        mask_img = overlay_mask_on_image(img_gray, pred_mask)

        # Format param text
        def fmt(tag, dets):
            lines = []
            for i, d in enumerate(dets[:3]):
                v_str = f" v={d['v']:.2f}" if "v" in d else ""
                lines.append(
                    f"{tag}[{i}] x={d['x_vertex']:.1f} y={d['y_vertex']:.1f} "
                    f"w={d['width']:.1f} h={d['height']:.1f} t={d['thickness']:.1f}"
                    f"{v_str}"
                )
            return lines or [f"{tag}: none"]

        fig = plt.figure(figsize=(20, 7))
        for sp, im_data, title in [
            (1, img_gray,  f"Input: {image_name}"),
            (2, cv2.cvtColor(gt_img,   cv2.COLOR_BGR2RGB), "GT"),
            (3, cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB), "Pred Mask"),
            (4, cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB), "Pred (physics height)"),
        ]:
            plt.subplot(1, 4, sp)
            plt.imshow(im_data, cmap="gray" if sp == 1 else None)
            plt.title(title)
            plt.axis("off")

        bottom_text = "\n".join(fmt("GT", gt_dets) + fmt("Pred", pred_dets))
        fig.text(0.01, 0.01, bottom_text, ha="left", va="bottom",
                 fontsize=8, family="monospace")

        save_path = os.path.join(out_dir, image_name.replace(".jpg", "_pred.png"))
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


# =========================================================
# 10. 评估
# =========================================================

def build_gt_mask_from_meta(meta, input_size):
    input_h, input_w = input_size
    gt_mask = np.zeros((input_h, input_w), dtype=np.float32)
    for obj in meta["objects"]:
        gt_mask = np.maximum(gt_mask, detection_to_mask(obj, input_size))
    return gt_mask


@torch.no_grad()
def evaluate_mask_overlap(model, dataset, device, input_size, obj_thresh=0.5):
    total_inter = total_union = total_pred = total_gt = 0.0
    per_iou = []

    for idx in range(len(dataset)):
        _, _, _, meta = dataset[idx]
        _, pred_mask, _ = predict_single_image(
            model, meta["image_path"], input_size, device, obj_thresh=obj_thresh,
        )

        pred_mask = remove_small_mask_components(pred_mask, min_area=mask_min_area)
        gt_mask   = build_gt_mask_from_meta(meta, input_size)

        pred_bin = pred_mask > 0
        gt_bin   = gt_mask > 0

        inter = float(np.logical_and(pred_bin, gt_bin).sum())
        union = float(np.logical_or(pred_bin, gt_bin).sum())
        total_inter += inter
        total_union += union
        total_pred  += float(pred_bin.sum())
        total_gt    += float(gt_bin.sum())
        per_iou.append(inter / max(union, 1e-6))

    pixel_p = total_inter / max(total_pred, 1e-6)
    pixel_r = total_inter / max(total_gt,   1e-6)
    pixel_f1 = 2 * pixel_p * pixel_r / max(pixel_p + pixel_r, 1e-6)

    return {
        "num_test_images": len(dataset),
        "global_iou":      total_inter / max(total_union, 1e-6),
        "global_overlap":  total_inter / max(total_gt,    1e-6),
        "pixel_precision": pixel_p,
        "pixel_recall":    pixel_r,
        "pixel_f1":        pixel_f1,
        "mean_image_iou":  float(np.mean(per_iou)) if per_iou else 0.0,
    }


@torch.no_grad()
def evaluate_param_error(model, dataset, device, input_size, obj_thresh=0.5):
    """Compute MAE on v (and derived height) for matched predictions."""
    err_v = err_h = 0.0
    n_matched = 0

    for idx in range(len(dataset)):
        _, _, _, meta = dataset[idx]
        _, _, pred_dets = predict_single_image(
            model, meta["image_path"], input_size, device, obj_thresh=obj_thresh,
        )

        gt_dets = meta["objects"]
        if not pred_dets or not gt_dets:
            continue

        pred = pred_dets[0]
        gt   = gt_dets[0]

        # GT v from pixel-space annotation (need normalized quantities)
        ih, iw = input_size
        gt_y_norm = gt["y_vertex"] / ih
        gt_w_norm = gt["width"]    / iw
        gt_h_norm = gt["height"]   / ih
        gt_v = v_from_annotation(gt_y_norm, gt_w_norm, gt_h_norm)

        err_v += abs(pred.get("v", 0.0) - gt_v)
        err_h += abs(pred["height"] - gt["height"])
        n_matched += 1

    return {
        "matched": n_matched,
        "mae_v":   err_v  / max(n_matched, 1),
        "mae_height_derived": err_h / max(n_matched, 1),
    }


def save_results_csv(rows, csv_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results_table(rows):
    if not rows:
        return
    keys = ["seed", "lambda", "global_iou", "global_overlap",
            "pixel_f1", "mae_v", "mae_height_derived"]
    header = " | ".join(f"{k:>18}" for k in keys)
    print("\nTest Metrics")
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = [
            str(r.get("seed", "")),
            str(r.get("lambda_size", "")),
            f"{r.get('global_iou', 0):.4f}",
            f"{r.get('global_overlap', 0):.4f}",
            f"{r.get('pixel_f1', 0):.4f}",
            f"{r.get('mae_v', 0):.4f}",
            f"{r.get('mae_height_derived', 0):.2f}",
        ]
        print(" | ".join(f"{v:>18}" for v in vals))


# =========================================================
# 11. 主程序
# =========================================================

def main():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    now = datetime.now()
    work_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{script_name}_{now.strftime('%m%d_%H%M')}",
    )
    ensure_dir(work_dir)

    print("Using device:", device)

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.6f}")

    # Build dataset
    raw_datasets = [
        HyperbolaDataset(
            image_dir=src["image_dir"],
            annotation_json=src["annotation_json"],
            input_size=input_size,
        )
        for src in data_sources
    ]
    dataset = raw_datasets[0] if len(raw_datasets) == 1 else ConcatDataset(raw_datasets)

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    n_test  = n_total - n_train - n_val
    print(f"Total: {n_total}  Train: {n_train}  Val: {n_val}  Test: {n_test}")

    all_results = []

    for seed in seed_list:
        set_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(
            dataset, [n_train, n_val, n_test], generator=gen,
        )

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=0, collate_fn=hyperbola_collate_fn)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=0, collate_fn=hyperbola_collate_fn)

        seed_dir = os.path.join(work_dir, f"seed_{seed:02d}")
        ensure_dir(seed_dir)

        print(f"\n{'#'*60}\nSeed {seed}\n{'#'*60}")

        for lambda_size in lambda_size_list:
            lambda_tag = str(lambda_size).replace(".", "p")
            exp_dir  = os.path.join(seed_dir, f"lambda_{lambda_tag}")
            ckpt_dir = os.path.join(exp_dir, "checkpoints")
            vis_dir  = os.path.join(exp_dir, "visuals")
            ensure_dir(ckpt_dir)
            ensure_dir(vis_dir)

            print(f"\n{'='*60}")
            print(f"seed={seed}  lambda={lambda_size}")
            print(f"{'='*60}")

            model = HyperbolaNet(in_ch=1, base_ch=32).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            obj_loss_fn = nn.BCEWithLogitsLoss()

            best_val = float("inf")

            for epoch in range(1, num_epochs + 1):
                tr_loss = train_one_epoch(model, train_loader, optimizer, device,
                                          obj_loss_fn, lambda_size)
                va_loss = validate_one_epoch(model, val_loader, device,
                                             obj_loss_fn, lambda_size)
                print(f"  [Epoch {epoch:03d}] train={tr_loss:.6f}  val={va_loss:.6f}")

                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
                if va_loss < best_val:
                    best_val = va_loss
                    torch.save(model.state_dict(),
                               os.path.join(ckpt_dir, "best_model.pth"))
                    print(f"    -> best model saved")

            model.load_state_dict(torch.load(
                os.path.join(ckpt_dir, "best_model.pth"), map_location=device,
            ))

            save_prediction_visualization(
                model, test_set, list(range(len(test_set))),
                vis_dir, device, input_size, obj_thresh=hm_thresh,
            )

            mask_metrics  = evaluate_mask_overlap(
                model, test_set, device, input_size, obj_thresh=hm_thresh,
            )
            param_metrics = evaluate_param_error(
                model, test_set, device, input_size, obj_thresh=hm_thresh,
            )

            row = {"seed": seed, "lambda_size": lambda_size}
            row.update(mask_metrics)
            row.update(param_metrics)
            all_results.append(row)

            print(
                f"  IoU={mask_metrics['global_iou']:.4f}  "
                f"Overlap={mask_metrics['global_overlap']:.4f}  "
                f"F1={mask_metrics['pixel_f1']:.4f}  "
                f"MAE_v={param_metrics['mae_v']:.4f}  "
                f"MAE_h={param_metrics['mae_height_derived']:.2f}"
            )

    print_results_table(all_results)
    csv_path = os.path.join(work_dir, "test_metrics_summary.csv")
    save_results_csv(all_results, csv_path)
    print(f"\nSaved summary to: {csv_path}")


if __name__ == "__main__":
    main()
