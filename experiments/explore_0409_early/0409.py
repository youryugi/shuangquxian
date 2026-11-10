import os
import json
import math
import random
from typing import List, Dict, Tuple
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


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
      - heatmap: [1, H, W]
      - size map: [3, H, W]  -> width, height, thickness
      - reg mask: [1, H, W]  -> 只有顶点位置为1，其余为0
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

            x_i = int(round(x_v))
            y_i = int(round(y_v))

            if not (0 <= x_i < self.input_w and 0 <= y_i < self.input_h):
                continue

            draw_gaussian(heatmap, (x_i, y_i), self.sigma)

            reg_mask[0, y_i, x_i] = 1.0

            if self.normalize_param:
                size_map[0, y_i, x_i] = width / self.input_w
                size_map[1, y_i, x_i] = height / self.input_h
                size_map[2, y_i, x_i] = thickness / self.input_h
            else:
                size_map[0, y_i, x_i] = width
                size_map[1, y_i, x_i] = height
                size_map[2, y_i, x_i] = thickness

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
                         hm_thresh=0.35, max_det=20):
    input_h, input_w = input_size

    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    img_resized = img.resize((input_w, input_h), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        pred_hm, pred_size = model(x)

    pred_hm = pred_hm[0, 0].cpu()
    pred_size = pred_size[0].cpu()  # [3,H,W]

    peaks = local_peak_extract(pred_hm, thresh=hm_thresh, max_det=max_det)

    detections = []
    for score, x_i, y_i in peaks:
        w = pred_size[0, y_i, x_i].item()
        h = pred_size[1, y_i, x_i].item()
        t = pred_size[2, y_i, x_i].item()

        if normalize_param:
            w = w * input_w
            h = h * input_h
            t = t * input_h

        detections.append({
            "score": score,
            "x_vertex": float(x_i),
            "y_vertex": float(y_i),
            "width": float(w),
            "height": float(h),
            "thickness": float(t),
        })

    return np.array(img_resized), pred_hm.numpy(), detections


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
        x_v = det["x_vertex"]
        y_v = det["y_vertex"]
        width = max(det["width"], 2.0)
        height = max(det["height"], 1.0)
        thickness = max(det["thickness"], 1.0)
        score = det["score"]

        x_left = int(max(0, round(x_v - width / 2)))
        x_right = int(min(W - 1, round(x_v + width / 2)))

        pts = []
        for x in range(x_left, x_right + 1):
            dx = (x - x_v) / (width / 2 + 1e-6)
            y = y_v + height * (dx ** 2)
            y = int(round(y))
            if 0 <= y < H:
                pts.append((x, y))

        # 画中心曲线
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, max(1, int(round(thickness / 8))))

        # 画顶点
        cv2.circle(img, (int(round(x_v)), int(round(y_v))), 3, (0, 0, 255), -1)

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
                                  hm_thresh=0.35, max_det=20):
    ensure_dir(out_dir)

    for idx in indices:
        image_tensor, gt_hm, gt_size, reg_mask, meta = dataset[idx]
        image_name = meta["image_name"]
        image_path = os.path.join(dataset.image_dir, image_name)

        img_gray, pred_hm, detections = predict_single_image(
            model=model,
            image_path=image_path,
            input_size=(dataset.input_h, dataset.input_w),
            device=device,
            normalize_param=dataset.normalize_param,
            hm_thresh=hm_thresh,
            max_det=max_det
        )

        pred_overlay = overlay_heatmap_on_image(img_gray, pred_hm)
        draw_img = draw_hyperbola_on_image(img_gray, detections)

        gt_overlay = overlay_heatmap_on_image(
            img_gray,
            gt_hm.squeeze(0).numpy()
        )

        fig = plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_gray, cmap="gray")
        plt.title(f"Input: {image_name}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
        plt.title("GT Heatmap")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
        plt.title("Prediction")
        plt.axis("off")

        save_path = os.path.join(out_dir, image_name.replace(".jpg", "_pred.png"))
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


# =========================================================
# 7. 主程序
# =========================================================

def main():
    # -----------------------
    # 路径设置
    # -----------------------
    image_dir = r"C:/Users/79152/Desktop/github/shuangquxian/biaozhu/Utilities"                 # 图片文件夹
    annotation_json = r"C:/Users/79152/Desktop/github/shuangquxian/biaozhu/Utilities/annotations.json" # JSON 标注文件

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    now = datetime.now()
    time_tag = now.strftime("%m%d_%H%M")
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{script_name}_{time_tag}")
    ensure_dir(work_dir)
    work_dir = "./runs_hyperbola"
    ckpt_dir = os.path.join(work_dir, "checkpoints")
    vis_dir = os.path.join(work_dir, "visuals")
    ensure_dir(work_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(vis_dir)

    # -----------------------
    # 超参数
    # -----------------------
    set_seed(42)

    input_size = (256, 256)
    sigma = 3.0

    batch_size = 8
    num_epochs = 50
    lr = 1e-3
    train_ratio = 0.8

    lambda_size = 2
    normalize_param = True

    hm_thresh = 0.35
    max_det = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------
    # 数据
    # -----------------------
    dataset = HyperbolaDataset(
        image_dir=image_dir,
        annotation_json=annotation_json,
        input_size=input_size,
        sigma=sigma,
        normalize_param=normalize_param
    )

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=hyperbola_collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=hyperbola_collate_fn
    )

    print(f"Total samples: {n_total}, Train: {n_train}, Val: {n_val}")

    # -----------------------
    # 模型
    # -----------------------
    model = HyperbolaNet(in_ch=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hm_loss_fn = FocalHeatmapLoss()

    best_val = 1e9

    # -----------------------
    # 训练
    # -----------------------
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, hm_loss_fn, lambda_size=lambda_size
        )
        val_loss = validate_one_epoch(
            model, val_loader, device, hm_loss_fn, lambda_size=lambda_size
        )

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model to {best_path}")

    # -----------------------
    # 可视化若干验证样本
    # -----------------------
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))
    print("Loaded best model.")

    # 从原始 dataset 中选一些样本做可视化
    sample_indices = list(range(min(10, len(dataset))))
    save_prediction_visualization(
        model=model,
        dataset=dataset,
        indices=sample_indices,
        out_dir=vis_dir,
        device=device,
        hm_thresh=hm_thresh,
        max_det=max_det
    )

    print(f"Visualization saved to: {vis_dir}")


if __name__ == "__main__":
    main()