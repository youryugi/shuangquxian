"""
自己完结・单文件版（通用注意力模块对比版）：
  YOLO 网格(anchor-free) 矩形框检测头 + **通用即插即用注意力模块**（无监督，不用 band mask 监督）。
  本文件由 attn_cnn_yolo_final.py 改造而来，**核心区别是把「band mask 显式监督的注意力(abs/soft)」
  换成一组通用注意力模块**，用 --attn_blocks 选择，作为「显式监督注意力」的对照基线。
  其余（YOLO 风格检测、数据集、训练、评估、划分）全部保持不变，保证与 band 监督版/backbone 版可比。

【注意力模块插在哪】
  插在 bottleneck 之后、检测 head 之前（stride8 检测特征上）。模块对特征做重标定，无需任何额外监督。
  none = 不插模块(纯检测基线，等价 nn.Identity)。

【四种通用注意力（--attn_blocks 选择，可多选做对比实验）】
  se       : SE 通道注意力（Squeeze-and-Excitation，Hu 2018）——全局池化→MLP→通道重加权。
  cbam     : CBAM（Woo 2018）——通道注意力 + 空间注意力 串联。
  nonlocal : Non-local 空间自注意力（Wang 2018，embedded Gaussian，含 subsample 省显存）。
  coord    : Coordinate Attention（Hou 2021）——沿 H/W 分别池化，编码位置信息的注意力。
  （均为无监督，不读 band；band 仍由 dataset 产出但本版本不使用。）

【检测风格：YOLO 网格 anchor-free】
  - 监督：objectness 硬目标(GT 中心单元=1，其余=0)，obj/noobj 平衡 BCE。
  - 解码：sigmoid(objectness) ≥ 阈值 → 解码框 → 框级 IoU-NMS。
  - 直接用最终 epoch 的模型评估（不挑 best、不回滚），训练结束保存 {tag}_final.pth。
"""
import os
import re
import csv
import json
import random
import argparse
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

# 严格确定性：CUBLAS_WORKSPACE_CONFIG 必须在 import torch（初始化 cuBLAS）之前设置
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset


# ══════════════════════════════════════════════════════════════════════════════
# 超参数（可调）
# ══════════════════════════════════════════════════════════════════════════════
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))   # ml/0409 -> ml -> repo

IMG_DIR   = os.path.join(_REPO_ROOT, "dataset3", "images-selected")
HYP_JSON  = os.path.join(IMG_DIR, "annotations.json")
RECT_JSON = os.path.join(IMG_DIR, "annotations_rect.json")

# —— 实验配置（直接改这几个；命令行 --attn_blocks / --seeds / --augment 可覆盖）——
ATTN_BLOCKS  = ["se", "cbam", "nonlocal", "coord"]   # 要对比的注意力模块（none=纯检测基线）
SEEDS        = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]      # 随机种子
TRAIN_FRACS  = [0.80]                     # 训练集占比扫描列表；命令行 --train_fracs 可覆盖
AUGMENT      = False                     # 训练集数据增强开关（仅水平翻转）
RUN_VAL      = False                      # 是否每个 epoch 跑一遍 val（仅打印监控），默认关、省时

input_size   = (640, 640)
HM_STRIDE    = 8                    # heatmap 下采样步长
ATTN_STRIDE  = 4                    # 注意力图下采样步长（dataset 仍按它产 band，本版本不使用）
HM_SIGMA     = 6                  # heatmap 高斯半径
batch_size   = 8
NUM_WORKERS  = min(os.cpu_count() or 1, 8)   # DataLoader 数据加载进程数；设 0 关闭多进程加载
num_epochs   = 80
LR           = 5e-4                 # Adam 学习率
nms_kernel   = 3                   # （YOLO 风格不再用热图 maxpool-NMS，此项无效，保留兼容）
max_det      = 5
HM_THRESH    = 0.9                  # YOLO：objectness 解码阈值
BBOX_IOU_THR = 0.5                 # 评测匹配 IoU 阈值
NMS_IOU_THR  = 0.5                 # YOLO 解码：框级 IoU-NMS 阈值
LAM_NOOBJ    = 0.5                 # YOLO：objectness 损失中 noobj(背景) 项权重
BASE_CH      = 32                   # 网络通道基数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 基础工具
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 严格确定性：强制所有算子走确定性实现（遇到没有确定性实现的算子会直接报错）
    torch.use_deterministic_algorithms(True)


def _worker_init(worker_id):
    """DataLoader worker 初始化：关掉 OpenCV 内部多线程，并按 worker 重新播种保持可复现。"""
    cv2.setNumThreads(0)
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def make_split(n_total, seed, train_frac=0.70, val_frac=0.0):
    """按种子随机划分 train/val/test（默认 70/0/30：不留 val，其余为 test）。"""
    idx = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(round(n_total * train_frac))
    n_val   = int(round(n_total * val_frac))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def render_gaussian(heatmap, cx, cy, sigma):
    hm_h, hm_w = heatmap.shape
    ys = np.arange(hm_h, dtype=np.float32)[:, None]
    xs = np.arange(hm_w, dtype=np.float32)[None, :]
    g  = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    np.maximum(heatmap, g, out=heatmap)


def rasterize_hyperbola_band_mask(h, w, x_v, y_v, width, height, thickness):
    """与标注工具一致：thickness 为竖直厚度（沿 y 方向偏移 ±thickness/2）。"""
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


def nms_iou(boxes, scores, iou_thr):
    """YOLO 风格框级 NMS：按分数降序，抑制与已保留框 IoU ≥ 阈值的框。返回保留的索引。"""
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if bbox_iou(boxes[i], boxes[j]) < iou_thr]
    return keep


# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据集
# ══════════════════════════════════════════════════════════════════════════════
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
            render_gaussian(hm, fx, fy, self.sigma)
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
            band_full = np.maximum(band_full, rasterize_hyperbola_band_mask(
                self.input_h, self.input_w, mo["x_vertex"], mo["y_vertex"],
                mo["width"], mo["height"], mo["thickness"]))
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


def augment_sample(img, hm, wh, off, peak, band):
    """训练集增强：仅水平翻转（标签同步翻转，x 偏移取负）。仅作用于训练样本。"""
    if random.random() < 0.5:
        img  = torch.flip(img,  dims=[2])
        hm   = torch.flip(hm,   dims=[2])
        wh   = torch.flip(wh,   dims=[2])
        peak = torch.flip(peak, dims=[2])
        band = torch.flip(band, dims=[2])
        off  = torch.flip(off,  dims=[2])
        off[0] = -off[0]                       # 水平翻转后 x 方向亚像素偏移取负
    return img, hm, wh, off, peak, band


class AugWrapper(Dataset):
    """只包训练集：每次取样随机做 augment_sample（meta 不变，仅训练 loss 用张量）。"""
    def __init__(self, subset):
        self.subset = subset
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, hm, wh, off, peak, band, meta = self.subset[idx]
        img, hm, wh, off, peak, band = augment_sample(img, hm, wh, off, peak, band)
        return img, hm, wh, off, peak, band, meta


# ══════════════════════════════════════════════════════════════════════════════
# 3. 通用注意力模块（无监督即插即用；均用确定性安全的算子，兼容 use_deterministic_algorithms）
# ══════════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    """SE 通道注意力（Hu et al. 2018）：全局平均池化 → 两层 1x1 MLP → sigmoid 通道权重。"""
    def __init__(self, ch, r=16):
        super().__init__()
        h = max(ch // r, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, ch, 1))

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)            # 全局平均（确定性 reduction）
        return x * torch.sigmoid(self.fc(s))


class _ChannelAttn(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        h = max(ch // r, 4)
        self.mlp = nn.Sequential(nn.Conv2d(ch, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, ch, 1))

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx  = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class _SpatialAttn(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a


class CBAM(nn.Module):
    """CBAM（Woo et al. 2018）：通道注意力 + 空间注意力 串联。"""
    def __init__(self, ch, r=16, k=7):
        super().__init__()
        self.ca = _ChannelAttn(ch, r)
        self.sa = _SpatialAttn(k)

    def forward(self, x):
        return self.sa(self.ca(x))


class NonLocalBlock(nn.Module):
    """Non-local 空间自注意力（Wang et al. 2018，embedded Gaussian）。
    phi/g 后接 maxpool 子采样（原论文 subsample trick），把 attn 矩阵从 HW×HW 降到 HW×(HW/sub^2) 省显存。
    out 卷积零初始化 → 残差起点为恒等映射，训练更稳。"""
    def __init__(self, ch, sub=2):
        super().__init__()
        inter = max(ch // 2, 1)
        self.inter = inter
        self.theta = nn.Conv2d(ch, inter, 1)
        self.phi   = nn.Conv2d(ch, inter, 1)
        self.g     = nn.Conv2d(ch, inter, 1)
        self.pool  = nn.MaxPool2d(sub) if sub > 1 else nn.Identity()
        self.out   = nn.Conv2d(inter, ch, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        theta = self.theta(x).reshape(b, self.inter, h * w).permute(0, 2, 1)   # b, HW, inter
        phi   = self.pool(self.phi(x)).reshape(b, self.inter, -1)              # b, inter, HW'
        attn  = torch.softmax(torch.bmm(theta, phi) / (self.inter ** 0.5), dim=-1)  # b, HW, HW'
        g     = self.pool(self.g(x)).reshape(b, self.inter, -1).permute(0, 2, 1)    # b, HW', inter
        y     = torch.bmm(attn, g).permute(0, 2, 1).reshape(b, self.inter, h, w)
        return x + self.out(y)


class CoordAttn(nn.Module):
    """Coordinate Attention（Hou et al. 2021）：沿 H、W 方向分别池化，编码位置信息后做注意力。"""
    def __init__(self, ch, r=32):
        super().__init__()
        h = max(ch // r, 8)
        self.conv1  = nn.Conv2d(ch, h, 1)
        self.bn1    = nn.BatchNorm2d(h)
        self.act    = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(h, ch, 1)
        self.conv_w = nn.Conv2d(h, ch, 1)

    def forward(self, x):
        b, c, H, W = x.shape
        x_h = x.mean(dim=3, keepdim=True)                       # b, c, H, 1（沿宽度池化）
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)   # b, c, W, 1（沿高度池化）
        y = torch.cat([x_h, x_w], dim=2)                        # b, c, H+W, 1
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                           # b, h, 1, W
        a_h = torch.sigmoid(self.conv_h(x_h))                   # b, c, H, 1
        a_w = torch.sigmoid(self.conv_w(x_w))                   # b, c, 1, W
        return x * a_h * a_w


def build_attn_block(name, ch):
    """按名字构建注意力模块；none → Identity（纯检测基线）。"""
    name = (name or "none").lower()
    if name in ("none", ""):
        return nn.Identity()
    if name == "se":
        return SEBlock(ch)
    if name == "cbam":
        return CBAM(ch)
    if name == "nonlocal":
        return NonLocalBlock(ch)
    if name == "coord":
        return CoordAttn(ch)
    raise ValueError(f"unknown attn_block: {name} (choose from none/se/cbam/nonlocal/coord)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 模型
# ══════════════════════════════════════════════════════════════════════════════
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


class AttnBBoxNet(nn.Module):
    """from-scratch 编码器 + 在 bottleneck 检测特征上插入一个通用注意力模块（attn_block）。"""
    def __init__(self, in_ch=1, base_ch=32, attn_block="none"):
        super().__init__()
        self.attn_name = attn_block
        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
        self.attn = build_attn_block(attn_block, mid)   # none=Identity
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
        _, x = self.down3(x)
        feat = self.bottleneck(x)
        feat = self.attn(feat)                          # 通用注意力（无监督；none 时为恒等）
        # 返回 4 元组（末位 a_logit 恒为 None）以复用 band 监督版的 predict/evaluate 接口
        return self.heatmap_head(feat), torch.sigmoid(self.wh_head(feat)), self.offset_head(feat), None


# ══════════════════════════════════════════════════════════════════════════════
# 5. Loss（纯检测，无注意力监督）
# ══════════════════════════════════════════════════════════════════════════════
def objectness_loss(obj_logit, peak, lam_noobj=LAM_NOOBJ):
    """YOLO 风格 objectness：硬目标(中心单元=1，其余=0) 的 BCE，正/负样本分开归一并加权平衡。"""
    bce = F.binary_cross_entropy_with_logits(obj_logit, peak, reduction="none")
    pos, neg = peak, 1.0 - peak
    n_pos = pos.sum().clamp(min=1.0)
    n_neg = neg.sum().clamp(min=1.0)
    return (bce * pos).sum() / n_pos + lam_noobj * (bce * neg).sum() / n_neg


def masked_l1(pred, gt, peak):
    mask = peak.expand_as(pred); n = mask.sum()
    if n == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred * mask, gt * mask, reduction="sum") / (n / pred.shape[1] + 1e-6)


def compute_loss(model, img, hm, wh, off, peak, band):
    obj_logit, wh_p, off_p, _ = model(img)             # 第一头当 objectness 用（YOLO 风格）
    return (objectness_loss(obj_logit, peak)
            + masked_l1(wh_p, wh, peak) + masked_l1(off_p, off, peak))


# ══════════════════════════════════════════════════════════════════════════════
# 6. 训练
# ══════════════════════════════════════════════════════════════════════════════
def train_model(attn_block, full, tr, va, n_ep, work, tag, seed, augment=False, run_val=False):
    set_seed(seed)
    tr_set = Subset(full, tr)
    if augment:
        tr_set = AugWrapper(tr_set)
    _pin = (device.type == "cuda")           # 有 GPU 才用锁页内存 + 异步拷贝
    tl = DataLoader(tr_set, batch_size=batch_size, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=_pin,
                    persistent_workers=(NUM_WORKERS > 0), worker_init_fn=_worker_init,
                    collate_fn=collate)
    vl = (DataLoader(Subset(full, va), batch_size=batch_size, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=_pin,
                     persistent_workers=(NUM_WORKERS > 0), worker_init_fn=_worker_init,
                     collate_fn=collate) if run_val else None)
    model = AttnBBoxNet(in_ch=1, base_ch=BASE_CH, attn_block=attn_block).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, n_ep + 1):
        model.train(); tr_tot = 0.0
        for img, hm, wh, off, pk, band, _ in tl:
            img, hm, wh, off, pk, band = [t.to(device, non_blocking=_pin) for t in (img, hm, wh, off, pk, band)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = compute_loss(model, img, hm, wh, off, pk, band)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_tot += loss.item()
        tr_loss = tr_tot / max(len(tl), 1)
        va_str = ""
        if run_val:
            model.eval(); tot = 0.0
            with torch.no_grad():
                for img, hm, wh, off, pk, band, _ in vl:
                    img, hm, wh, off, pk, band = [t.to(device, non_blocking=_pin) for t in (img, hm, wh, off, pk, band)]
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        tot += compute_loss(model, img, hm, wh, off, pk, band).item()
            va_str = f" val={tot / max(len(vl), 1):.4f}"
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} train={tr_loss:.4f}{va_str}", flush=True)
    torch.save(model.state_dict(), os.path.join(work, f"{tag}_final.pth"))
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7. 推理 / 评估
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, img_tensor):
    input_h, input_w = input_size
    gh, gw = input_h // HM_STRIDE, input_w // HM_STRIDE
    obj_logit, wh_p, off_p, _ = model(img_tensor.to(device))
    obj = torch.sigmoid(obj_logit[0, 0]).float().cpu().numpy()   # YOLO：objectness 网格
    ys, xs = np.where(obj >= HM_THRESH)                          # 阈值化得到候选单元
    boxes, scores = [], []
    if len(ys) > 0:
        whp, ofp = wh_p[0].float().cpu().numpy(), off_p[0].float().cpu().numpy()
        cand_b, cand_s = [], []
        for yi, xi in zip(ys, xs):
            bw = float(whp[0, yi, xi]) * input_w; bh = float(whp[1, yi, xi]) * input_h
            cx = (xi + float(np.clip(ofp[0, yi, xi], -0.5, 0.5))) / gw * input_w
            cy = (yi + float(np.clip(ofp[1, yi, xi], -0.5, 0.5))) / gh * input_h
            cand_b.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            cand_s.append(float(obj[yi, xi]))
        keep = nms_iou(cand_b, cand_s, NMS_IOU_THR)[:max_det]    # 框级 IoU-NMS
        boxes = [cand_b[i] for i in keep]; scores = [cand_s[i] for i in keep]
    return boxes, scores


@torch.no_grad()
def evaluate(model, dataset, test_idx):
    TP = FP = FN = 0
    for i in test_idx:
        img, _, _, _, _, _, meta = dataset[i]
        boxes, scores = predict(model, img.unsqueeze(0))
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
    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
    return {"bbox_P": P, "bbox_R": R, "bbox_F1": 2 * P * R / max(P + R, 1e-9)}


# ══════════════════════════════════════════════════════════════════════════════
# 8. 主程序：none / se / cbam / nonlocal / coord 多方对比
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_blocks", nargs="+", default=ATTN_BLOCKS,
                        choices=["none", "se", "cbam", "nonlocal", "coord"],
                        help="要对比的注意力模块（可多选）；none=纯检测基线")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="随机种子")
    parser.add_argument("--epochs", type=int, default=num_epochs)
    parser.add_argument("--augment", action="store_true", default=AUGMENT,
                        help="开启训练集数据增强（仅水平翻转），默认关闭")
    parser.add_argument("--val", action="store_true", default=RUN_VAL,
                        help="每个 epoch 跑一遍 val 做监控打印（不参与选模型），默认关闭")
    parser.add_argument("--train_fracs", nargs="+", type=float, default=TRAIN_FRACS,
                        help="训练集占比扫描列表，如 --train_fracs 0.3 0.5 0.7；其余为 test（val=0）")
    args = parser.parse_args()

    work = os.path.join(os.getcwd(), f"attn_cnn_yolo_attnblocks_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    # 把本次运行用的脚本快照复制到结果目录，便于复现（结果与代码一一对应）
    import shutil
    shutil.copy2(os.path.abspath(__file__), os.path.join(work, os.path.basename(__file__)))
    print("Using device:", device)
    full = AttnDataset(input_size=input_size, hm_stride=HM_STRIDE, sigma=HM_SIGMA)
    n = len(full)
    print(f"[yolo-attnblocks] n_total={n}  attn_blocks={args.attn_blocks}  seeds={args.seeds}  "
          f"train_fracs={args.train_fracs}  augment={args.augment}  run_val={args.val}", flush=True)

    rows = []
    for frac in args.train_fracs:
        print(f"\n■■■■■■■■■■ TRAIN_FRAC = {frac} ■■■■■■■■■■", flush=True)
        for seed in args.seeds:
            tr, va, te = make_split(n, seed, train_frac=frac, val_frac=0.0)
            te_pct = int(round((1.0 - frac) * 100))
            print(f"\n=== frac{frac} seed {seed}  train={len(tr)} val={len(va)} test={len(te)} "
                  f"({int(round(frac*100))}/0/{te_pct}) ===", flush=True)

            for block in args.attn_blocks:
                t0 = time.perf_counter()
                tag = f"frac{frac}_seed{seed}_{block}"
                model = train_model(block, full, tr, va, args.epochs, work, tag, seed,
                                     augment=args.augment, run_val=args.val)
                m = evaluate(model, full, te)
                print(f"  frac{frac} seed{seed} {block:>8}: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
                      f"F1={m['bbox_F1']:.4f}  [{time.perf_counter() - t0:.1f}s]", flush=True)
                rows.append({"train_frac": frac, "seed": seed, "config": block,
                             "bbox_P": m["bbox_P"], "bbox_R": m["bbox_R"], "bbox_F1": m["bbox_F1"]})
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    keys = ["bbox_P", "bbox_R", "bbox_F1"]
    print("\n" + "=" * 70)
    print(f"{'frac':>6}{'config':>10}" + "".join(f"{k:>16}" for k in keys))
    for frac in args.train_fracs:
        for cfg in args.attn_blocks:
            sub = [r for r in rows if r["train_frac"] == frac and r["config"] == cfg]
            if not sub:
                continue
            line = f"{frac:>6}{cfg:>10}"
            for k in keys:
                vals = [r[k] for r in sub if not (isinstance(r[k], float) and np.isnan(r[k]))]
                line += f"{np.mean(vals):>8.4f}±{np.std(vals):<7.4f}" if vals else f"{'nan':>16}"
            print(line)
    print("=" * 70)

    with open(os.path.join(work, "merged_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}", flush=True)


if __name__ == "__main__":
    main()
