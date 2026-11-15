"""
自己完结・单文件版：YOLO 网格(anchor-free) 矩形框检测头 + 显式注意力监督（双曲线带 mask 监督注意力图）。
把 0616-1.py / attn_cnn.py / attn_cnn_relative.py 中本实验需要的部分全部合并到这一个文件，
不再 import 其它脚本。

【检测风格：YOLO 网格 anchor-free（与 attn_cnn_merged_final.py 的 CenterNet 风格不同）】
  - 监督：objectness 用「硬」目标——只有 GT 框中心所在的网格单元 = 1，其余 = 0；
    用 obj/noobj 平衡的 BCE（不再用中心高斯热图 + focal）。
  - 解码：sigmoid(objectness) ≥ 阈值的单元 → 解码出框 → 框级 IoU-NMS（不再用热图 maxpool-NMS）。
  - 框回归(wh/offset)、注意力(none/abs/soft, gate/concat)、数据集、训练、评估、划分 全部不变。
  - 注：dataset 里仍会算高斯 hm，但 YOLO 风格用的是 peak(硬中心)，所以 HM_SIGMA 在本版本不起作用。

【与 attn_cnn_merged.py 的区别】
  本版本**直接用最终 epoch 的模型**做推理评估，不再按 val loss 挑选 best、也不回滚。
  每个 epoch 仍会跑一遍 val 但只用于打印监控；训练结束保存 {tag}_final.pth。

三种注意力模式作为参数（--modes 选择，默认三种全跑）：
  none : 无注意力        —— 纯 CenterNet，不加注意力 head / 不加注意力 loss
  abs  : abs 注意力      —— A(x) ≈ band mask 绝对逐像素匹配（BCE + Dice）
  soft : 软（相对）注意力 —— 只要求 带内均值 − 带外均值 ≥ margin 的 margin-hinge 软约束
                            （原 attn_cnn_relative.py 的 "rel"，相对约束、对边界误差几乎免疫）

完整对照表 @ objectness 阈值 = 0.9
config	lam	P	R	F1	band_iou	gap
none	—	0.520	0.745	0.599 ±0.133	nan	nan
abs	0.3	0.690	0.769	0.726 ±0.038	0.270	0.344
abs	0.5	0.607	0.787	0.673 ±0.126	0.281	0.340
abs	1	0.487	0.703	0.551 ±0.242	0.278	0.345
abs	3	0.534	0.778	0.615 ±0.195	0.333	0.418
abs	5	0.465	0.809	0.580 ±0.126	0.357	0.464
"""
import os
import re
import csv
import json
import math
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
IMG_DIR   = os.path.join(_REPO_ROOT, "dataset3", "images-selected-shuangquxian")
HYP_JSON  = os.path.join(IMG_DIR, "annotations.json")
RECT_JSON = os.path.join(IMG_DIR, "annotations_rect.json")

# —— 实验配置（直接改这几个；命令行 --modes / --seeds / --augment 可覆盖）——
MODES        = ["none", "abs", "soft"]    #soft的结果可能需要跑一下更大的 lam 因为和abs的原始数量级不同abs 是 BCE+Dice(02),soft 是 margin-hinge(00.5,很小)
MODES        = ["none","abs"]  #  abs(绝对)
SEEDS        =  [10]
         # 随机种子，如 [1, 2, 3, 4]
TRAIN_FRACS  = [0.6]                     # 训练集占比扫描列表，如 [0.3,0.5,0.7]：逐个对比不同训练数据量；命令行 --train_fracs 可覆盖
AUGMENT      = False                   # 训练集数据增强开关（仅水平翻转），先关着，需要确认数据集中是否已经有了翻转。
FUSE         = "concat"                     # 注意力融合方式：gate(乘法门控,默认) / concat(拼接+1x1卷积)
RUN_VAL      = False                      # 是否每个 epoch 跑一遍 val（仅打印监控）。还没做自动选参，默认关，省时

input_size   = (640, 640)
HM_STRIDE    = 8                    # heatmap 下采样步长
ATTN_STRIDE  = 4                    # 注意力图下采样步长
HM_SIGMA     = 6                  # heatmap 高斯半径
batch_size   = 8
NUM_WORKERS  = 8   # DataLoader 数据加载进程数：自动按 CPU 核数（封顶 8）；设 0 关闭多进程加载
num_epochs   = 80
LR           = 5e-4                 # Adam 学习率
nms_kernel   = 3                   # （YOLO 风格不再用热图 maxpool-NMS，此项无效，保留兼容）
max_det      = 5
HM_THRESH    = 0.9                  # YOLO：objectness 解码阈值（sigmoid(obj) ≥ 此值才算候选；扫描显示 0.9~0.97 最佳）
BBOX_IOU_THR = 0.5                 # 评测匹配 IoU 阈值
NMS_IOU_THR  = 0.5                 # YOLO 解码：框级 IoU-NMS 阈值（IoU ≥ 此值的低分框被抑制）
LAM_NOOBJ    = 0.5                 # YOLO：objectness 损失中 noobj(背景) 项权重，平衡正负样本
BASE_CH      = 32                   # 网络通道基数

# —— 注意力监督相关（调参核心）——
LAM_ATT      = [0.5,1,3,5]
LAM_ATT      = [0.7]            # 注意力 loss 权重扫描列表（abs/soft 特有）：逐个训练对比；命令行 --lam_att 可覆盖
LAM_ATT_CUR  = 1.0                  # 占位用。。。运行时当前权重（主循环按 LAM_ATT 逐个设置；compute_loss 实际用它）
MARGIN       = 0.5                  # soft 模式：要求 带内均值 − 带外均值 ≥ margin

# —— 推理可视化（把网络的注意力/检测响应画成 heatmap 叠加到原图）——
VIS                 = True          # 训练后是否输出可视化图（abs/soft 看注意力 A；none 看 objectness 响应）
VIS_NUM             = 0             # 每个模型可视化多少张 test 图（设 0 或负数 = 全部 test 图）
VIS_ONLY_FIRST_SEED = False          # 只对第一个 seed 出图（省时省空间），其余 seed 跳过
VIS_ALPHA           = 0.45          # heatmap 叠加透明度（0~1）

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
    """DataLoader worker 初始化：关掉 OpenCV 内部多线程（避免多进程下争抢 CPU），并按 worker 重新播种保持可复现。"""
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


def hyperbola_ab(y_vertex, slope):
    """与标注工具一致的物理半轴：a=顶点深度(=y_vertex)，b=a/slope（slope=a/b 编码波速）。"""
    a = max(1.0, abs(float(y_vertex)))
    s = max(1e-3, float(slope))
    return a, a / s


def rasterize_hyperbola_band_mask(h, w, x_v, y_v, span, slope, thickness):
    """与标注工具一致的「真双曲线」带：中心线 y_c=y_v+a*(sqrt(1+((x-x_v)/b)^2)-1)，
    thickness 为竖直厚度（沿 y 方向偏移 ±thickness/2）；span 为水平绘制范围。"""
    span      = max(float(span),      2.0)
    thickness = max(float(thickness), 1.0)
    a, b      = hyperbola_ab(y_v, slope)
    half_w    = span / 2.0
    n_pts     = max(40, int(round(span)))
    upper_pts, lower_pts = [], []
    for i in range(n_pts + 1):
        t  = i / max(n_pts, 1)
        x  = (x_v - half_w) + span * t
        yc = y_v + a * (math.sqrt(1.0 + ((x - x_v) / b) ** 2) - 1.0)
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
    """真双曲线带 → 轴对齐外接框（与标注工具 export_rectangles 一致）。"""
    hw = o["span"] / 2.0
    a, b = hyperbola_ab(o["y_vertex"], o["slope"])
    y_edge = o["y_vertex"] + a * (math.sqrt(1.0 + (hw / b) ** 2) - 1.0)
    return [o["x_vertex"] - hw, o["y_vertex"] - o["thickness"] / 2.0,
            o["x_vertex"] + hw, y_edge + o["thickness"] / 2.0]


def _heatmap_nms(hm, kernel):
    t    = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    tmax = F.max_pool2d(t, kernel, stride=1, padding=kernel // 2)
    keep = (t == tmax).squeeze(0).squeeze(0).numpy()
    return hm * keep


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
            # 缩放到 640 输入空间：x/span 用 sx，y/thickness 用 sy；
            # slope=a/b（a 竖向、b 横向）在各向异性缩放下按 sy/sx 变换，保证曲线形状一致。
            mo = {"x_vertex": o["x_vertex"] * sx, "y_vertex": o["y_vertex"] * sy,
                  "span": o["span"] * sx, "slope": o["slope"] * sy / sx,
                  "thickness": o["thickness"] * sy}
            meta_objs.append(mo)
            band_full = np.maximum(band_full, rasterize_hyperbola_band_mask(
                self.input_h, self.input_w, mo["x_vertex"], mo["y_vertex"],
                mo["span"], mo["slope"], mo["thickness"]))
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
# 3. 模型
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
    def __init__(self, in_ch=1, base_ch=32, use_attn=True, fuse="gate"):
        super().__init__()
        self.use_attn = use_attn
        self.fuse = fuse                       # "gate"(乘法门控) 或 "concat"(通道拼接+1x1融合)
        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
        if use_attn:
            self.attn_head = nn.Sequential(
                nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 2, 1, 1))
            if fuse == "concat":               # 把注意力图作为额外1通道拼上, 再用1x1卷积压回 mid
                self.fuse_conv = nn.Sequential(
                    nn.Conv2d(mid + 1, mid, 1, bias=False),
                    nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
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
            gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)   # 对齐到 bottleneck 分辨率
            if self.fuse == "concat":
                feat = self.fuse_conv(torch.cat([feat, gate], dim=1))  # 拼接1通道再1x1融合
            else:
                feat = feat * (1.0 + gate)   # gate: 只增强双曲线带、不抑制背景
        return self.heatmap_head(feat), torch.sigmoid(self.wh_head(feat)), self.offset_head(feat), a_logit


# ══════════════════════════════════════════════════════════════════════════════
# 4. Loss
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


def attn_loss_abs(a_logit, band):
    """abs：A(x) ≈ band mask 绝对逐像素匹配（BCE + Dice）。"""
    bce = F.binary_cross_entropy_with_logits(a_logit, band)
    p = torch.sigmoid(a_logit)
    dice = 1.0 - 2.0 * (p * band).sum() / (p.sum() + band.sum() + 1e-6)
    return bce + dice


def attn_loss_soft(a_logit, band, margin=MARGIN):
    """soft（相对）：margin-hinge，只约束带内/带外平均注意力的相对差。"""
    A = torch.sigmoid(a_logit)
    eps = 1e-6
    s_in  = (A * band).sum(dim=(1, 2, 3))
    s_out = (A * (1.0 - band)).sum(dim=(1, 2, 3))
    n_in  = band.sum(dim=(1, 2, 3)).clamp(min=eps)
    n_out = (1.0 - band).sum(dim=(1, 2, 3)).clamp(min=eps)
    mean_in  = s_in / n_in
    mean_out = s_out / n_out
    return torch.relu(margin - (mean_in - mean_out)).mean()


def compute_loss(model, img, hm, wh, off, peak, band, attn_mode):
    obj_logit, wh_p, off_p, a_logit = model(img)   # 第一头当 objectness 用（YOLO 风格）
    loss = (objectness_loss(obj_logit, peak)       # 硬中心 BCE，不再用高斯 hm/focal
            + masked_l1(wh_p, wh, peak) + masked_l1(off_p, off, peak))
    if model.use_attn:
        if attn_mode == "abs":
            loss = loss + LAM_ATT_CUR * attn_loss_abs(a_logit, band)
        elif attn_mode == "soft":
            loss = loss + LAM_ATT_CUR * attn_loss_soft(a_logit, band)
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# 5. 训练
# ══════════════════════════════════════════════════════════════════════════════
def train_model(attn_mode, full, tr, va, n_ep, work, tag, seed, augment=False, fuse="gate", run_val=False):
    set_seed(seed)
    use_attn = (attn_mode != "none")
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
    model = AttnBBoxNet(in_ch=1, base_ch=BASE_CH, use_attn=use_attn, fuse=fuse).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, n_ep + 1):
        model.train(); tr_tot = 0.0
        for img, hm, wh, off, pk, band, _ in tl:
            img, hm, wh, off, pk, band = [t.to(device, non_blocking=_pin) for t in (img, hm, wh, off, pk, band)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss = compute_loss(model, img, hm, wh, off, pk, band, attn_mode)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_tot += loss.item()
        tr_loss = tr_tot / max(len(tl), 1)
        # 验证仅用于打印监控（不参与选模型）；run_val=False 时整段跳过、省时
        va_str = ""
        if run_val:
            model.eval(); tot = 0.0
            with torch.no_grad():
                for img, hm, wh, off, pk, band, _ in vl:
                    img, hm, wh, off, pk, band = [t.to(device, non_blocking=_pin) for t in (img, hm, wh, off, pk, band)]
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        tot += compute_loss(model, img, hm, wh, off, pk, band, attn_mode).item()
            va_str = f" val={tot / max(len(vl), 1):.4f}"
        if ep % 20 == 0 or ep == n_ep:
            print(f"  [{tag}] epoch {ep}/{n_ep} train={tr_loss:.4f}{va_str}", flush=True)
    # 直接用最终 epoch 的模型推理（不做 val 模型选择 / 不回滚到 best）
    torch.save(model.state_dict(), os.path.join(work, f"{tag}_final.pth"))
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 6. 推理 / 评估
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, img_tensor):
    input_h, input_w = input_size
    gh, gw = input_h // HM_STRIDE, input_w // HM_STRIDE
    obj_logit, wh_p, off_p, a_logit = model(img_tensor.to(device))
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


@torch.no_grad()
def attn_gap(model, full, test_idx):
    """test 上的 带内 − 带外 平均注意力差（soft 模式的直接目标）。"""
    ins, outs = [], []
    for i in test_idx:
        img, _, _, _, _, band, _ = full[i]
        _, _, A = predict(model, img.unsqueeze(0))
        if A is None:
            return float("nan")
        b = band[0].numpy()
        m = b > 0.5
        if m.sum() < 1:
            continue
        ins.append(float(A[m].mean())); outs.append(float(A[~m].mean()))
    if not ins:
        return float("nan")
    return float(np.mean(ins) - np.mean(outs))


def _colorize(heat01, size_wh):
    """0..1 单通道热图 → JET 伪彩色 BGR，并 resize 到 (W, H)。"""
    h = cv2.resize(heat01.astype(np.float32), size_wh, interpolation=cv2.INTER_LINEAR)
    h = np.clip(h, 0.0, 1.0)
    return cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _draw_boxes(im, bs, c, thick=2):
    for b in bs:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(im, (x1, y1), (x2, y2), c, thick)


def _scale_box(b, sx, sy):
    """640 输入空间的框 → 原分辨率坐标（x*sx, y*sy）。"""
    return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]


def _overlay(base_bgr, heat01, alpha):
    """按热图值做**逐像素**透明度叠加：低值(背景)几乎不着色、保持原灰度图，
    只有高响应处才显现伪彩色 —— 避免 JET 低值把整幅图蒙上一层蓝。"""
    H, W = base_bgr.shape[:2]
    color = _colorize(heat01, (W, H)).astype(np.float32)
    h = cv2.resize(heat01.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    a = (np.clip(h, 0.0, 1.0) * alpha)[:, :, None]          # 逐像素 alpha ∝ 热图值
    out = base_bgr.astype(np.float32) * (1.0 - a) + color * a
    return out.clip(0, 255).astype(np.uint8)


def grad_cam(model, img_tensor, peak=None):
    """对「融合后特征」(检测 head 的输入) 做 Grad-CAM。
    目标分数 = **所有 GT 中心格子的 objectness 之和**（对全部真双曲线中心一起反传），
    比单点 max() 更能反映「网络为了检测这些双曲线、依赖了哪片区域」；
    无 GT(peak 为空或全 0) 时退回 objectness 峰值。
    所有模型(none/abs/soft) 用同一个探针，是 none 与 abs/soft 之间唯一公平可比的「关注」。
    需要梯度，故不在 no_grad 下；并临时关确定性算子(部分 backward 无确定性实现，可视化无需复现)。"""
    store = {}

    def _pre_hook(module, inp):
        feat = inp[0]
        feat.retain_grad()
        store["feat"] = feat

    h = model.heatmap_head.register_forward_pre_hook(_pre_hook)
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            obj_logit, _, _, _ = model(img_tensor.to(device))        # (1,1,gh,gw)
            if peak is not None and float(peak.sum()) > 0:
                pk = peak.to(device).reshape(1, 1, *obj_logit.shape[-2:])
                score = (obj_logit * pk).sum()                       # GT 中心 objectness 求和
            else:
                score = obj_logit.max()                              # 无 GT 退回峰值
            score.backward()                                         # 对(所有)真双曲线中心反传
        feat = store["feat"]
        weights = feat.grad.mean(dim=(2, 3), keepdim=True)   # 梯度全局平均 → 通道权重
        cam = torch.relu((weights * feat).sum(dim=1))[0]     # 加权求和 + ReLU
        cam = (cam / (cam.max() + 1e-6)).float().detach().cpu().numpy()
    finally:
        h.remove()
        model.zero_grad(set_to_none=True)
        torch.use_deterministic_algorithms(prev_det)
    return cam


def _panel_row(model, img, peak, base_bgr, gt, alpha, sx, sy):
    """单个模型、单张图 → 一行 4 panel（均为原分辨率、不叠加任何文字标注）：
       [原图+GT(蓝)/预测(绿)] | [objectness] | [注意力A / none:灰底] | [Grad-CAM(公平对比列)]。
       sx, sy: 640 输入空间 → 原分辨率的缩放，用于把预测框映射回原图。"""
    with torch.no_grad():
        obj_logit, _, _, a_logit = model(img.unsqueeze(0).to(device))
    obj = torch.sigmoid(obj_logit[0, 0]).float().cpu().numpy()
    A = (torch.sigmoid(a_logit[0, 0]).float().cpu().numpy() if a_logit is not None else None)
    boxes, _, _ = predict(model, img.unsqueeze(0))
    boxes = [_scale_box(b, sx, sy) for b in boxes]                    # 640 空间 → 原分辨率
    cam = grad_cam(model, img.unsqueeze(0), peak)                     # 目标分数 = GT 中心 objectness 之和

    p1 = base_bgr.copy(); _draw_boxes(p1, gt, (255, 0, 0)); _draw_boxes(p1, boxes, (0, 255, 0))

    p2 = _overlay(base_bgr, obj, alpha)
    _draw_boxes(p2, boxes, (0, 255, 0))

    if A is not None:
        p3 = _overlay(base_bgr, A, alpha)
    else:
        p3 = np.full_like(base_bgr, 60)

    p4 = _overlay(base_bgr, cam, alpha)
    return np.hstack([p1, p2, p3, p4])


def visualize_compare(trained, full, idxs, out_dir, alpha=VIS_ALPHA):
    """同一张 test 图在多个模型下的结果竖直堆成多行对比图：none 一行、每个 abs/soft+lam 一行。
    trained: [(label, model), ...]；每行 4 panel，最后一列 Grad-CAM 是所有模型公平可比的「网络关注」。
    底图用**原分辨率**灰度图（从 image_path 重新读入），热图/框 均映射回原分辨率；不叠加任何文字、不勾勒带、低响应处不着色。"""
    os.makedirs(out_dir, exist_ok=True)
    H, W = input_size
    for i in idxs:
        img, _, _, _, peak, _, meta = full[i]
        oh, ow = meta["orig_size"]
        base = np.array(Image.open(meta["image_path"]).convert("L"), dtype=np.uint8)  # 原分辨率
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        sx, sy = ow / W, oh / H                                       # 640 空间 → 原分辨率
        gt = [_scale_box(hyperbola_to_bbox(o), sx, sy) for o in meta["objects"]]
        rows_img = [_panel_row(model, img, peak, base_bgr, gt, alpha, sx, sy)
                    for _label, model in trained]
        cv2.imwrite(os.path.join(out_dir, f"cmp_{os.path.splitext(meta['image_name'])[0]}.png"),
                    np.vstack(rows_img))


# ══════════════════════════════════════════════════════════════════════════════
# 7. 主程序：none / abs / soft 三方对比
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global LAM_ATT_CUR
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=MODES,
                        choices=["none", "abs", "soft"], help="要跑的注意力模式")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="随机种子")
    parser.add_argument("--epochs", type=int, default=num_epochs)
    parser.add_argument("--augment", action="store_true", default=AUGMENT,
                        help="开启训练集数据增强（仅水平翻转），默认关闭")
    parser.add_argument("--fuse", default=FUSE, choices=["gate", "concat"],
                        help="注意力融合方式：gate(乘法门控) / concat(拼接+1x1卷积)")
    parser.add_argument("--val", action="store_true", default=RUN_VAL,
                        help="每个 epoch 跑一遍 val 做监控打印（不参与选模型），默认关闭")
    parser.add_argument("--lam_att", nargs="+", type=float, default=LAM_ATT,
                        help="注意力 loss 权重扫描列表，如 --lam_att 0.3 0.5 2；默认取文件里的 LAM_ATT")
    parser.add_argument("--train_fracs", nargs="+", type=float, default=TRAIN_FRACS,
                        help="训练集占比扫描列表，如 --train_fracs 0.3 0.5 0.7；其余为 test（val=0）。默认取文件里的 TRAIN_FRACS")
    parser.add_argument("--vis", action="store_true", default=VIS,
                        help="训练后输出注意力/objectness 可视化叠加图（存到结果目录的 vis/ 下）")
    parser.add_argument("--no-vis", dest="vis", action="store_false",
                        help="关闭可视化")
    parser.add_argument("--vis_num", type=int, default=VIS_NUM,
                        help="每个模型可视化多少张 test 图（0 或负数 = 全部 test 图）")
    args = parser.parse_args()

    work = os.path.join(os.getcwd(), f"attn_cnn_yolo_final_camgt_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    # 把本次运行用的脚本快照复制到结果目录，便于复现（结果与代码一一对应）
    import shutil
    shutil.copy2(os.path.abspath(__file__), os.path.join(work, os.path.basename(__file__)))
    print("Using device:", device)
    full = AttnDataset(input_size=input_size, hm_stride=HM_STRIDE, sigma=HM_SIGMA)
    n = len(full)
    print(f"[yolo] n_total={n}  modes={args.modes}  seeds={args.seeds}  train_fracs={args.train_fracs}  "
          f"margin={MARGIN}  lam_att_sweep={args.lam_att}  augment={args.augment}  fuse={args.fuse}  run_val={args.val}", flush=True)

    # none 不用注意力 loss，与 LAM_ATT 无关 → 每个 (frac, seed) 只训一次（lam 记为 "-"）；
    # abs/soft 才需要对每个 lam 各训一次。
    attn_modes = [m for m in args.modes if m != "none"]
    run_none   = ("none" in args.modes)

    rows = []
    for frac in args.train_fracs:
        print(f"\n■■■■■■■■■■ TRAIN_FRAC = {frac} ■■■■■■■■■■", flush=True)
        for seed in args.seeds:
            tr, va, te = make_split(n, seed, train_frac=frac, val_frac=0.0)
            te_pct = int(round((1.0 - frac) * 100))
            print(f"\n=== frac{frac} seed {seed}  train={len(tr)} val={len(va)} test={len(te)} "
                  f"({int(round(frac*100))}/0/{te_pct}) ===", flush=True)

            trained = []                        # 收集本 (frac, seed) 下训练好的模型，最后统一画多行对比图

            if run_none:                        # none：lam 无关，只跑一次
                t0 = time.perf_counter()
                tag = f"frac{frac}_seed{seed}_none"
                model = train_model("none", full, tr, va, args.epochs, work, tag, seed,
                                     augment=args.augment, fuse=args.fuse, run_val=args.val)
                m = evaluate(model, full, te)
                gap = attn_gap(model, full, te)
                print(f"  frac{frac} seed{seed} none: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
                      f"F1={m['bbox_F1']:.4f} attn_iou={m['attn_band_iou']:.4f} gap={gap:.4f}  "
                      f"[{time.perf_counter() - t0:.1f}s]", flush=True)
                rows.append({"train_frac": frac, "seed": seed, "config": "none", "lam_att": "-",
                             "bbox_P": m["bbox_P"], "bbox_R": m["bbox_R"], "bbox_F1": m["bbox_F1"],
                             "attn_band_iou": m["attn_band_iou"], "attn_gap": gap})
                trained.append(("none", model))

            for lam in args.lam_att:            # abs/soft：每个 lam 各训一次
                LAM_ATT_CUR = lam               # compute_loss 读这个全局，改它即可换注意力权重
                for attn_mode in attn_modes:
                    t0 = time.perf_counter()
                    tag = f"frac{frac}_seed{seed}_{attn_mode}_lam{lam}"
                    model = train_model(attn_mode, full, tr, va, args.epochs, work, tag, seed,
                                         augment=args.augment, fuse=args.fuse, run_val=args.val)
                    m = evaluate(model, full, te)
                    gap = attn_gap(model, full, te)
                    print(f"  frac{frac} lam{lam} seed{seed} {attn_mode:>4}: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
                          f"F1={m['bbox_F1']:.4f} attn_iou={m['attn_band_iou']:.4f} gap={gap:.4f}  "
                          f"[{time.perf_counter() - t0:.1f}s]", flush=True)
                    rows.append({"train_frac": frac, "seed": seed, "config": attn_mode, "lam_att": lam,
                                 "bbox_P": m["bbox_P"], "bbox_R": m["bbox_R"], "bbox_F1": m["bbox_F1"],
                                 "attn_band_iou": m["attn_band_iou"], "attn_gap": gap})
                    trained.append((f"{attn_mode} lam{lam}", model))

            # 本 (frac, seed) 下所有模型训练完 → 多行对比可视化（none 一行、每个 abs/soft+lam 一行）
            if args.vis and (not VIS_ONLY_FIRST_SEED or seed == args.seeds[0]):
                vis_idxs = te if args.vis_num <= 0 else te[:args.vis_num]
                visualize_compare(trained, full, vis_idxs, os.path.join(work, "vis", f"frac{frac}_seed{seed}"))
                print(f"  [vis] saved {len(vis_idxs)} 张对比图（{len(trained)} 行/张）-> vis/frac{frac}_seed{seed}", flush=True)
            for _lbl, _mdl in trained:          # 释放本 seed 各模型占用的显存
                del _mdl
            del trained
            if device.type == "cuda":
                torch.cuda.empty_cache()

    keys = ["bbox_P", "bbox_R", "bbox_F1", "attn_band_iou", "attn_gap"]
    print("\n" + "=" * 113)
    print(f"{'frac':>6}{'config':>8}{'lam_att':>9}" + "".join(f"{k:>16}" for k in keys))
    for frac in args.train_fracs:
        for cfg in args.modes:
            # none 只有一行（lam="-"）；abs/soft 按 lam 逐行
            lam_list = ["-"] if cfg == "none" else args.lam_att
            for lam in lam_list:
                sub = [r for r in rows if r["train_frac"] == frac and r["config"] == cfg and r["lam_att"] == lam]
                if not sub:
                    continue
                line = f"{frac:>6}{cfg:>8}{str(lam):>9}"
                for k in keys:
                    vals = [r[k] for r in sub if not (isinstance(r[k], float) and np.isnan(r[k]))]
                    line += f"{np.mean(vals):>8.4f}±{np.std(vals):<7.4f}" if vals else f"{'nan':>16}"
                print(line)
    print("=" * 113)

    with open(os.path.join(work, "merged_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}", flush=True)


if __name__ == "__main__":
    main()
