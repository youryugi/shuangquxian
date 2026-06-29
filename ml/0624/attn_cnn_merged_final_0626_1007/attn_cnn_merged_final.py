"""
自己完结・单文件版：CenterNet 检测头 + 显式注意力监督（双曲线带 mask 监督注意力图）。
把 0616-1.py / attn_cnn.py / attn_cnn_relative.py 中本实验需要的部分全部合并到这一个文件，
不再 import 其它脚本。

【与 attn_cnn_merged.py 的区别】
  本版本**直接用最终 epoch 的模型**做推理评估，不再按 val loss 挑选 best、也不回滚。
  每个 epoch 仍会跑一遍 val 但只用于打印监控；训练结束保存 {tag}_final.pth。

三种注意力模式作为参数（--modes 选择，默认三种全跑）：
  none : 无注意力        —— 纯 CenterNet，不加注意力 head / 不加注意力 loss
  abs  : abs 注意力      —— A(x) ≈ band mask 绝对逐像素匹配（BCE + Dice）
  soft : 软（相对）注意力 —— 只要求 带内均值 − 带外均值 ≥ margin 的 margin-hinge 软约束
                            （原 attn_cnn_relative.py 的 "rel"，相对约束、对边界误差几乎免疫）

用法：
  python attn_cnn_merged_final.py                  # 跑 none / abs / soft 三种
  python attn_cnn_merged_final.py --modes soft     # 只跑软注意力
  python attn_cnn_merged_final.py --modes abs soft # 跑 abs + soft
  python attn_cnn_merged_final.py --seeds 0 1 2    # 指定随机种子
  python attn_cnn_merged_final.py --augment        # 开启训练集增强（默认关闭）

数据增强（开关 AUGMENT / --augment）：仅对训练集做 水平翻转(标签同步、x偏移取负)。
  这是一开始的注意力版本的结果
  config          bbox_P          bbox_R         bbox_F1   attn_band_iou        attn_gap
    none  0.7348±0.1635   0.5455±0.0235   0.6156±0.0733              nan             nan
     abs  0.7632±0.1198   0.5225±0.0899   0.6127±0.0770   0.2044±0.0154   0.2538±0.0288 
    soft  0.7332±0.0818   0.5014±0.0822   0.5895±0.0583   0.1484±0.0104   0.4075±0.0303 

 改成concat 此时lam att是1  ================================================================================================
  config          bbox_P          bbox_R         bbox_F1   attn_band_iou        attn_gap
     abs  0.7770±0.1249   0.5172±0.0333   0.6147±0.0413   0.2584±0.0136   0.3296±0.0174 
================================================================================================
========================================================================================================
  config  lam_att          bbox_P          bbox_R         bbox_F1   attn_band_iou        attn_gap
     abs      0.3  0.6545±0.1021   0.5799±0.0305   0.6107±0.0482   0.2214±0.0176   0.2818±0.0292 
     abs      0.5  0.6696±0.1181   0.5940±0.0410   0.6270±0.0748   0.2393±0.0177   0.3032±0.0283 
     abs        2  0.6960±0.0792   0.6602±0.0745   0.6697±0.0217   0.2754±0.0220   0.3421±0.0382 
========================================================================================================
  config  lam_att          bbox_P          bbox_R         bbox_F1   attn_band_iou        attn_gap
     abs        1  0.6864±0.1312   0.6030±0.0412   0.6343±0.0682   0.2584±0.0136   0.3296±0.0174 

  水平翻转增强
  然后
  config  lam_att          bbox_P          bbox_R         bbox_F1   attn_band_iou        attn_gap
    none        1  0.5740±0.1168   0.4822±0.0654   0.5157±0.0589              nan             nan
     abs        1  0.6367±0.0371   0.5753±0.0528   0.6027±0.0337   0.2500±0.0134   0.3193±0.0294      
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

# —— 实验配置（直接改这几个；命令行 --modes / --seeds / --augment 可覆盖）——
MODES        = ["none", "abs", "soft"]  
MODES        = ["none", "abs"]  #  abs(绝对)
SEEDS        = [0, 1, 2, 3, 4]           # 随机种子，如 [1, 2, 3, 4]
TRAIN_FRACS  = [0.70]                     # 训练集占比扫描列表，如 [0.3,0.5,0.7]：逐个对比不同训练数据量；命令行 --train_fracs 可覆盖
AUGMENT      = False                     # 训练集数据增强开关（仅水平翻转），先关着，需要确认数据集中是否已经有了翻转。
FUSE         = "gate"                     # 注意力融合方式：gate(乘法门控,默认) / concat(拼接+1x1卷积)
RUN_VAL      = False                      # 是否每个 epoch 跑一遍 val（仅打印监控）。还没做自动选参，默认关，省时

input_size   = (640, 640)
HM_STRIDE    = 8                    # heatmap 下采样步长
ATTN_STRIDE  = 4                    # 注意力图下采样步长
HM_SIGMA     = 6                  # heatmap 高斯半径
batch_size   = 8
NUM_WORKERS  = min(os.cpu_count() or 1, 8)   # DataLoader 数据加载进程数：自动按 CPU 核数（封顶 8）；设 0 关闭多进程加载
num_epochs   = 80
LR           = 5e-4                 # Adam 学习率
nms_kernel   = 3
max_det      = 5
HM_THRESH    = 0.3                # 解码峰值阈值
BBOX_IOU_THR = 0.5                 # 评测匹配 IoU 阈值
BASE_CH      = 32                   # 网络通道基数

# —— 注意力监督相关（调参核心）——
LAM_ATT      = [1]        # 注意力 loss 权重扫描列表（abs/soft 特有）：逐个训练对比；命令行 --lam_att 可覆盖
LAM_ATT_CUR  = 1.0                  # 占位用。。。运行时当前权重（主循环按 LAM_ATT 逐个设置；compute_loss 实际用它）
MARGIN       = 0.5                  # soft 模式：要求 带内均值 − 带外均值 ≥ margin

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


def _heatmap_nms(hm, kernel):
    t    = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    tmax = F.max_pool2d(t, kernel, stride=1, padding=kernel // 2)
    keep = (t == tmax).squeeze(0).squeeze(0).numpy()
    return hm * keep


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
def focal_loss_heatmap(pred_logit, target_hm, peak_mask, alpha=2.0, beta=4.0):
    pred     = torch.sigmoid(pred_logit)
    pos_mask = peak_mask
    neg_w    = (1.0 - target_hm) ** beta
    pos_loss = pos_mask * (1.0 - pred) ** alpha * torch.log(pred.clamp(min=1e-6))
    neg_loss = neg_w * (1.0 - pos_mask) * pred ** alpha * torch.log((1.0 - pred).clamp(min=1e-6))
    n_pos    = pos_mask.sum().clamp(min=1.0)
    return -(pos_loss + neg_loss).sum() / n_pos


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
    hm_logit, wh_p, off_p, a_logit = model(img)
    loss = (focal_loss_heatmap(hm_logit, hm, peak)
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
    hm_logit, wh_p, off_p, a_logit = model(img_tensor.to(device))
    hm = torch.sigmoid(hm_logit[0, 0]).float().cpu().numpy()
    hm_nms = _heatmap_nms(hm, nms_kernel)
    ys, xs = np.where(hm_nms >= HM_THRESH)
    boxes, scores = [], []
    if len(ys) > 0:
        sc = hm_nms[ys, xs]; order = np.argsort(sc)[::-1][:max_det]
        ys, xs, sc = ys[order], xs[order], sc[order]
        whp, ofp = wh_p[0].float().cpu().numpy(), off_p[0].float().cpu().numpy()
        for yi, xi in zip(ys, xs):
            bw = float(whp[0, yi, xi]) * input_w; bh = float(whp[1, yi, xi]) * input_h
            cx = (xi + float(np.clip(ofp[0, yi, xi], -0.5, 0.5))) / gw * input_w
            cy = (yi + float(np.clip(ofp[1, yi, xi], -0.5, 0.5))) / gh * input_h
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]); scores.append(float(hm_nms[yi, xi]))
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
    args = parser.parse_args()

    work = os.path.join(os.getcwd(), f"attn_cnn_merged_final_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    # 把本次运行用的脚本快照复制到结果目录，便于复现（结果与代码一一对应）
    import shutil
    shutil.copy2(os.path.abspath(__file__), os.path.join(work, os.path.basename(__file__)))
    print("Using device:", device)
    full = AttnDataset(input_size=input_size, hm_stride=HM_STRIDE, sigma=HM_SIGMA)
    n = len(full)
    print(f"[merged] n_total={n}  modes={args.modes}  seeds={args.seeds}  train_fracs={args.train_fracs}  "
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
