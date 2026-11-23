"""论文第一个基础实验（本文三层 backbone，有 / 无注意力监督）在新标注、新划分、统一评估口径下的重跑。

    python train_basic.py --mode none --seed 0
    python train_basic.py --mode abs  --seed 0

模型、损失、优化器、轮数与 experiments/additional_centernet/centernet_band_attention.py 完全一致
（AttnBBoxNet base_ch=32，CenterNet 式热图头，attn 分支 BCE+Dice，concat 融合，Adam 5e-4，bf16，120 轮，最后一轮权重）。
按 paper/revision/revision_map.md 改掉的只有数据和评估：

    数据      prepare_data.py 导出的目录（默认 ../shared/datasets/ds2_f0_s100）
              - 划分：data/splits/dataset2_mendeley.json fold 0（固定测试集），不再按 seed 随机切
              - 检测框：官方框（gt_boxes_official_aug6）
              - 掩码：HBAT 画图代码直接导出的双曲线带 PNG（masks_band/），训练不再自己画抛物线
    评估      shared/evaluate.py 的统一口径：导出全部检测（峰值 >= 0.001），
              阈值在验证集上选，报告测试集 F1 / AP50；另给旧脚本口径（阈值 0.375、每图最多 5 个）作参照

输出 --runs/ds2/<exp>/seed<k>/ ：
    train_log.csv        每轮训练损失
    final.pth            最后一轮权重
    preds_val.json / preds_test.json   原图像素坐标的检测结果 [[x1,y1,x2,y2,score],...]
    run_info.json        配置、参数量、用时、注意力带 IoU、旧口径指标
    metrics.json         evaluate.evaluate_run 的统一指标
    vis/                 测试图：注意力图叠加 + 官方框(红) + 检测框(黄, 验证集选出的阈值)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
sys.path.insert(0, str(SHARED))
from evaluate import evaluate_run  # noqa: E402
from attn_blocks import ATTN_BLOCKS, build_attn_block  # noqa: E402  实验 4

# ───────────────────────── 与原脚本相同的超参数 ─────────────────────────
INPUT_SIZE = (224, 224)
HM_STRIDE = 8
ATTN_STRIDE = 4
HM_SIGMA = 6
BATCH = 8
EPOCHS = 120
LR = 5e-4
NMS_KERNEL = 3
BASE_CH = 32


def configure_det_stride(stride):
    """检测头步长 8（原样）或 4（第 8 轮：U 形解码回 56×56 再出头）。热图 sigma、NMS 核按物理尺度等比换算。"""
    global HM_STRIDE, HM_SIGMA, NMS_KERNEL
    HM_STRIDE = stride
    HM_SIGMA = 6 * 8 // stride
    NMS_KERNEL = 3 if stride == 8 else 5


def configure_attn_stride(stride):
    """attention map 的步长：4（原样，56×56）或 1（第 16 轮：U 形解码的 attention 头，在 224×224 上输出和监督）。"""
    global ATTN_STRIDE
    ATTN_STRIDE = stride


PAPER_HM_THRESH = 0.375   # 原脚本解码阈值（参照口径）
PAPER_MAX_DET = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _worker_init(worker_id):
    cv2.setNumThreads(0)
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s); random.seed(s)


def render_gaussian(heatmap, cx, cy, sigma):
    hm_h, hm_w = heatmap.shape
    ys = np.arange(hm_h, dtype=np.float32)[:, None]
    xs = np.arange(hm_w, dtype=np.float32)[None, :]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    np.maximum(heatmap, g, out=heatmap)


# ───────────────────────── 数据：读 prepare_data.py 的目录 ─────────────────────────
class PreparedDataset(Dataset):
    """images/<phase>/<stem>.jpg + gt_boxes.json 的官方框 + <mask_dir>/<phase>/<stem>.png 的带掩码。"""

    def __init__(self, data_dir, phase, mask_dir="masks_band", subset_frac=1.0, subset_seed=0, weight_dir=None):
        self.data_dir, self.phase = Path(data_dir), phase
        # 第 20 轮：attention 损失的像素权重图（make_weight_maps.py）；None = 不加权（与之前逐位相同）
        self.weight_dir = self.data_dir / weight_dir / phase if weight_dir else None
        self.gt = json.load(open(self.data_dir / "gt_boxes.json", encoding="utf-8"))[phase]
        self.stems = sorted(self.gt)
        if subset_frac < 1.0:
            # 第 9 轮：按固定种子取训练子集，两臂、各 seed 看到同一批图（只由 fold 决定）
            rng = np.random.default_rng(subset_seed)
            n_keep = max(1, int(round(len(self.stems) * subset_frac)))
            keep = sorted(rng.permutation(len(self.stems))[:n_keep].tolist())
            self.stems = [self.stems[i] for i in keep]
        img_dir = self.data_dir / "images" / phase
        self.files = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
        self.mask_dir = self.data_dir / mask_dir / phase
        self.input_h, self.input_w = INPUT_SIZE
        self.gh, self.gw = self.input_h // HM_STRIDE, self.input_w // HM_STRIDE
        self.attn_h, self.attn_w = self.input_h // ATTN_STRIDE, self.input_w // ATTN_STRIDE
        missing = [s for s in self.stems if not (self.mask_dir / f"{s}.png").exists()]
        if missing:
            raise FileNotFoundError(f"[{phase}] {len(missing)} 张图缺掩码: {missing[:5]}")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        return self.encode(idx, flip=False)

    def encode(self, idx, flip=False):
        """flip = True：几何水平翻转（图、框 x → W − x、band mask 一起翻），之后再编码目标。
        2026-11-21 修 bug：原来的 augment_sample 在编码之后把热图 / offset 当数组翻转，目标落在 27 − fx 格，
        真实位置是 28 − fx 格，每个翻转样本的目标都向左偏 8 px（第 4–16 轮所有 --augment run 都受影响）。"""
        stem = self.stems[idx]
        img = Image.open(self.files[stem]).convert("L")
        ow, oh = img.size
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        if flip:
            img_np = img_np[:, ::-1].copy()
        sx, sy = self.input_w / ow, self.input_h / oh

        hm = np.zeros((self.gh, self.gw), np.float32)
        wh = np.zeros((2, self.gh, self.gw), np.float32)
        off = np.zeros((2, self.gh, self.gw), np.float32)
        peak = np.zeros((self.gh, self.gw), np.float32)
        for x1, y1, x2, y2 in self.gt[stem]["boxes"]:
            if flip:
                x1, x2 = ow - x2, ow - x1
            bw, bh = (x2 - x1) * sx, (y2 - y1) * sy
            cx, cy = x1 * sx + bw / 2.0, y1 * sy + bh / 2.0
            fx, fy = cx / HM_STRIDE, cy / HM_STRIDE
            xi = int(np.clip(round(fx), 0, self.gw - 1)); yi = int(np.clip(round(fy), 0, self.gh - 1))
            render_gaussian(hm, fx, fy, HM_SIGMA)
            wh[0, yi, xi] = bw / self.input_w; wh[1, yi, xi] = bh / self.input_h
            off[0, yi, xi] = fx - xi; off[1, yi, xi] = fy - yi
            peak[yi, xi] = 1.0

        m = cv2.imread(str(self.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if m.shape != (self.input_h, self.input_w):
            m = cv2.resize(m, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST)
        band_full = (m > 127).astype(np.float32)
        if flip:
            band_full = band_full[:, ::-1].copy()
        if (self.attn_h, self.attn_w) == (self.input_h, self.input_w):
            band = band_full          # 第 16 轮：attention 在输入分辨率上监督，不缩小、不重新二值化
        else:
            # 与原脚本相同：面积插值到注意力分辨率后按 0.5 二值化
            band = cv2.resize(band_full, (self.attn_w, self.attn_h), interpolation=cv2.INTER_AREA)
            band = (band > 0.5).astype(np.float32)
        band_t = torch.from_numpy(band).unsqueeze(0).float()
        if self.weight_dir is not None:
            # 权重图和 band 一起翻转、缩到注意力分辨率；再归一化成 band 像素平均 1、背景 1（只重新分配 band 内部的权重）
            w = np.load(self.weight_dir / f"{stem}.npy").astype(np.float32)
            if w.shape != (self.input_h, self.input_w):
                w = cv2.resize(w, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            if flip:
                w = w[:, ::-1].copy()
            if w.shape != band.shape:
                w = cv2.resize(w, (self.attn_w, self.attn_h), interpolation=cv2.INTER_AREA)
            on = band > 0
            w = np.where(on, w, 1.0).astype(np.float32)
            if on.any():
                w[on] *= on.sum() / w[on].sum()
            band_t = torch.from_numpy(np.stack([band, w])).float()     # 第 2 通道 = 权重，compute_loss 拆开

        meta = {"stem": stem, "orig_size": (oh, ow), "scale": (sx, sy), "boxes": self.gt[stem]["boxes"]}
        return (torch.from_numpy(img_np).unsqueeze(0).float(), torch.from_numpy(hm).unsqueeze(0).float(),
                torch.from_numpy(wh).float(), torch.from_numpy(off).float(),
                torch.from_numpy(peak).unsqueeze(0).float(), band_t, meta)


def collate(batch):
    imgs, hms, whs, offs, pks, bands, metas = zip(*batch)
    return (torch.stack(imgs), torch.stack(hms), torch.stack(whs), torch.stack(offs),
            torch.stack(pks), torch.stack(bands), list(metas))


FLIP_LEGACY = False   # --flip-legacy：复现第 4–16 轮的旧翻转（有 8 px 的 bug），只为对照旧结果


def augment_sample(img, hm, wh, off, peak, band):
    """原脚本自带的训练增强（有 bug，只在 --flip-legacy 时用）：编码后把标签当数组翻转、x 偏移取负，目标向左偏 1 格 = 8 px。"""
    if random.random() < 0.5:
        img = torch.flip(img, dims=[2]); hm = torch.flip(hm, dims=[2]); wh = torch.flip(wh, dims=[2])
        peak = torch.flip(peak, dims=[2]); band = torch.flip(band, dims=[2]); off = torch.flip(off, dims=[2])
        off[0] = -off[0]
    return img, hm, wh, off, peak, band


class AugWrapper(Dataset):
    """只包训练集：每次取样以 0.5 概率水平翻转（默认几何翻转；--flip-legacy 时用旧的数组翻转）。"""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if FLIP_LEGACY:
            img, hm, wh, off, peak, band, meta = self.ds[idx]
            img, hm, wh, off, peak, band = augment_sample(img, hm, wh, off, peak, band)
            return img, hm, wh, off, peak, band, meta
        # 和旧版本一样每个样本取一次 random.random()（编码本身不用 random），所以每个样本翻不翻和旧 run 相同
        return self.ds.encode(idx, flip=random.random() < 0.5)


# ───────────────────────── 模型（原样） ─────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

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


EXTENT_KERNELS = (17, 33)   # 56×56 上的盒式滤波核（≈ 68 / 132 px）：把"带在这附近延伸多宽"算成特征


def extent_channels(A, pool):
    """A: (B,1,56,56) 带概率图 -> 4 通道范围特征：原图池化、水平 17、竖直 17、水平 33 盒式均值。
    pool=True 时再 2×2 平均池化到 28×28（步长 8 的检测头）。"""
    k1, k2 = EXTENT_KERNELS
    chans = [A,
             F.avg_pool2d(A, (1, k1), stride=1, padding=(0, k1 // 2)),
             F.avg_pool2d(A, (k1, 1), stride=1, padding=(k1 // 2, 0)),
             F.avg_pool2d(A, (1, k2), stride=1, padding=(0, k2 // 2))]
    out = torch.cat(chans, dim=1)
    return F.avg_pool2d(out, 2) if pool else out


class AttnDecoder(nn.Module):
    """第 16 轮：U 形解码的 attention 头（--attn-head unet）。从 bottleneck（28×28，receptive field ≈ 68 px）逐级上采样，
    接 f3 / f2 / f1 的 skip，在输入分辨率（224×224）上输出 band。原来的头只看 f3（receptive field ≈ 40 px，一条双曲线宽 ≈ 65 px）。"""

    def __init__(self, base_ch):
        super().__init__()
        self.d3 = ConvBlock(base_ch * 8 + base_ch * 4, base_ch * 2)
        self.d2 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch)
        self.d1 = ConvBlock(base_ch + base_ch, base_ch // 2)
        self.out = nn.Conv2d(base_ch // 2, 1, 1)

    def forward(self, feat, f3, f2, f1):
        y = self.d3(torch.cat([F.interpolate(feat, scale_factor=2, mode="nearest"), f3], dim=1))
        y = self.d2(torch.cat([F.interpolate(y, scale_factor=2, mode="nearest"), f2], dim=1))
        y = self.d1(torch.cat([F.interpolate(y, scale_factor=2, mode="nearest"), f1], dim=1))
        return self.out(y)


class AttnBBoxNet(nn.Module):
    """三层 backbone + CenterNet 头（原样）。扩展：
    oracle      第 7 轮：不设注意力头，把真实带（56×56）直接当 gate 融合 —— 融合通路的上限。
    fuse=extent 第 8 轮 a：gate 不再是 1 通道池化，而是 4 通道"范围"特征（extent_channels）。
    det_stride=4 第 8 轮 b：bottleneck 上采样回 56×56 与 f3 拼接后出头，注意力在 56×56 融合、不再池化。
    generic     实验 4：在 bottleneck 检测特征上（band 融合的同一位置）插一个无监督的通用注意力模块（SE / CBAM / Non-local / CA）。"""

    def __init__(self, in_ch=1, base_ch=32, use_attn=True, fuse="concat", oracle=False, det_stride=8, attn_arch="conv", generic="none"):
        super().__init__()
        self.use_attn, self.fuse, self.oracle, self.det_stride, self.attn_arch = use_attn, fuse, oracle, det_stride, attn_arch
        self.generic_name = generic
        assert attn_arch == "conv" or (det_stride == 8 and fuse in ("concat", "gate")), "unet attention 头只支持 det_stride=8、concat / gate 融合"
        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
        if det_stride == 4:
            self.up = ConvBlock(mid + base_ch * 4, mid)
        if generic != "none":
            self.generic = build_attn_block(generic, mid)
        if use_attn and attn_arch == "unet":
            self.attn_head = AttnDecoder(base_ch)
        elif use_attn:
            self.attn_head = nn.Sequential(
                nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True), nn.Conv2d(base_ch * 2, 1, 1))
        n_gate = 0
        if use_attn or oracle:
            n_gate = {"concat": 1, "extent": 4, "gate": 0}[fuse]
        if n_gate:
            self.fuse_conv = nn.Sequential(nn.Conv2d(mid + n_gate, mid, 1, bias=False),
                                           nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.heatmap_head = nn.Sequential(nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, 1, 1))
        self.wh_head = nn.Sequential(nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(mid // 2), nn.ReLU(inplace=True), nn.Conv2d(mid // 2, 2, 1))
        self.offset_head = nn.Sequential(nn.Conv2d(mid, mid // 4, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(mid // 4), nn.ReLU(inplace=True), nn.Conv2d(mid // 4, 2, 1))

    def forward(self, x, band=None):
        f1, x = self.down1(x); f2, x = self.down2(x)
        f3, x = self.down3(x)
        feat = self.bottleneck(x)
        if self.det_stride == 4:
            feat = self.up(torch.cat([F.interpolate(feat, scale_factor=2, mode="nearest"), f3], dim=1))
        if self.generic_name != "none":
            feat = self.generic(feat)
        a_logit, A = None, None
        if self.use_attn:
            a_logit = self.attn_head(feat, f3, f2, f1) if self.attn_arch == "unet" else self.attn_head(f3)
            A = torch.sigmoid(a_logit)
        elif self.oracle:
            assert band is not None, "oracle 模式需要真实带"
            A = band.to(feat.dtype)
        if A is not None:
            pool = self.det_stride == 8
            if self.fuse == "extent":
                feat = self.fuse_conv(torch.cat([feat, extent_channels(A, pool)], dim=1))
            else:
                gate = F.avg_pool2d(A, HM_STRIDE // ATTN_STRIDE) if pool else A      # 原样：56 → 28（池化 2）；unet 头：224 → 28（池化 8）
                if self.fuse == "concat":
                    feat = self.fuse_conv(torch.cat([feat, gate], dim=1))
                else:
                    feat = feat * (1.0 + gate)
        return self.heatmap_head(feat), torch.sigmoid(self.wh_head(feat)), self.offset_head(feat), a_logit


def build_model(use_attn, fuse, seed, oracle=False, det_stride=8, attn_arch="conv", generic="none"):
    """配对初始化：backbone + 三个检测头用 seed 初始化，两臂逐参数相同；注意力分支（attn_head / fuse_conv）
    用 seed+10000 单独初始化。这样同一 seed 下两臂的差异只来自注意力分支本身。
    实验 4 的通用注意力模块（generic.*）同样用 seed+10000 初始化，backbone 和检测头与 baseline 逐参数相同。"""
    torch.manual_seed(seed)
    shared = AttnBBoxNet(in_ch=1, base_ch=BASE_CH, use_attn=False, fuse=fuse, oracle=False, det_stride=det_stride)
    if generic != "none":
        assert not use_attn and not oracle, "通用注意力模块只和 --mode none 组合"
        torch.manual_seed(seed + 10000)
        model = AttnBBoxNet(in_ch=1, base_ch=BASE_CH, use_attn=False, fuse=fuse, oracle=False, det_stride=det_stride, generic=generic)
        res = model.load_state_dict(shared.state_dict(), strict=False)
        assert not res.unexpected_keys and res.missing_keys and all(k.startswith("generic.") for k in res.missing_keys), res
        return model.to(device)
    if not use_attn and not oracle:
        return shared.to(device)
    torch.manual_seed(seed + 10000)
    model = AttnBBoxNet(in_ch=1, base_ch=BASE_CH, use_attn=use_attn, fuse=fuse, oracle=oracle, det_stride=det_stride, attn_arch=attn_arch)
    res = model.load_state_dict(shared.state_dict(), strict=False)
    assert not res.unexpected_keys and all(k.startswith(("attn_head.", "fuse_conv.")) for k in res.missing_keys), res
    return model.to(device)


def init_fingerprint(model):
    """共享层初始化指纹（两臂应完全相同）：第一层卷积与热图头最后一层权重的和。"""
    with torch.no_grad():
        return [float(model.down1.conv.block[0].weight.double().sum()), float(model.heatmap_head[-1].weight.double().sum())]


# ───────────────────────── 损失（原样） ─────────────────────────
def focal_loss_heatmap(pred_logit, target_hm, peak_mask, alpha=2.0, beta=4.0):
    pred = torch.sigmoid(pred_logit)
    neg_w = (1.0 - target_hm) ** beta
    pos_loss = peak_mask * (1.0 - pred) ** alpha * torch.log(pred.clamp(min=1e-6))
    neg_loss = neg_w * (1.0 - peak_mask) * pred ** alpha * torch.log((1.0 - pred).clamp(min=1e-6))
    return -(pos_loss + neg_loss).sum() / peak_mask.sum().clamp(min=1.0)


def masked_l1(pred, gt, peak):
    mask = peak.expand_as(pred); n = mask.sum()
    if n == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred * mask, gt * mask, reduction="sum") / (n / pred.shape[1] + 1e-6)


def attn_loss_abs(a_logit, band, w=None):
    if w is None:
        bce = F.binary_cross_entropy_with_logits(a_logit, band)
        p = torch.sigmoid(a_logit)
        dice = 1.0 - 2.0 * (p * band).sum() / (p.sum() + band.sum() + 1e-6)
        return bce + dice
    # 第 20 轮：像素加权的 BCE + Dice（w 在背景为 1，band 像素平均为 1）
    bce = F.binary_cross_entropy_with_logits(a_logit, band, weight=w)
    p = torch.sigmoid(a_logit)
    dice = 1.0 - 2.0 * (w * p * band).sum() / ((w * p).sum() + (w * band).sum() + 1e-6)
    return bce + dice


def compute_loss(model, img, hm, wh, off, peak, band, mode, lam):
    w = None
    if band.shape[1] == 2:          # --attn-weight-dir：第 2 通道是像素权重
        band, w = band[:, :1], band[:, 1:]
    hm_logit, wh_p, off_p, a_logit = model(img, band)
    det = focal_loss_heatmap(hm_logit, hm, peak) + masked_l1(wh_p, wh, peak) + masked_l1(off_p, off, peak)
    att = attn_loss_abs(a_logit, band, w) if (model.use_attn and mode == "abs") else torch.zeros((), device=img.device)
    return det + lam * att, det.detach(), att.detach()


# ───────────────────────── 推理 ─────────────────────────
def _heatmap_nms(hm, kernel):
    t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    tmax = F.max_pool2d(t, kernel, stride=1, padding=kernel // 2)
    return hm * (t == tmax).squeeze(0).squeeze(0).numpy()


@torch.no_grad()
def predict(model, img_tensor, meta, hm_thresh, max_det, band=None):
    """返回原图像素坐标的 [[x1,y1,x2,y2,score],...] 与注意力图（输入分辨率 /4）。oracle 模式需传 band (1,1,56,56)。"""
    input_h, input_w = INPUT_SIZE
    gh, gw = input_h // HM_STRIDE, input_w // HM_STRIDE
    hm_logit, wh_p, off_p, a_logit = model(img_tensor.to(device), band.to(device) if band is not None else None)
    hm = torch.sigmoid(hm_logit[0, 0]).float().cpu().numpy()
    hm_nms = _heatmap_nms(hm, NMS_KERNEL)
    ys, xs = np.where(hm_nms >= hm_thresh)
    sx, sy = meta["scale"]
    dets = []
    if len(ys) > 0:
        sc = hm_nms[ys, xs]; order = np.argsort(-sc, kind="stable")[:max_det]
        ys, xs, sc = ys[order], xs[order], sc[order]
        whp, ofp = wh_p[0].float().cpu().numpy(), off_p[0].float().cpu().numpy()
        for yi, xi, s in zip(ys, xs, sc):
            bw = float(whp[0, yi, xi]) * input_w; bh = float(whp[1, yi, xi]) * input_h
            cx = (xi + float(np.clip(ofp[0, yi, xi], -0.5, 0.5))) / gw * input_w
            cy = (yi + float(np.clip(ofp[1, yi, xi], -0.5, 0.5))) / gh * input_h
            dets.append([(cx - bw / 2) / sx, (cy - bh / 2) / sy, (cx + bw / 2) / sx, (cy + bh / 2) / sy, float(s)])
    A = torch.sigmoid(a_logit[0, 0]).float().cpu().numpy() if a_logit is not None else None
    return dets, A


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def paper_protocol(preds, gt, thresh, max_det):
    """旧脚本口径：固定阈值、每图最多 max_det 个、IoU0.5 贪心匹配。"""
    TP = FP = FN = 0
    for stem, boxes in gt.items():
        ds = sorted([d for d in preds.get(stem, []) if d[4] >= thresh], key=lambda d: -d[4])[:max_det]
        matched = [False] * len(boxes)
        for d in ds:
            bi, bj = 0.0, -1
            for j, g in enumerate(boxes):
                if not matched[j]:
                    iou = box_iou(d[:4], g)
                    if iou > bi:
                        bi, bj = iou, j
            if bi >= 0.5 and bj >= 0:
                matched[bj] = True; TP += 1
            else:
                FP += 1
        FN += len(boxes) - sum(matched)
    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
    return {"thresh": thresh, "max_det": max_det, "TP": TP, "FP": FP, "FN": FN, "P": P, "R": R,
            "F1": 2 * P * R / max(P + R, 1e-9)}


@torch.no_grad()
def export_and_attn(model, ds, out_path, hm_thresh, max_det):
    """导出检测 + 统计注意力与带的 IoU / 带内外均值差。"""
    model.eval()
    preds, ai, au, ins, outs = {}, 0.0, 0.0, [], []
    for i in range(len(ds)):
        img, _, _, _, _, band, meta = ds[i]
        dets, A = predict(model, img.unsqueeze(0), meta, hm_thresh, max_det, band.unsqueeze(0) if model.oracle else None)
        preds[meta["stem"]] = dets
        if A is not None:
            b = band[0].numpy() > 0.5
            ab = A > 0.5
            ai += float((ab & b).sum()); au += float((ab | b).sum())
            if b.sum() >= 1:
                ins.append(float(A[b].mean())); outs.append(float(A[~b].mean()))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    stats = {"attn_band_iou": (ai / au) if au > 0 else None,
             "attn_gap": (float(np.mean(ins) - np.mean(outs)) if ins else None)}
    return preds, stats


@torch.no_grad()
def visualize(model, ds, out_dir, conf, n=12):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i in range(min(n, len(ds))):
        img, _, _, _, _, band, meta = ds[i]
        dets, A = predict(model, img.unsqueeze(0), meta, 0.001, 50, band.unsqueeze(0) if model.oracle else None)
        base = cv2.imread(str(ds.files[meta["stem"]]))
        h, w = base.shape[:2]
        panels = [base.copy()]
        if A is not None:
            heat = cv2.applyColorMap((cv2.resize(A, (w, h)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            panels.append(cv2.addWeighted(base, 0.5, heat, 0.5, 0))
            bm = cv2.resize(band[0].numpy(), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            gtp = base.copy(); gtp[bm] = (0.4 * gtp[bm] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            panels.append(gtp)
        for p in panels:
            for x1, y1, x2, y2 in meta["boxes"]:
                cv2.rectangle(p, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        for x1, y1, x2, y2, s in dets:
            if s >= conf:
                cv2.rectangle(panels[0], (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                cv2.putText(panels[0], f"{s:.2f}", (int(x1), max(10, int(y1) - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"{meta['stem']}.png"), np.concatenate(panels, axis=1))


# ───────────────────────── 主流程 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(SHARED / "datasets" / "ds2_f0_s100"))
    ap.add_argument("--mode", required=True, choices=["none", "abs", "oracle"],
                    help="none=无注意力基线, abs=带监督注意力(本文方法), oracle=真实带直接当 gate 融合(第 7 轮上限)")
    ap.add_argument("--exp", default=None, help="实验名，默认 basic_<mode>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lam-att", type=float, default=0.7, help="注意力损失权重（论文默认 0.7）")
    ap.add_argument("--fuse", default="concat", choices=["concat", "gate", "extent"],
                    help="concat=1 通道池化后拼接(原样), gate=feat*(1+gate), extent=4 通道范围特征拼接(第 8 轮)")
    ap.add_argument("--det-stride", type=int, default=8, choices=[4, 8], help="检测头步长；4 = 解码回 56×56 出头(第 8 轮)")
    ap.add_argument("--train-frac", type=float, default=1.0, help="训练集比例(第 9 轮数据量曲线)，子集只由 fold 决定")
    ap.add_argument("--mask-dir", default="masks_band")
    ap.add_argument("--attn-weight-dir", default=None,
                    help="第 20 轮：attention 损失的像素权重图目录（make_weight_maps.py 输出，如 wmaps_band_b2），只用于训练集")
    ap.add_argument("--attn-head", default="conv", choices=["conv", "unet"],
                    help="conv=原样(f3 上一个 3×3 卷积，56×56)；unet=U 形解码到 224×224，全分辨率监督(第 16 轮)")
    ap.add_argument("--attn-block", default="none", choices=list(ATTN_BLOCKS),
                    help="实验 4：bottleneck 检测特征上插一个无监督的通用注意力模块（只和 --mode none 组合）")
    ap.add_argument("--augment", action="store_true", help="训练集水平翻转增强（几何翻转，2026-11-21 修过 8 px 的 bug）")
    ap.add_argument("--flip-legacy", action="store_true", help="用第 4–16 轮的旧翻转（目标偏 8 px 的 bug），只为复现旧结果")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--fold-in-path", action="store_true", help="run 目录名带折号：f<fold>_seed<k>（5 折实验用）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pred-thresh", type=float, default=0.001)
    ap.add_argument("--pred-max-det", type=int, default=50)
    args = ap.parse_args()
    global FLIP_LEGACY
    FLIP_LEGACY = args.flip_legacy   # 在建 DataLoader 之前设好（Linux fork 的 worker 会继承）

    exp = args.exp or f"basic_{args.mode}"
    data_dir = Path(args.data)
    manifest = json.load(open(data_dir / "manifest.json", encoding="utf-8"))
    dataset_name = manifest.get("dataset", data_dir.name)
    fold = manifest.get("fold", 0)
    run_dir = Path(args.runs) / dataset_name / exp / (f"f{fold}_seed{args.seed}" if args.fold_in_path else f"seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run_dir={run_dir}", flush=True)

    set_seed(args.seed)
    configure_det_stride(args.det_stride)
    configure_attn_stride(1 if (args.attn_head == "unet" and args.mode == "abs") else 4)
    tr = PreparedDataset(data_dir, "train", args.mask_dir, subset_frac=args.train_frac, subset_seed=1000 + fold,
                         weight_dir=args.attn_weight_dir)
    va = PreparedDataset(data_dir, "val", args.mask_dir)
    te = PreparedDataset(data_dir, "test", args.mask_dir)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  mode={args.mode} attn_block={args.attn_block} lam={args.lam_att} fuse={args.fuse} "
          f"epochs={args.epochs} seed={args.seed} augment={args.augment} data={data_dir.name}", flush=True)
    pin = device.type == "cuda"
    # 数据顺序（和 worker 内的翻转随机数）用独立生成器，两臂每轮看到的样本顺序完全相同
    order_gen = torch.Generator()
    order_gen.manual_seed(seed_of_run := args.seed)
    tl = DataLoader(AugWrapper(tr) if args.augment else tr, batch_size=BATCH, shuffle=True, num_workers=args.workers, pin_memory=pin,
                    persistent_workers=args.workers > 0, worker_init_fn=_worker_init, collate_fn=collate, generator=order_gen)

    model = build_model(args.mode == "abs", args.fuse, seed_of_run, oracle=(args.mode == "oracle"), det_stride=args.det_stride,
                        attn_arch=args.attn_head, generic=args.attn_block)
    fingerprint = init_fingerprint(model)
    print(f"paired init fingerprint (shared layers): {fingerprint}", flush=True)
    torch.manual_seed(seed_of_run)   # 训练中其余随机数（如有）从同一状态出发
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.perf_counter()
    log_rows = []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = det_tot = att_tot = 0.0
        for img, hm, wh, off, pk, band, _ in tl:
            img, hm, wh, off, pk, band = [t.to(device, non_blocking=pin) for t in (img, hm, wh, off, pk, band)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, det, att = compute_loss(model, img, hm, wh, off, pk, band, args.mode, args.lam_att)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); det_tot += det.item(); att_tot += att.item()
        nb = max(len(tl), 1)
        log_rows.append({"epoch": ep, "loss": tot / nb, "det_loss": det_tot / nb, "att_loss": att_tot / nb})
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  epoch {ep}/{args.epochs}  loss {tot / nb:.4f}  det {det_tot / nb:.4f}  att {att_tot / nb:.4f}", flush=True)
    train_minutes = (time.perf_counter() - t0) / 60
    torch.save(model.state_dict(), run_dir / "final.pth")
    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0])); w.writeheader(); w.writerows(log_rows)

    # 导出检测（统一口径）+ 注意力统计
    preds_val, stats_val = export_and_attn(model, va, run_dir / "preds_val.json", args.pred_thresh, args.pred_max_det)
    preds_test, stats_test = export_and_attn(model, te, run_dir / "preds_test.json", args.pred_thresh, args.pred_max_det)
    paper = paper_protocol(preds_test, {s: te.gt[s]["boxes"] for s in te.stems}, PAPER_HM_THRESH, PAPER_MAX_DET)

    info = dict(dataset=dataset_name, exp=exp, seed=args.seed,
                model=f"AttnBBoxNet(base_ch={BASE_CH}, 3 down blocks, CenterNet head, det_stride={args.det_stride})",
                pretrained=False, mode=args.mode, lam_att=args.lam_att if args.mode == "abs" else None, fuse=args.fuse,
                oracle_band=(args.mode == "oracle"), attn_block=args.attn_block, det_stride=args.det_stride, attn_head=args.attn_head, attn_stride=ATTN_STRIDE,
                train_frac=args.train_frac, n_train=len(tr),
                mask_dir=args.mask_dir, attn_weight_dir=args.attn_weight_dir, augment=args.augment,
                flip=("legacy (array flip, 8 px bug)" if args.flip_legacy else "geometric (fixed 2026-11-21)") if args.augment else None,
                box_source=manifest.get("box_source"), fold=fold,
                paired_init=True, init_fingerprint=fingerprint,
                imgsz=INPUT_SIZE[0], epochs=args.epochs, batch=BATCH, lr=LR, optimizer="Adam",
                amp="bfloat16", model_selection="last epoch (no val-based selection, same as the paper script)",
                params_M=round(n_params / 1e6, 4), train_minutes=round(train_minutes, 2),
                torch=torch.__version__, data_dir=str(data_dir), args=vars(args),
                attn_val=stats_val, attn_test=stats_test,
                paper_protocol_test=paper)
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)

    m = evaluate_run(run_dir, data_dir)
    t = m["test"]["at_selected"]
    # 同一批预测对另一套框再评一遍（官方框 / HBAT 派生框），写 metrics_<name>.json
    for alt in ("official", "hbat"):
        if (data_dir / f"gt_boxes_{alt}.json").exists():
            ma = evaluate_run(run_dir, data_dir, gt_name=f"gt_boxes_{alt}.json", out_name=f"metrics_{alt}.json")
            ta = ma["test"]["at_selected"]
            print(f"    vs {alt} boxes: val conf {ma['val']['selected_conf']:.2f} -> test F1 {ta['F1']:.4f} "
                  f"(P {ta['P']:.3f} R {ta['R']:.3f}) AP50 {ma['test']['AP50']:.4f}", flush=True)
    visualize(model, te, run_dir / "vis", m["val"]["selected_conf"])
    print(f"[{dataset_name}/{exp}/seed{args.seed}] val-selected conf {m['val']['selected_conf']:.2f} -> "
          f"test F1 {t['F1']:.4f} (P {t['P']:.3f} R {t['R']:.3f})  AP50 {m['test']['AP50']:.4f}  "
          f"| paper-protocol(0.375, max5) F1 {paper['F1']:.4f} (P {paper['P']:.3f} R {paper['R']:.3f}) "
          f"| attn band IoU {stats_test['attn_band_iou']}  | {n_params / 1e6:.3f} M params, {train_minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
