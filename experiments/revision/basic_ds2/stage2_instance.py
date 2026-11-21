"""第 14 轮：按每条双曲线做 band attention（第二阶段，instance-level）。

动机（2026-11-15）：整张图上没有一个尺度同时适合所有双曲线（数据集 2 的目标宽 <53 到 ≥80 px，band 竖直厚度 7–41 px），
band 定框对宽目标 recall +0.145、对窄目标只有 +0.014（Round 12）。所以把 attention 从"整张图"换成"每条双曲线"：

  1. 第一阶段：已有的 baseline 检测器（runs_cv/ds2/hbatbox_aug_none，不重训）给出候选框；
  2. 每个候选框取一个以它为中心、宽高各放大 K 倍的窗口，重采样成固定的 S×S——每条双曲线进网络时大小一致；
  3. 在窗口上做 band attention：U 形小网络只预测"窗口中间这一条"双曲线的 band（实例级监督，不画相邻的双曲线），
     attention map 和窗口拼成 2 通道送进回归网络，输出修正后的框。分数沿用第一阶段的（本轮不重打分）。

arm（回归网络的结构、初始化、训练窗口完全相同；只有 attention 分支不同）：
  plain    没有 attention 分支（第二通道恒为 0）   —— "普通的二阶段框回归"对照，只用框标注
  unsup    有 attention 分支但不监督（λ = 0）       —— 同等参数量的对照
  boxmask  attention 用填实的实例框监督             —— "稠密头但没有 band 形状"的对照（同 Round 13），只用框标注
  band     attention 用这一条双曲线的 band 监督     —— 方法
  oracle   不设 attention 分支，把真实的实例 band 直接当 attention map（训练和推理都给）—— 上限

配对初始化：回归网络用 seed 初始化（各 arm 逐参数相同），attention 分支用 seed+10000；训练窗口的抖动、顺序、翻转由
(seed, fold) 决定的独立随机数发生器给出，各 arm 看到完全相同的窗口。

用法
    python stage2_instance.py train [--arms ...] [--folds ...] [--seeds ...] [--jobs 8]   # 每个 (fold, seed) 一个进程，依次训各 arm
    python stage2_instance.py build                                                       # 验证集全局选参，写新实验的 run 目录
    python stage2_instance.py diag                                                        # 窗口级诊断：框 IoU 前后、band IoU 按宽度分层

train 的原始输出在 runs_cv/s2_cache/<arm>/f<fold>_seed<k>/（不入库）；build 生成的新实验（格式与 train_basic.py 相同，
aggregate_cv.py / diag_strata.py 直接可用）：
    hbatbox_aug_none_s2{plain,unsup,boxmask,band,oracle}   框 = 回归网络的输出
    hbatbox_aug_none_s2{boxmask,band}_mb                   框 = attention map 二值化后最大连通块的外接框（实例级的"mask 定框"）
    hbatbox_aug_none_nms                                   对照：第一阶段的框不修正，只加和第二阶段相同的 NMS
--src 换第一阶段（例如 --src hbatbox_aug_abs = 带 attention 的第一阶段），输出相应的 <src>_s2* 实验。
参数（迭代次数 1/2、NMS 的 IoU、mask 阈值）每个变体在"10 个 seed 的 5 折合并验证集 F1 均值"上全局选一组，测试集不参与。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DETCMP = HERE.parent / "detector_comparison"
DATASETS = DETCMP / "datasets"
sys.path.insert(0, str(DETCMP))
sys.path.insert(0, str(REPO / "tools" / "annotation"))
from evaluate import box_iou, collect, evaluate_run, prf_at, select_conf  # noqa: E402
import hbat  # noqa: E402

HBAT_JSON = REPO / "data" / "additional_utilities" / "hbat" / "annotations.json"
RUNS = HERE / "runs_cv" / "ds2"
CACHE = HERE / "runs_cv" / "s2_cache"
OUT = HERE / "runs_cv" / "s2"
SRC_EXP = "hbatbox_aug_none"
FOLDS, SEEDS = list(range(5)), list(range(10))
ARMS = ("plain", "unsup", "boxmask", "band", "oracle")
ATT_ARMS = ("unsup", "boxmask", "band")

# ───────────────────────── 超参数（见 rounds.md Round 14 的"设计"一节） ─────────────────────────
S = 96                 # 窗口重采样后的边长
K = 2.5                # 窗口 = 候选框宽高各放大 K 倍（验证集上 K=2.5 时 82% 的 GT 整个落在窗口内）
MIN_SIDE = 12.0        # 候选框的宽高下限（px），防止窗口退化
EPOCHS = 100            # 试跑（fold 0–2 × seed 0–1，只看验证集窗口）：plain 在 20–150 轮之间验证 IoU 0.57–0.59 无趋势，100 轮最高；band 同样 100 轮最高
REPEAT = 8             # 每个 epoch 每个训练目标抖动出 REPEAT 个窗口
BATCH = 64
LR = 5e-4
LAM = 0.7              # attention loss 的权重，与第一阶段相同
AMP = True             # 卷积部分 bf16 混合精度 + channels_last（单进程实测 plain 8.1 → 3.5 ms/iter，band 25.1 → 9.6 ms/iter）
# 训练窗口的抖动：按第一阶段在合并验证集上的框误差拟合（score ≥ 0.3、IoU ≥ 0.1 的预测，n = 1271）
#   宽/高比 中位数 0.92/0.96、四分位 0.72–1.15 / 0.75–1.23；中心偏移 / GT 宽高 的四分位 ±0.12 / ±0.14，2.5% 分位 ±0.39 / ±0.6
JIT_SCALE_MU, JIT_SCALE_SIG, JIT_SCALE_CLIP = 0.93, 0.36, (0.4, 2.2)
JIT_DX_SIG, JIT_DX_CLIP = 0.17, 0.5
JIT_DY_SIG, JIT_DY_CLIP = 0.27, 0.75
T_GRID = (0.3, 0.5, 0.7)
NMS_GRID = (None, 0.5, 0.7)
ASSIGN_IOU = 0.1       # 候选框与 GT 的 IoU ≥ 此值才算"对着这个目标"（只用于 oracle arm 和诊断）


def data_dir(fold):
    return DATASETS / f"ds2_f{fold}_s100_hbatbox"


def cache_dir(arm, fold, seed):
    """默认来源（baseline）沿用 s2_cache/<arm>/；其他来源（--src）放在 s2_cache/<src>__<arm>/。"""
    sub = arm if SRC_EXP == "hbatbox_aug_none" else f"{SRC_EXP}__{arm}"
    return CACHE / sub / f"f{fold}_seed{seed}"


def set_source(src):
    """换第一阶段的来源实验（候选框从 runs_cv/ds2/<src>/ 读），并重建变体名。"""
    global SRC_EXP, VARIANTS
    SRC_EXP = src
    VARIANTS = make_variants(src)


def run_dir(exp, fold, seed):
    return RUNS / exp / f"f{fold}_seed{seed}"


# ───────────────────────── 数据：图像 + 每个 GT 框对应的实例 band ─────────────────────────
class Phase:
    """一个 fold 的一个 phase：图像张量、GT 框、每个 GT 框自己的 band（实例级）。"""

    def __init__(self, fold, phase, ann, device):
        dd = data_dir(fold)
        gt = json.load(open(dd / "gt_boxes.json", encoding="utf-8"))[phase]
        self.stems = sorted(gt)
        self.gt = gt
        files = {p.stem: p for p in (dd / "images" / phase).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
        imgs, inst_img, inst_box, inst_band, inst_has = [], [], [], [], []
        self.inst_of = {}          # (图像序号, GT 序号) -> 实例序号
        self.n_union_mismatch = 0  # 实例 band 的并集与 masks_band 不一致的像素数（应为 0）
        for i, stem in enumerate(self.stems):
            item = gt[stem]
            im = Image.open(files[stem]).convert("L")
            w, h = im.size
            imgs.append(np.asarray(im, np.float32) / 255.0)
            cand = []
            for o in ann.get(item.get("band_key") or f"{stem}.jpg", []):
                b = hbat.hyperbola_to_bbox(o, (w, h))
                if b is not None:
                    cand.append(([b["x1"], b["y1"], round(b["x1"] + b["width"], 2), round(b["y1"] + b["height"], 2)], o))
            union = np.zeros((h, w), bool)
            for j, g in enumerate(item["boxes"]):
                best, bo = 1e9, None
                for box, o in cand:
                    d = max(abs(a - b) for a, b in zip(box, g))
                    if d < best:
                        best, bo = d, o
                band = np.zeros((h, w), bool)
                has = bo is not None and best <= 0.06
                if has:
                    band = np.asarray(hbat.band_mask(bo, (w, h))) > 0
                    union |= band
                self.inst_of[(i, j)] = len(inst_img)
                inst_img.append(i); inst_box.append(g); inst_band.append(band); inst_has.append(has)
            ref = cv2.imread(str(dd / "masks_band" / phase / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
            self.n_union_mismatch += int((ref != union).sum())
        self.H, self.W = imgs[0].shape
        assert all(im.shape == (self.H, self.W) for im in imgs), "图像尺寸不一致"
        self.imgs = torch.from_numpy(np.stack(imgs)).unsqueeze(1).to(device)
        n = len(inst_img)
        self.inst_img = torch.tensor(inst_img, dtype=torch.long)
        self.inst_box = torch.tensor(inst_box, dtype=torch.float32).reshape(n, 4)
        self.inst_has = torch.tensor(inst_has, dtype=torch.bool)
        self.bands = torch.from_numpy(np.stack(inst_band).astype(np.float32)).unsqueeze(1).to(device) if n else torch.zeros(0, 1, self.H, self.W, device=device)


# ───────────────────────── 窗口几何 ─────────────────────────
def windows(boxes):
    """候选框 [B,4] (x1,y1,x2,y2) -> 窗口 [B,4] (cx, cy, 宽, 高)。"""
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=MIN_SIDE)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=MIN_SIDE)
    return torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, K * w, K * h], 1)


def to_win(boxes, win):
    """图像坐标的框 -> 窗口归一化坐标（窗口内为 [0,1]，可以超出）。"""
    x0, y0 = win[:, 0] - win[:, 2] / 2, win[:, 1] - win[:, 3] / 2
    return torch.stack([(boxes[:, 0] - x0) / win[:, 2], (boxes[:, 1] - y0) / win[:, 3],
                        (boxes[:, 2] - x0) / win[:, 2], (boxes[:, 3] - y0) / win[:, 3]], 1)


def from_win(u, win):
    x0, y0 = win[:, 0] - win[:, 2] / 2, win[:, 1] - win[:, 3] / 2
    return torch.stack([x0 + u[:, 0] * win[:, 2], y0 + u[:, 1] * win[:, 3], x0 + u[:, 2] * win[:, 2], y0 + u[:, 3] * win[:, 3]], 1)


def crop(src, idx, win, out=S):
    """src [N,1,H,W]，idx [B] 选图，win [B,4] -> [B,1,out,out]。先按 2×out 双线性采样再 2×2 平均，减轻缩小时的混叠；图外补 0。"""
    H, W = src.shape[-2:]
    theta = torch.zeros(len(idx), 2, 3, device=src.device)
    theta[:, 0, 0] = win[:, 2] / W; theta[:, 0, 2] = 2 * win[:, 0] / W - 1
    theta[:, 1, 1] = win[:, 3] / H; theta[:, 1, 2] = 2 * win[:, 1] / H - 1
    grid = F.affine_grid(theta, (len(idx), 1, 2 * out, 2 * out), align_corners=False)
    return F.avg_pool2d(F.grid_sample(src[idx], grid, mode="bilinear", padding_mode="zeros", align_corners=False), 2)


def rect_mask(u, out=S):
    """窗口归一化坐标的框 [B,4] -> 填实的矩形 mask [B,1,out,out]。"""
    c = (torch.arange(out, device=u.device, dtype=torch.float32) + 0.5) / out
    mx = ((c[None] >= u[:, 0:1]) & (c[None] <= u[:, 2:3])).float()
    my = ((c[None] >= u[:, 1:2]) & (c[None] <= u[:, 3:4])).float()
    return (my[:, :, None] * mx[:, None, :]).unsqueeze(1)


def clip_boxes(b, W, H):
    b = b.clone()
    b[:, 0::2] = b[:, 0::2].clamp(0, W); b[:, 1::2] = b[:, 1::2].clamp(0, H)
    x1, x2 = torch.minimum(b[:, 0], b[:, 2]), torch.maximum(b[:, 0], b[:, 2])
    y1, y2 = torch.minimum(b[:, 1], b[:, 3]), torch.maximum(b[:, 1], b[:, 3])
    return torch.stack([x1, y1, torch.maximum(x2, x1 + 1.0), torch.maximum(y2, y1 + 1.0)], 1)


# ───────────────────────── 模型 ─────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class BandUNet(nn.Module):
    """attention 分支：窗口 -> 窗口中间这一条双曲线的 band（S×S，全分辨率，不经过 pooling 就交给回归网络）。"""

    def __init__(self, ch=32):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(1, ch), ConvBlock(ch, ch * 2), ConvBlock(ch * 2, ch * 4)
        self.b = ConvBlock(ch * 4, ch * 4)
        self.d3, self.d2, self.d1 = ConvBlock(ch * 8, ch * 2), ConvBlock(ch * 4, ch), ConvBlock(ch * 2, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        f1 = self.e1(x); f2 = self.e2(F.max_pool2d(f1, 2)); f3 = self.e3(F.max_pool2d(f2, 2))
        y = self.b(F.max_pool2d(f3, 2))
        y = self.d3(torch.cat([F.interpolate(y, scale_factor=2, mode="nearest"), f3], 1))
        y = self.d2(torch.cat([F.interpolate(y, scale_factor=2, mode="nearest"), f2], 1))
        y = self.d1(torch.cat([F.interpolate(y, scale_factor=2, mode="nearest"), f1], 1))
        return self.out(y)


class BoxRegressor(nn.Module):
    """[窗口, attention map] 2 通道 -> 窗口归一化坐标的框 (x1, y1, x2, y2)。各 arm 共用、初始化相同。"""

    def __init__(self, ch=32):
        super().__init__()
        self.c1, self.c2, self.c3 = ConvBlock(2, ch), ConvBlock(ch, ch * 2), ConvBlock(ch * 2, ch * 4)
        self.b = ConvBlock(ch * 4, ch * 8)
        self.reduce = nn.Sequential(nn.Conv2d(ch * 8, ch * 2, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(ch * 2 * 36, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))
        nn.init.zeros_(self.fc[-1].weight)
        lo, hi = 0.5 - 0.5 / K, 0.5 + 0.5 / K          # 初始输出 = 候选框本身（不修正）
        with torch.no_grad():
            self.fc[-1].bias.copy_(torch.tensor([lo, lo, hi, hi]))

    def trunk(self, x):
        y = self.c1(x); y = self.c2(F.max_pool2d(y, 2)); y = self.c3(F.max_pool2d(y, 2)); y = self.b(F.max_pool2d(y, 2))
        return F.adaptive_avg_pool2d(self.reduce(y), 6).flatten(1)

    def forward(self, x):
        return self.fc(self.trunk(x).float())


class S2Net(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.arm = arm
        torch.manual_seed(seed)
        self.reg = BoxRegressor()
        self.unet = None
        if arm in ATT_ARMS:
            torch.manual_seed(seed + 10000)
            self.unet = BandUNet()

    def forward(self, x, a_in=None):
        """卷积部分用 bf16 混合精度（与第一阶段相同）+ channels_last；attention 的 logit 和最后的全连接回归用 float32。"""
        a_logit = None
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and x.is_cuda):
            if self.unet is not None:
                a_logit = self.unet(x).float()
                A = torch.sigmoid(a_logit)
            elif a_in is not None:
                A = a_in
            else:
                A = torch.zeros_like(x)
            feat = self.reg.trunk(torch.cat([x, A], 1).contiguous(memory_format=torch.channels_last))
        return self.reg.fc(feat.float()), a_logit, A


def init_fingerprint(model):
    with torch.no_grad():
        return [float(model.reg.c1.block[0].weight.double().sum()), float(model.reg.fc[0].weight.double().sum())]


# ───────────────────────── 训练 ─────────────────────────
def sample_proposals(gt, gen):
    """GT 框 [B,4] -> 抖动后的候选框（模拟第一阶段的误差）。"""
    n = len(gt)
    w, h = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    sw = torch.exp(torch.randn(n, generator=gen) * JIT_SCALE_SIG + math.log(JIT_SCALE_MU)).clamp(*JIT_SCALE_CLIP)
    sh = torch.exp(torch.randn(n, generator=gen) * JIT_SCALE_SIG + math.log(JIT_SCALE_MU)).clamp(*JIT_SCALE_CLIP)
    dx = (torch.randn(n, generator=gen) * JIT_DX_SIG).clamp(-JIT_DX_CLIP, JIT_DX_CLIP) * w
    dy = (torch.randn(n, generator=gen) * JIT_DY_SIG).clamp(-JIT_DY_CLIP, JIT_DY_CLIP) * h
    cx, cy = (gt[:, 0] + gt[:, 2]) / 2 + dx, (gt[:, 1] + gt[:, 3]) / 2 + dy
    pw, ph = sw * w, sh * h
    return torch.stack([cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2], 1)


def attn_loss(a_logit, target, valid):
    """BCE + Dice（与第一阶段相同的形式；Dice 按窗口算），只对 valid 的窗口计。"""
    bce = F.binary_cross_entropy_with_logits(a_logit, target, reduction="none").mean((1, 2, 3))
    p = torch.sigmoid(a_logit)
    dice = 1.0 - 2.0 * (p * target).sum((1, 2, 3)) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)
    v = valid.float()
    return ((bce + dice) * v).sum() / v.sum().clamp(min=1.0)


def train_arm(arm, fold, seed, tr, device, epochs=EPOCHS, log_every=10):
    model = S2Net(arm, seed).to(device).to(memory_format=torch.channels_last)
    fp = init_fingerprint(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(1000 * seed + fold)      # 窗口抖动 / 顺序 / 翻转：各 arm 相同
    M = len(tr.inst_img)
    log = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.cat([torch.randperm(M, generator=gen) for _ in range(REPEAT)])
        props = sample_proposals(tr.inst_box[order], gen)
        flips = torch.rand(len(order), generator=gen) < 0.5
        tot = {"box": 0.0, "att": 0.0, "n": 0}
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            win = windows(props[i:i + BATCH]).to(device)
            x = crop(tr.imgs, tr.inst_img[idx].to(device), win)
            tgt = to_win(tr.inst_box[idx].to(device), win)
            m_band = crop(tr.bands, idx.to(device), win)
            fl = flips[i:i + BATCH].to(device)
            if fl.any():
                x = torch.where(fl[:, None, None, None], x.flip(-1), x)
                m_band = torch.where(fl[:, None, None, None], m_band.flip(-1), m_band)
                tgt = torch.where(fl[:, None], torch.stack([1 - tgt[:, 2], tgt[:, 1], 1 - tgt[:, 0], tgt[:, 3]], 1), tgt)
            pred, a_logit, _ = model(x, m_band if arm == "oracle" else None)
            l_box = F.l1_loss(pred, tgt)
            l_att = torch.zeros((), device=device)
            if arm == "band":
                l_att = attn_loss(a_logit, m_band, tr.inst_has[idx].to(device))
            elif arm == "boxmask":
                l_att = attn_loss(a_logit, rect_mask(tgt), torch.ones(len(idx), dtype=torch.bool, device=device))
            loss = l_box + LAM * l_att
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot["box"] += l_box.item() * len(idx); tot["att"] += float(l_att) * len(idx); tot["n"] += len(idx)
        log.append({"epoch": ep, "box_l1": tot["box"] / tot["n"], "att_loss": tot["att"] / tot["n"]})
        if ep % log_every == 0 or ep == epochs:
            print(f"  [{arm}] epoch {ep}/{epochs}  box L1 {log[-1]['box_l1']:.4f}  att {log[-1]['att_loss']:.4f}", flush=True)
    return model, log, fp, (time.time() - t0) / 60


# ───────────────────────── 推理：修正第一阶段的候选框 ─────────────────────────
def mask_boxes(A, t, win, fallback):
    """attention map [B,1,S,S] 按 t 二值化 -> 最大连通块的外接框（图像坐标）；没有像素时退回 fallback（回归框）。"""
    a = (A.squeeze(1).cpu().numpy() >= t).astype(np.uint8)
    out = fallback.clone()
    u = torch.zeros(len(a), 4)
    ok = torch.zeros(len(a), dtype=torch.bool)
    for i, m in enumerate(a):
        if not m.any():
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = (int(stats[k, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        u[i] = torch.tensor([x / S, y / S, (x + w) / S, (y + h) / S]); ok[i] = True
    if ok.any():
        out[ok] = from_win(u.to(win.device), win)[ok]
    return out


@torch.no_grad()
def refine_phase(model, arm, ph, preds, device, chunk=256):
    """对一个 phase 的全部候选框做两次迭代修正。返回 {stem: {...}}（坐标为图像像素）。"""
    model.eval()
    stems, cand, cimg = [], [], []
    for i, stem in enumerate(ph.stems):
        for d in preds.get(stem, []):
            stems.append(stem); cand.append(d); cimg.append(i)
    if not cand:
        return {s: {"src": [], "reg1": [], "reg2": [], "mb1": {str(t): [] for t in T_GRID}, "mb2": {str(t): [] for t in T_GRID},
                    "gt": [], "iou0": [], "mask_iou": []} for s in ph.stems}
    cand = torch.tensor(cand, dtype=torch.float32)
    cimg_t = torch.tensor(cimg, dtype=torch.long)
    box0 = cand[:, :4]
    # 每个候选框对着哪个 GT（IoU 最大且 ≥ ASSIGN_IOU；只用于 oracle arm 和诊断）
    gt_j, iou0, inst = [], [], []
    for c in range(len(cand)):
        gts = ph.gt[stems[c]]["boxes"]
        ious = [box_iou(box0[c].tolist(), g) for g in gts]
        j = int(np.argmax(ious)) if ious else -1
        if j >= 0 and ious[j] < ASSIGN_IOU:
            j = -1
        gt_j.append(j); iou0.append(max(ious) if ious else 0.0); inst.append(ph.inst_of[(cimg[c], j)] if j >= 0 else -1)
    inst_t = torch.tensor(inst, dtype=torch.long)
    res = {}
    box = box0.clone()
    mask_iou = torch.full((len(cand),), float("nan"))
    for it in (1, 2):
        regs, mbs = [], {t: [] for t in T_GRID}
        for s in range(0, len(cand), chunk):
            sl = slice(s, s + chunk)
            win = windows(box[sl]).to(device)
            x = crop(ph.imgs, cimg_t[sl].to(device), win)
            ii = inst_t[sl]
            m_gt = torch.zeros_like(x)
            if (ii >= 0).any():
                m_gt[ii >= 0] = crop(ph.bands, ii[ii >= 0].to(device), win[ii >= 0])
            pred, a_logit, A = model(x, m_gt if arm == "oracle" else None)
            reg = clip_boxes(from_win(pred, win), ph.W, ph.H)
            regs.append(reg.cpu())
            for t in T_GRID:
                mbs[t].append(clip_boxes(mask_boxes(A, t, win, reg), ph.W, ph.H).cpu() if arm in ("boxmask", "band") else reg.cpu())
            if it == 1 and arm in ("boxmask", "band"):
                # 诊断：预测的 mask 与真值 mask（band arm = 实例 band；boxmask arm = 实例框）的 IoU，都在窗口坐标里算
                if arm == "band":
                    tgt = m_gt >= 0.5
                    valid = (ii >= 0) & torch.tensor([bool(ph.inst_has[k]) if k >= 0 else False for k in ii.tolist()])
                else:
                    gbox = torch.stack([ph.inst_box[k] if k >= 0 else torch.zeros(4) for k in ii.tolist()]).to(device)
                    tgt = rect_mask(to_win(gbox, win)) >= 0.5
                    valid = ii >= 0
                p = A >= 0.5
                inter = (p & tgt).sum((1, 2, 3)).float(); union = (p | tgt).sum((1, 2, 3)).float()
                mi = (inter / union.clamp(min=1)).cpu()
                mi[~valid] = float("nan")
                mask_iou[sl] = mi
        res[f"reg{it}"] = torch.cat(regs)
        res[f"mb{it}"] = {t: torch.cat(v) for t, v in mbs.items()}
        box = res[f"reg{it}"]
    out = {}
    by_stem = {}
    for c, stem in enumerate(stems):
        by_stem.setdefault(stem, []).append(c)
    r2 = lambda t: [[round(float(v), 2) for v in row] for row in t.tolist()]  # noqa: E731
    for stem in ph.stems:
        cs = by_stem.get(stem, [])
        ix = torch.tensor(cs, dtype=torch.long)
        out[stem] = {"src": [[round(float(v), 2) for v in cand[c, :4]] + [float(cand[c, 4])] for c in cs],
                     "reg1": r2(res["reg1"][ix]) if cs else [], "reg2": r2(res["reg2"][ix]) if cs else [],
                     "mb1": {str(t): (r2(res["mb1"][t][ix]) if cs else []) for t in T_GRID},
                     "mb2": {str(t): (r2(res["mb2"][t][ix]) if cs else []) for t in T_GRID},
                     "gt": [gt_j[c] for c in cs], "iou0": [round(iou0[c], 4) for c in cs],
                     "mask_iou": [None if math.isnan(float(mask_iou[c])) else round(float(mask_iou[c]), 4) for c in cs]}
    return out


def worker(fold, seed, arms, epochs, force=False):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann = json.load(open(HBAT_JSON, encoding="utf-8"))
    phases = {p: Phase(fold, p, ann, device) for p in ("train", "val", "test")}
    tr = phases["train"]
    print(f"fold {fold} seed {seed}: train 实例 {len(tr.inst_img)}（有 band {int(tr.inst_has.sum())}），"
          f"实例 band 并集与 masks_band 不一致的像素：" + " / ".join(f"{p} {phases[p].n_union_mismatch}" for p in phases), flush=True)
    src = run_dir(SRC_EXP, fold, seed)
    preds = {p: json.load(open(src / f"preds_{p}.json", encoding="utf-8")) for p in ("val", "test")}
    for arm in arms:
        cd = cache_dir(arm, fold, seed)
        if not force and all((cd / f"out_{p}.json").exists() for p in ("val", "test")) and (cd / "info.json").exists():
            print(f"  [{arm}] 已有输出，跳过", flush=True)
            continue
        cd.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed); torch.manual_seed(seed)
        model, log, fp, minutes = train_arm(arm, fold, seed, tr, device, epochs)
        for p in ("val", "test"):
            json.dump(refine_phase(model, arm, phases[p], preds[p], device), open(cd / f"out_{p}.json", "w"))
        info = {"arm": arm, "fold": fold, "seed": seed, "source_exp": SRC_EXP, "init_fingerprint": fp, "train_minutes": round(minutes, 3),
                "n_train_instances": len(tr.inst_img), "n_train_with_band": int(tr.inst_has.sum()),
                "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
                "hparams": {"S": S, "K": K, "amp": "bfloat16" if AMP else None, "epochs": epochs, "repeat": REPEAT, "batch": BATCH, "lr": LR, "lam_att": LAM,
                            "jitter": {"scale": [JIT_SCALE_MU, JIT_SCALE_SIG, *JIT_SCALE_CLIP], "dx": [JIT_DX_SIG, JIT_DX_CLIP], "dy": [JIT_DY_SIG, JIT_DY_CLIP]}},
                "torch": torch.__version__, "train_log": log}
        json.dump(info, open(cd / "info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  [{arm}] done in {minutes:.2f} min  指纹 {fp[0]:.6f}", flush=True)


def launch(arms, folds, seeds, jobs, epochs, force):
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = []
    for fold, seed in itertools.product(folds, seeds):
        if not force and all((cache_dir(a, fold, seed) / "info.json").exists() for a in arms):
            continue
        todo.append((fold, seed))
    print(f"{len(todo)} 个 (fold, seed) 要跑，arms = {arms}  ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    running, t0, done = [], time.time(), 0
    while todo or running:
        while todo and len(running) < jobs:
            fold, seed = todo.pop(0)
            tag = "" if SRC_EXP == "hbatbox_aug_none" else f"{SRC_EXP}_"
            lf = open(CACHE / f"log_{tag}f{fold}_seed{seed}.txt", "a", encoding="utf-8")
            cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "worker", "--fold", str(fold), "--seed", str(seed),
                   "--arms", *arms, "--epochs", str(epochs), "--src", SRC_EXP] + (["--force"] if force else [])
            running.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE)), fold, seed, lf))
        time.sleep(2)
        still = []
        for p, fold, seed, lf in running:
            if p.poll() is None:
                still.append((p, fold, seed, lf)); continue
            lf.close(); done += 1
            print(f"[{done}] f{fold} seed{seed} rc={p.returncode}  {(time.time() - t0) / 60:.1f} min", flush=True)
        running = still
    print(f"ALL_DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


# ───────────────────────── build：选参 + 写新实验的 run 目录 ─────────────────────────
def nms(dets, thr):
    if thr is None:
        return dets
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(box_iou(d, k) <= thr for k in keep):
            keep.append(d)
    return keep


def apply_cfg(out, cfg):
    """cfg = (来源, 迭代, mask 阈值或 None, NMS 的 IoU 或 None) -> {stem: [[x1,y1,x2,y2,score]]}"""
    kind, it, t, nms_thr = cfg
    res = {}
    for stem, o in out.items():
        boxes = [b[:4] for b in o["src"]] if kind == "src" else (o[f"reg{it}"] if kind == "reg" else o[f"mb{it}"][str(t)])
        res[stem] = nms([list(b) + [s[4]] for b, s in zip(boxes, o["src"])], nms_thr)
    return res


GRID_REG = [("reg", it, None, n) for it in (1, 2) for n in NMS_GRID]
GRID_MB = [("mb", it, t, n) for it in (1, 2) for t in T_GRID for n in NMS_GRID]
GRID_SRC = [("src", 0, None, n) for n in NMS_GRID if n is not None]     # 对照：第一阶段的框不修正，只加 NMS
def make_variants(src):
    return {
        f"{src}_nms":          ("plain", GRID_SRC),      # 只加 NMS 的对照（读 plain arm 输出里保存的原始候选框）
        f"{src}_s2plain":      ("plain", GRID_REG),
        f"{src}_s2unsup":      ("unsup", GRID_REG),
        f"{src}_s2boxmask":    ("boxmask", GRID_REG),
        f"{src}_s2band":       ("band", GRID_REG),
        f"{src}_s2oracle":     ("oracle", GRID_REG),
        f"{src}_s2boxmask_mb": ("boxmask", GRID_MB),
        f"{src}_s2band_mb":    ("band", GRID_MB),
    }


VARIANTS = make_variants(SRC_EXP)


def pooled(preds_by_fold, gt_by_fold):
    sc, tp, n = [], [], 0
    for f in FOLDS:
        s, t, k = collect(preds_by_fold[f], gt_by_fold[f]); sc.append(s); tp.append(t); n += k
    return np.concatenate(sc), np.concatenate(tp), n


def build(variants, seeds=SEEDS):
    gt_val = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["val"] for f in FOLDS}
    gt_test = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["test"] for f in FOLDS}
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = json.load(open(OUT / "chosen.json", encoding="utf-8")) if (OUT / "chosen.json").exists() else {}
    sections = {}
    if (OUT / "sweep.md").exists():
        cur = None
        for line in (OUT / "sweep.md").read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                cur = line[3:].split("（")[0].strip(); sections[cur] = []
            if cur is not None:
                sections[cur].append(line)
    for variant in variants:
        arm, grid = VARIANTS[variant]
        print(f"== {variant}（arm {arm}，{len(grid)} 组参数）", flush=True)
        val_f1 = {c: [] for c in grid}; test_f1 = {c: [] for c in grid}
        outs = {}
        for seed in seeds:
            for f in FOLDS:
                for p in ("val", "test"):
                    outs[(seed, f, p)] = json.load(open(cache_dir(arm, f, seed) / f"out_{p}.json"))
            for c in grid:
                pv = {f: apply_cfg(outs[(seed, f, "val")], c) for f in FOLDS}
                pt = {f: apply_cfg(outs[(seed, f, "test")], c) for f in FOLDS}
                sv, tv, nv = pooled(pv, gt_val); conf, f1v = select_conf(sv, tv, nv)
                st, tt, nt = pooled(pt, gt_test)
                val_f1[c].append(f1v); test_f1[c].append(prf_at(st, tt, nt, conf)["F1"])
        best = max(grid, key=lambda c: np.mean(val_f1[c]))
        chosen[variant] = {"arm": arm, "source_exp": SRC_EXP, "config": list(best), "config_keys": ["box", "iter", "mask_thresh", "nms_iou"],
                           "val_F1_mean": float(np.mean(val_f1[best])), "test_F1_mean_at_chosen": float(np.mean(test_f1[best]))}
        lines = [f"## {variant}（arm {arm}）", "", f"选中（按 {len(seeds)} seed 合并验证 F1 均值）：{best}", "",
                 "| 参数（框来源, 迭代, mask 阈值, NMS IoU） | 合并验证 F1 均值 | 测试 F1 均值 ± std（只看敏感性） |", "|---|---|---|"]
        for c in grid:
            lines.append(f"| {c} | {np.mean(val_f1[c]):.3f} | {np.mean(test_f1[c]):.3f} ± {np.std(test_f1[c], ddof=1):.3f}{' **←**' if c == best else ''} |")
        lines.append(""); sections[variant] = lines
        for seed in seeds:
            for f in FOLDS:
                rd = run_dir(variant, f, seed); rd.mkdir(parents=True, exist_ok=True)
                json.dump(apply_cfg(outs[(seed, f, "val")], best), open(rd / "preds_val.json", "w"))
                json.dump(apply_cfg(outs[(seed, f, "test")], best), open(rd / "preds_test.json", "w"))
                info = json.load(open(run_dir(SRC_EXP, f, seed) / "run_info.json", encoding="utf-8"))
                s2 = json.load(open(cache_dir(arm, f, seed) / "info.json", encoding="utf-8"))
                info.update(exp=variant, source_exp=SRC_EXP, source_run=str(run_dir(SRC_EXP, f, seed).relative_to(HERE)),
                            stage2={"arm": arm, "config": list(best), "config_keys": ["box", "iter", "mask_thresh", "nms_iou"],
                                    "selection": f"global on pooled val F1 over {len(seeds)} seeds",
                                    **{k: s2[k] for k in ("init_fingerprint", "train_minutes", "params_M", "hparams", "n_train_instances", "n_train_with_band")},
                                    "final_box_l1": s2["train_log"][-1]["box_l1"], "final_att_loss": s2["train_log"][-1]["att_loss"]})
                json.dump(info, open(rd / "run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                evaluate_run(rd, data_dir(f))
                for alt in ("official", "hbat"):
                    if (data_dir(f) / f"gt_boxes_{alt}.json").exists():
                        evaluate_run(rd, data_dir(f), gt_name=f"gt_boxes_{alt}.json", out_name=f"metrics_{alt}.json")
        print(f"  选中 {best}：验证 F1 {chosen[variant]['val_F1_mean']:.3f}，测试 F1 {chosen[variant]['test_F1_mean_at_chosen']:.3f}", flush=True)
    json.dump(chosen, open(OUT / "chosen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    body = "\n".join(l for v in sections.values() for l in v)
    (OUT / "sweep.md").write_text("# 第二阶段（Round 14）参数敏感性（选参只用验证列，测试列仅供查看）\n\n" + body.rstrip("\n") + "\n", encoding="utf-8")


# ───────────────────────── diag：窗口级诊断 ─────────────────────────
def diag(arms, seeds=SEEDS, score_min=0.3):
    """只看第一阶段 score ≥ score_min 且对着某个 GT（IoU ≥ ASSIGN_IOU）的测试候选框：修正前后与该 GT 的 IoU；预测 mask 的 IoU 按 GT 宽度分层。"""
    gt_test = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["test"] for f in FOLDS}
    widths = np.array([g[2] - g[0] for f in FOLDS for it in gt_test[f].values() for g in it["boxes"]])
    q1, q2 = np.percentile(widths, [100 / 3, 200 / 3])
    wname = lambda w: f"narrow (<{q1:.0f}px)" if w < q1 else (f"wide (>={q2:.0f}px)" if w >= q2 else f"medium ({q1:.0f}-{q2:.0f}px)")  # noqa: E731
    strata = [f"narrow (<{q1:.0f}px)", f"medium ({q1:.0f}-{q2:.0f}px)", f"wide (>={q2:.0f}px)"]
    res = {}
    for arm in arms:
        per_seed = []
        for seed in seeds:
            rows = []
            for f in FOLDS:
                out = json.load(open(cache_dir(arm, f, seed) / "out_test.json"))
                for stem, o in out.items():
                    gts = gt_test[f][stem]["boxes"]
                    for c, j in enumerate(o["gt"]):
                        if j < 0 or o["src"][c][4] < score_min:
                            continue
                        g = gts[j]
                        rows.append((wname(g[2] - g[0]), box_iou(o["src"][c], g), box_iou(o["reg1"][c], g), box_iou(o["reg2"][c], g),
                                     box_iou(o["mb1"]["0.5"][c], g), o["mask_iou"][c]))
            d = {"n": len(rows)}
            for name, sel in [("all", lambda r: True)] + [(s, (lambda r, s=s: r[0] == s)) for s in strata]:
                rr = [r for r in rows if sel(r)]
                if not rr:
                    continue
                a = np.array([[r[1], r[2], r[3], r[4]] for r in rr])
                mi = np.array([r[5] for r in rr if r[5] is not None], float)
                d[name] = {"n": len(rr), "iou_src": float(a[:, 0].mean()), "iou_reg1": float(a[:, 1].mean()), "iou_reg2": float(a[:, 2].mean()),
                           "iou_mb1": float(a[:, 3].mean()), "hit_src": float((a[:, 0] >= 0.5).mean()), "hit_reg1": float((a[:, 1] >= 0.5).mean()),
                           "hit_reg2": float((a[:, 2] >= 0.5).mean()), "hit_mb1": float((a[:, 3] >= 0.5).mean()),
                           "hit70_src": float((a[:, 0] >= 0.7).mean()), "hit70_reg2": float((a[:, 2] >= 0.7).mean()),
                           "mask_iou": float(mi.mean()) if len(mi) else None}
            per_seed.append(d)
        res[arm] = per_seed
    lines = [f"# 第二阶段窗口级诊断（测试集；第一阶段 score ≥ {score_min} 且与某个 GT 的 IoU ≥ {ASSIGN_IOU} 的候选框；{len(seeds)} 个 seed 的均值）", "",
             "IoU 都是候选框与它对着的那个 GT 的 IoU；命中率 = IoU ≥ 0.5 的比例。mask IoU：band arm 是预测 band 与该实例真实 band 的 IoU，boxmask arm 是预测掩码与实例框的 IoU（窗口坐标）。", ""]
    for name in ["all"] + strata:
        lines += [f"## {name}", "", "| arm | n/seed | IoU 修正前 | 回归 1 次 | 回归 2 次 | mask 定框(t=0.5) | 命中率 修正前 | 回归 1 次 | 回归 2 次 | mask 定框 | IoU≥0.7 前 → 回归 2 次 | mask IoU |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            ds = [d[name] for d in res[arm] if name in d]
            if not ds:
                continue
            m = lambda k: np.mean([d[k] for d in ds if d[k] is not None]) if any(d[k] is not None for d in ds) else float("nan")  # noqa: E731
            lines.append(f"| {arm} | {m('n'):.0f} | {m('iou_src'):.3f} | {m('iou_reg1'):.3f} | {m('iou_reg2'):.3f} | {m('iou_mb1'):.3f} | {m('hit_src'):.3f} | "
                         f"{m('hit_reg1'):.3f} | {m('hit_reg2'):.3f} | {m('hit_mb1'):.3f} | {m('hit70_src'):.3f} → {m('hit70_reg2'):.3f} | {m('mask_iou'):.3f} |")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(res, open(OUT / "diag.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "worker", "build", "diag"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--src", default=SRC_EXP, help="第一阶段的来源实验（候选框）。默认 baseline；例如 hbatbox_aug_abs = 带 attention 的第一阶段")
    ap.add_argument("--variants", nargs="+", default=None, help="build 阶段要生成的变体（默认该来源下已有输出的全部 arm）")
    args = ap.parse_args()
    set_source(args.src)
    if args.variants is None:
        args.variants = [v for v, (a, _) in VARIANTS.items() if (cache_dir(a, FOLDS[0], args.seeds[0]) / "info.json").exists()] or list(VARIANTS)
    if args.stage == "train":
        launch(args.arms, args.folds, args.seeds, args.jobs, args.epochs, args.force)
    elif args.stage == "worker":
        worker(args.fold, args.seed, args.arms, args.epochs, args.force)
    elif args.stage == "build":
        build(args.variants, args.seeds)
    else:
        diag([a for a in args.arms if (cache_dir(a, 0, args.seeds[0]) / "out_test.json").exists()], args.seeds)
