"""实验 0：以曲线为单位的检测器（双曲线聚合层 = 可学习的 migration / Hough）。设计见 README.md。

    python train_curve.py --data ../shared/datasets/ds2_f0_s100_hbatbox --exp curve_hough --layout hough --seed 0

每个假设 = (顶点格子 28×28, slope 档位 8)。沿假设的曲线在 M 个水平位置 × L 个横向偏移上双线性采样 56×56 证据特征，
逐点 MLP → 沿弧门控（band attention，span 监督）→ 两翼加权聚合 → 分数 + 曲线参数。框由曲线算（与 HBAT 推框公式相同）。

layout   hough   = 沿曲线采样（完整方法）
         local   = 只看顶点一个位置（M = 1），没有沿曲线聚合
         boxgrid = 采样点数相同，散布在曲线外接框里，不沿曲线
targets  band      = 曲线参数、半宽、门控目标都来自 HBAT 标注
         boxpseudo = 全部由框反推的伪曲线（只用框标注）

训练设置与 exp1_basic/train_basic.py 相同：Adam 5e-4，batch 8，bf16，120 轮，最后一轮权重，几何水平翻转，
DataLoader 的顺序和翻转与 hbatbox_flipfix_* 逐样本相同（--workers 1）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "exp1_basic"
SHARED = HERE.parent / "shared"
REPO = HERE.parents[2]
sys.path.insert(0, str(EXP1))
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(REPO / "tools" / "annotation"))

import train_basic as tb  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG, backbone blocks, focal loss, seeding)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from evaluate import evaluate_run, box_iou  # noqa: E402
import hbat  # noqa: E402

device = tb.device

# ───────────────────────── 几何常量 ─────────────────────────
IMG = 224
HSTRIDE = 8                 # 假设格子步长 → 28×28
GH = GW = IMG // HSTRIDE
FSTRIDE = 4                 # 证据特征步长 → 56×56
FH = FW = IMG // FSTRIDE
SLOPES = np.exp(np.linspace(math.log(1.1), math.log(5.2), 8)).astype(np.float64)
LOG_S0, LOG_DS = math.log(SLOPES[0]), math.log(SLOPES[1] / SLOPES[0])
NS = len(SLOPES)
DX_POS = [0, 3, 6, 10, 14, 19, 25, 32, 40, 49, 60, 72, 86, 100]
DX_FULL = [-d for d in reversed(DX_POS[1:])] + DX_POS          # 27 个水平位置，中心是顶点
L_OFF = [-4.0, 0.0, 4.0]                                         # 横向（竖直）偏移，px
W_REF, TH_REF = 32.0, 16.0                                       # 半宽、厚度的 log 回归基准


def curve_y(dx, yv, s):
    """HBAT 模型的中心线高度（hbat.hyperbola_ab：a = max(1, |y_v|)，b = a / slope）。"""
    a = max(1.0, abs(yv)); b = a / max(1e-3, s)
    return yv + a * (math.sqrt(1.0 + (dx / b) ** 2) - 1.0)


def curve_box(xv, yv, s, w, th):
    """曲线 → 框，与 hbat.hyperbola_to_bbox 相同（tail = 0），裁到图内。"""
    w = max(1.0, w); th = max(1.0, th)
    y2 = curve_y(w, yv, s) + th / 2.0
    x1, x2, y1 = xv - w, xv + w, yv - th / 2.0
    x1, x2 = max(0.0, x1), min(float(IMG), x2)
    y1, y2 = max(0.0, y1), min(float(IMG), y2)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2, y2]


def box_to_curve(box, th=TH_REF):
    """只用框反推的伪曲线：顶点 = 框上边中点下 th/2，半宽 = 框宽 / 2，slope 由框高解出。"""
    x1, y1, x2, y2 = box
    th = min(th, max(2.0, (y2 - y1) / 2.0))
    xv, yv, w = (x1 + x2) / 2.0, y1 + th / 2.0, max(1.0, (x2 - x1) / 2.0)
    y_edge = max(yv + 1.0, y2 - th / 2.0)
    a = max(1.0, abs(yv)); q = (y_edge - yv) / a + 1.0
    s = a * math.sqrt(max(q * q - 1.0, 1e-6)) / w
    return dict(xv=xv, yv=yv, s=float(np.clip(s, 0.5, 10.0)), w=w, th=th, src="pseudo")


# ───────────────────────── 采样矩阵（固定，与图像无关） ─────────────────────────
def _interp_rows(pos, n):
    """pos: (P,) 网格坐标 → (P, n) 双线性权重，越界部分为 0（等价于 grid_sample 的 zeros padding）。"""
    m = np.zeros((len(pos), n), np.float32)
    f = np.floor(pos).astype(np.int64); t = pos - f
    for p in range(len(pos)):
        for q, wgt in ((f[p], 1.0 - t[p]), (f[p] + 1, t[p])):
            if 0 <= q < n:
                m[p, q] += wgt
    return m


def build_geometry(layout):
    """返回 Sx (M, GW, FW)、R (S, M, L, GH, FH)、inimg (S, M, GH, GW)、dx 列表。
    样本 (k, m, l) 在假设格子 (i, j) 的位置：x = cx_j + dx_m，y = py(i, k, m, l)。"""
    dxs = [0] if layout == "local" else DX_FULL
    M, L = len(dxs), len(L_OFF)
    c = M // 2
    cx = (np.arange(GW) + 0.5) * HSTRIDE
    cy = (np.arange(GH) + 0.5) * HSTRIDE
    Sx = np.stack([_interp_rows((cx + dx) / FSTRIDE - 0.5, FW) for dx in dxs])
    R = np.zeros((NS, M, L, GH, FH), np.float32)
    inimg = np.zeros((NS, M, GH, GW), np.float32)
    golden = 0.6180339887
    for k, s in enumerate(SLOPES):
        for m, dx in enumerate(dxs):
            for li, lo in enumerate(L_OFF):
                py = np.empty(GH)
                for i, yv in enumerate(cy):
                    if layout == "boxgrid":
                        if m == c:
                            frac = 0.0
                        else:
                            frac = (abs(m - c) * golden + (0.5 if m < c else 0.0)) % 1.0
                        py[i] = yv + frac * (curve_y(DX_POS[-1], yv, s) - yv) + lo
                    else:
                        py[i] = curve_y(dx, yv, s) + lo
                R[k, m, li] = _interp_rows(py / FSTRIDE - 0.5, FH)
                inx = ((cx + dx) >= 0) & ((cx + dx) < IMG)
                iny = (py >= 0) & (py < IMG)
                inimg[k, m] += (iny[:, None] & inx[None, :]).astype(np.float32) / L
    return torch.from_numpy(Sx), torch.from_numpy(R), torch.from_numpy(inimg), dxs


# ───────────────────────── 模型 ─────────────────────────
class CurveNet(nn.Module):
    def __init__(self, layout, base_ch=32, ce=16, hid=32):
        super().__init__()
        self.layout = layout
        # backbone：与 AttnBBoxNet 相同的层和创建顺序（同一 seed 下逐参数相同）
        self.down1 = tb.DownBlock(1, base_ch)
        self.down2 = tb.DownBlock(base_ch, base_ch * 2)
        self.down3 = tb.DownBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = tb.ConvBlock(base_ch * 4, base_ch * 8)
        mid = base_ch * 8
        # 各 arm 形状相同的 head 层（先建，同一 seed 下逐参数相同）
        self.up = tb.ConvBlock(mid + base_ch * 4, base_ch * 4)
        self.proj_e = nn.Conv2d(base_ch * 4, ce, 1)
        self.proj_b = nn.Conv2d(mid, hid, 1)
        L = len(L_OFF)
        self.pt1 = nn.Linear(ce * L, hid)
        self.w_in = nn.Parameter(torch.randn(hid) * 0.1)
        self.slope_emb = nn.Parameter(torch.randn(NS, hid) * 0.1)
        self.pt2 = nn.Linear(hid, hid)
        self.gate = nn.Linear(hid, 1)
        self.head1 = nn.Linear(4 * hid + 2, 64)
        self.head2 = nn.Linear(64, 6)
        with torch.no_grad():
            self.head2.bias[0] = -2.19
        Sx, R, inimg, dxs = build_geometry(layout)
        self.register_buffer("Sx", Sx, persistent=False)
        self.register_buffer("R", R, persistent=False)
        self.register_buffer("inimg", inimg, persistent=False)
        self.dxs = dxs
        M = len(dxs)
        # 与 M 有关的参数最后建（arm 之间唯一不同的初始化）
        self.pos_emb = nn.Parameter(torch.randn(M, hid) * 0.1)
        self.gate_bias = nn.Parameter(torch.zeros(NS, M))

    def forward(self, x):
        _, x = self.down1(x); _, x = self.down2(x)
        f3, x = self.down3(x)
        b = self.bottleneck(x)                                                     # (B, 256, 28, 28)
        e = self.up(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), f3], dim=1))
        E = self.proj_e(e)                                                         # (B, C, 56, 56)
        Sx, R = self.Sx.to(E.dtype), self.R.to(E.dtype)
        Ex = torch.einsum("mwv,bchv->bcmhw", Sx, E)                               # (B, C, M, 56, 28)
        samp = torch.einsum("kmlih,bcmhw->bkmiwcl", R, Ex)                         # (B, S, M, 28, 28, C, L)
        B, S, M = samp.shape[:3]
        samp = samp.reshape(B, S, M, GH, GW, -1)
        inimg = self.inimg.to(samp.dtype)
        u = self.pt1(samp) + inimg[None, ..., None] * self.w_in \
            + self.pos_emb[None, None, :, None, None, :] + self.slope_emb[None, :, None, None, None, :]
        u = F.relu(self.pt2(F.relu(u)))                                           # (B, S, M, 28, 28, H)
        a = torch.sigmoid(self.gate(u).squeeze(-1) + self.gate_bias[None, :, :, None, None])   # (B, S, M, 28, 28)
        c = M // 2
        u0 = u[:, :, c]
        if M > 1:
            aL, aR = a[:, :, :c], a[:, :, c + 1:]
            zL = (aL[..., None] * u[:, :, :c]).sum(2) / (aL.sum(2)[..., None] + 1.0)
            zR = (aR[..., None] * u[:, :, c + 1:]).sum(2) / (aR.sum(2)[..., None] + 1.0)
            gL, gR = aL.mean(2), aR.mean(2)
        else:
            zL = zR = torch.zeros_like(u0)
            gL = gR = torch.zeros_like(u0[..., 0])
        fb = self.proj_b(b).permute(0, 2, 3, 1)[:, None].expand(-1, S, -1, -1, -1)
        h = torch.cat([zL, zR, u0, fb, gL[..., None], gR[..., None]], dim=-1)
        out = self.head2(F.relu(self.head1(h)))                                    # (B, S, 28, 28, 6)
        return out, a


def build_model(layout, seed):
    torch.manual_seed(seed)
    return CurveNet(layout).to(device)


# ───────────────────────── 数据 ─────────────────────────
class CurveDataset(Dataset):
    def __init__(self, data_dir, phase, targets="band", hm_sigma=2.0, dxs=DX_FULL):
        self.data_dir, self.phase, self.targets = Path(data_dir), phase, targets
        self.hm_sigma, self.dxs = hm_sigma, np.array(dxs, np.float32)
        gt_all = json.load(open(self.data_dir / "gt_boxes.json", encoding="utf-8"))
        self.gt = gt_all[phase]
        self.stems = sorted(self.gt)
        man = json.load(open(self.data_dir / "manifest.json", encoding="utf-8"))
        ann = json.load(open(man["hbat_json"], encoding="utf-8"))
        img_dir = self.data_dir / "images" / phase
        self.files = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
        self.params, self.n_band, self.n_pseudo = {}, 0, 0
        for stem in self.stems:
            g = self.gt[stem]
            assert (g["width"], g["height"]) == (IMG, IMG), (stem, g["width"], g["height"])
            boxes = g["boxes"]
            curves = []
            used = [False] * len(boxes)
            if targets == "band" and g.get("band_key"):
                for o in ann.get(g["band_key"], []):
                    o = hbat.normalize_obj(o)
                    if not hbat.has_band(o):
                        continue
                    o = hbat.cap_span_to_bottom(o, (IMG, IMG))
                    cv = dict(xv=o["x_vertex"], yv=o["y_vertex"], s=o["slope"], w=o["span"] / 2.0, th=o["thickness"],
                              src="band", visibility=o.get("visibility"), completeness=o.get("completeness"))
                    bx = curve_box(cv["xv"], cv["yv"], cv["s"], cv["w"], cv["th"])
                    if bx is None:
                        continue
                    best, bj = 0.0, -1
                    for j, gb in enumerate(boxes):
                        if not used[j]:
                            iou = box_iou(bx, gb)
                            if iou > best:
                                best, bj = iou, j
                    if best > 0.95:
                        used[bj] = True
                        curves.append(cv)
            for j, gb in enumerate(boxes):
                if not used[j]:
                    curves.append(box_to_curve(gb))
            self.params[stem] = curves
            self.n_band += sum(c["src"] == "band" for c in curves)
            self.n_pseudo += sum(c["src"] == "pseudo" for c in curves)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        return self.encode(idx, flip=False)

    def encode(self, idx, flip=False):
        stem = self.stems[idx]
        img = np.array(Image.open(self.files[stem]).convert("L").resize((IMG, IMG), Image.BILINEAR), np.float32) / 255.0
        if flip:
            img = img[:, ::-1].copy()
        hm = np.zeros((NS, GH, GW), np.float32)
        peak = np.zeros((NS, GH, GW), np.float32)
        ks = np.arange(NS, dtype=np.float32)[:, None, None]
        ys = np.arange(GH, dtype=np.float32)[None, :, None]
        xs = np.arange(GW, dtype=np.float32)[None, None, :]
        idxs, regs, gates = [], [], []
        for cv in self.params[stem]:
            xv = IMG - cv["xv"] if flip else cv["xv"]
            yv, s, w, th = cv["yv"], cv["s"], cv["w"], cv["th"]
            fx = min(max(xv, 0.0), IMG - 1e-3) / HSTRIDE - 0.5
            fy = min(max(yv, 0.0), IMG - 1e-3) / HSTRIDE - 0.5
            fk = (math.log(s) - LOG_S0) / LOG_DS
            g = np.exp(-((xs - fx) ** 2 + (ys - fy) ** 2) / (2 * self.hm_sigma ** 2) - (ks - fk) ** 2 / 2.0)
            np.maximum(hm, g, out=hm)
            xi, yi = int(np.clip(round(fx), 0, GW - 1)), int(np.clip(round(fy), 0, GH - 1))
            ki = int(np.clip(round(fk), 0, NS - 1))
            hm[ki, yi, xi] = 1.0
            peak[ki, yi, xi] = 1.0
            idxs.append((ki, yi, xi))
            regs.append((xv / HSTRIDE - (xi + 0.5), yv / HSTRIDE - (yi + 0.5), math.log(s / SLOPES[ki]),
                         math.log(w / W_REF), math.log(th / TH_REF)))
            gates.append((np.abs(self.dxs) <= w).astype(np.float32))
        meta = {"stem": stem, "boxes": self.gt[stem]["boxes"]}
        return (torch.from_numpy(img)[None], torch.from_numpy(hm), torch.from_numpy(peak),
                np.array(idxs, np.int64).reshape(-1, 3), np.array(regs, np.float32).reshape(-1, 5),
                np.array(gates, np.float32).reshape(-1, len(self.dxs)), meta)


def collate(batch):
    imgs, hms, pks, idxs, regs, gates, metas = zip(*batch)
    bidx = np.concatenate([np.concatenate([np.full((len(ix), 1), b), ix], 1) for b, ix in enumerate(idxs)], 0)
    return (torch.stack(imgs), torch.stack(hms), torch.stack(pks), torch.from_numpy(bidx.astype(np.int64)),
            torch.from_numpy(np.concatenate(regs, 0)), torch.from_numpy(np.concatenate(gates, 0)), list(metas))


class AugWrapper(Dataset):
    """与 train_basic.AugWrapper 相同：每个样本取一次 random.random() 决定翻转（几何翻转）。"""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds.encode(idx, flip=random.random() < 0.5)


# ───────────────────────── 损失与解码 ─────────────────────────
def compute_loss(model, img, hm, peak, bidx, reg, gate_t, lam_gate):
    out, a = model(img)
    logit = out[..., 0]                                                       # (B, S, 28, 28)
    focal = tb.focal_loss_heatmap(logit.float(), hm, peak)
    n = max(len(bidx), 1)
    b, k, i, j = bidx[:, 0], bidx[:, 1], bidx[:, 2], bidx[:, 3]
    p = out[b, k, i, j, 1:].float()
    l_reg = F.l1_loss(p, reg, reduction="sum") / n
    if a.shape[2] > 1 and lam_gate > 0:
        ag = a[b, k, :, i, j].float().clamp(1e-4, 1 - 1e-4)
        l_gate = -(gate_t * ag.log() + (1 - gate_t) * (1 - ag).log()).sum() / (n * ag.shape[1])
    else:
        l_gate = torch.zeros((), device=img.device)
    return focal + l_reg + lam_gate * l_gate, focal.detach(), l_reg.detach(), l_gate.detach()


@torch.no_grad()
def decode(out, thresh=0.001, max_det=50):
    """out: (S, 28, 28, 6) → 框 [[x1,y1,x2,y2,score]] 和曲线 [[x_v,y_v,slope,w,th,score]]。"""
    heat = torch.sigmoid(out[..., 0].float())
    hmax, kidx = heat.max(0)
    pooled = F.max_pool2d(hmax[None, None], 3, stride=1, padding=1)[0, 0]
    keep = (hmax == pooled) & (hmax >= thresh)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    sc = hmax[ys, xs]
    order = torch.argsort(sc, descending=True)[:max_det]
    boxes, curves = [], []
    o = out.float().cpu().numpy()
    for t in order.tolist():
        i, j = int(ys[t]), int(xs[t]); k = int(kidx[i, j]); s_ = float(sc[t])
        ox, oy, ds, lw, lt = (float(v) for v in o[k, i, j, 1:])
        xv = (j + 0.5 + ox) * HSTRIDE; yv = (i + 0.5 + oy) * HSTRIDE
        s = float(SLOPES[k] * math.exp(np.clip(ds, -2, 2)))
        w = W_REF * math.exp(np.clip(lw, -4, 2)); th = TH_REF * math.exp(np.clip(lt, -3, 2))
        bx = curve_box(xv, yv, s, w, th)
        if bx is None:
            continue
        boxes.append([*bx, s_])
        curves.append([xv, yv, s, w, th, s_])
    return boxes, curves


@torch.no_grad()
def export(model, ds, path_boxes, path_curves, bs=8):
    model.eval()
    preds, curves = {}, {}
    for st in range(0, len(ds), bs):
        items = [ds[i] for i in range(st, min(st + bs, len(ds)))]
        img = torch.stack([it[0] for it in items]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out, _ = model(img)
        for it, o in zip(items, out):
            bx, cv = decode(o)
            preds[it[-1]["stem"]] = bx
            curves[it[-1]["stem"]] = cv
    json.dump(preds, open(path_boxes, "w", encoding="utf-8"))
    json.dump(curves, open(path_curves, "w", encoding="utf-8"))
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(SHARED / "datasets" / "ds2_f0_s100_hbatbox"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--layout", default="hough", choices=["hough", "local", "boxgrid"])
    ap.add_argument("--targets", default="band", choices=["band", "boxpseudo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=tb.EPOCHS)
    ap.add_argument("--hm-sigma", type=float, default=2.0, help="顶点热图高斯 sigma（28×28 格子）；slope 方向固定 1 档")
    ap.add_argument("--lam-gate", type=float, default=0.5)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    data_dir = Path(args.data)
    manifest = json.load(open(data_dir / "manifest.json", encoding="utf-8"))
    dataset_name, fold = manifest.get("dataset", data_dir.name), manifest.get("fold", 0)
    run_dir = Path(args.runs) / dataset_name / args.exp / f"f{fold}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run_dir={run_dir}", flush=True)

    tb.set_seed(args.seed)
    dxs = [0] if args.layout == "local" else DX_FULL
    tr = CurveDataset(data_dir, "train", args.targets, args.hm_sigma, dxs)
    va = CurveDataset(data_dir, "val", "band", args.hm_sigma, dxs)
    te = CurveDataset(data_dir, "test", "band", args.hm_sigma, dxs)
    print(f"train {len(tr)} (curves: band {tr.n_band}, pseudo {tr.n_pseudo})  val {len(va)}  test {len(te)}  "
          f"layout={args.layout} targets={args.targets} sigma={args.hm_sigma} lam_gate={args.lam_gate}", flush=True)
    pin = device.type == "cuda"
    order_gen = torch.Generator(); order_gen.manual_seed(args.seed)
    view = tr if args.no_augment else AugWrapper(tr)
    tl = DataLoader(view, batch_size=tb.BATCH, shuffle=True, num_workers=args.workers, pin_memory=pin,
                    persistent_workers=args.workers > 0, worker_init_fn=tb._worker_init, collate_fn=collate,
                    generator=order_gen)
    model = build_model(args.layout, args.seed)
    with torch.no_grad():
        fp = [float(model.down1.conv.block[0].weight.double().sum()), float(model.head2.weight.double().sum())]
    print(f"init fingerprint (backbone, head2): {fp}", flush=True)
    torch.manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=tb.LR)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.perf_counter()
    rows = []
    for ep in range(1, args.epochs + 1):
        model.train(); acc = np.zeros(4)
        for img, hm, pk, bidx, reg, gate_t, _ in tl:
            img, hm, pk, bidx, reg, gate_t = [t.to(device, non_blocking=pin) for t in (img, hm, pk, bidx, reg, gate_t)]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss, lf, lr_, lg = compute_loss(model, img, hm, pk, bidx, reg, gate_t, args.lam_gate)
            opt.zero_grad(); loss.backward(); opt.step()
            acc += [loss.item(), lf.item(), lr_.item(), lg.item()]
        acc /= max(len(tl), 1)
        rows.append({"epoch": ep, "loss": acc[0], "focal": acc[1], "reg": acc[2], "gate": acc[3]})
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  epoch {ep}/{args.epochs}  loss {acc[0]:.4f}  focal {acc[1]:.4f}  reg {acc[2]:.4f}  gate {acc[3]:.4f}", flush=True)
    minutes = (time.perf_counter() - t0) / 60
    torch.save(model.state_dict(), run_dir / "final.pth")
    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    export(model, va, run_dir / "preds_val.json", run_dir / "curves_val.json")
    export(model, te, run_dir / "preds_test.json", run_dir / "curves_test.json")
    info = dict(dataset=dataset_name, exp=args.exp, seed=args.seed, fold=fold, box_source=manifest.get("box_source"),
                layout=args.layout, targets=args.targets,
                model="CurveNet (exp1 backbone + hyperbola aggregation layer)", n_slopes=NS, slopes=SLOPES.tolist(),
                dx=list(dxs), l_off=L_OFF, hm_sigma=args.hm_sigma, lam_gate=args.lam_gate,
                augment=not args.no_augment, n_train=len(tr), n_train_band=tr.n_band, n_train_pseudo=tr.n_pseudo,
                init_fingerprint=fp, epochs=args.epochs, batch=tb.BATCH, lr=tb.LR, optimizer="Adam", amp="bfloat16",
                model_selection="last epoch", params_M=round(n_params / 1e6, 4), train_minutes=round(minutes, 2),
                torch=torch.__version__, data_dir=str(data_dir), args=vars(args))
    json.dump(info, open(run_dir / "run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    m = evaluate_run(run_dir, data_dir)
    t = m["test"]["at_selected"]
    print(f"[{dataset_name}/{args.exp}/f{fold}_seed{args.seed}] val conf {m['val']['selected_conf']:.2f} -> test F1 {t['F1']:.4f} "
          f"(P {t['P']:.3f} R {t['R']:.3f})  AP50 {m['test']['AP50']:.4f} | {n_params / 1e6:.3f} M params, {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
