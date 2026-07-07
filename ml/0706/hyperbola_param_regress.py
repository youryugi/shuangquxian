"""
双曲线「参数回归」单文件版（anchor-free 网格头）。

结论先行 —— 到底该回归哪几个参数？
--------------------------------------------------------------------------
标注文件 annotations.json（新格式，见 dataset3/biaozhu-hyperbola-annotator.py）
里每条双曲线有 5 个字段：
    x_vertex, y_vertex, slope, span, thickness
但 GPR 点/柱状目标的双曲线，物理自由度只有 3 个：(x0, t0, v)
    x_vertex  <->  x0   顶点横坐标
    y_vertex  <->  t0   顶点深度（图像顶部=地表=time-zero，所以 a = y_vertex）
    slope     <->  v    渐近线斜率 a/b，编码波速（b = y_vertex / slope）
中心线完全由这 3 个量决定：
    y_c(x) = y_v + a * ( sqrt(1 + ((x - x_v)/b)^2) - 1 ),  a = y_v, b = y_v/slope

而 span、thickness 不是双曲线的几何自由度：
    - span      只是标注时「画多长的两臂 / 截断范围」，曲线本身延伸到无穷；
    - thickness 只是把中心线加粗成带的「带宽」，本数据里基本是常数(≈19~27)。
所以你的直觉是对的：**只回归顶点 (x_vertex, y_vertex) + 斜率 slope 这 3 个数
就足以唯一确定双曲线形状**。span/thickness 若需要可另外回归或直接取常数，
它们对「形状/曲线」没有影响，只影响画图/带区域。

本脚本做什么
--------------------------------------------------------------------------
一张图里可能有多条双曲线（数量不定），所以不是「一张图 -> 3 个数」的纯回归，
而是沿用你 ml/0624 里的 YOLO anchor-free 思路：把图切成网格，每个格子回归
    [objectness, dx, dy, log_slope]
即「这个格子里有没有顶点 + 顶点精确位置 + 斜率」。把 YOLO 的 (x,y,w,h)
换成 (x,y,slope) —— 正是「只回归顶点 + 斜率」。

用法：
    python hyperbola_param_regress.py                # 训练 + 评估
    python hyperbola_param_regress.py --epochs 60
数据默认取 dataset3/images-selected-shuangquxian（新 slope 格式）；
若 JSON 是旧格式(width/height 或 a/b) 也会被 normalize_obj 自动换算成 slope。
"""
import os
import json
import math
import random
import argparse
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ══════════════════════════════════════════════════════════════════════════
# 超参数
# ══════════════════════════════════════════════════════════════════════════
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))         # ml/0706 -> ml -> repo

IMG_DIR    = os.path.join(_REPO_ROOT, "dataset3", "images-selected-shuangquxian")
HYP_JSON   = os.path.join(IMG_DIR, "annotations.json")

INPUT_SIZE = 640          # 正方形输入
STRIDE     = 16           # 网格步长 -> 40x40 网格
GRID       = INPUT_SIZE // STRIDE

BATCH_SIZE = 8
EPOCHS     = 80
LR         = 5e-4
SEED       = 42
TRAIN_FRAC = 0.8

# loss 权重
W_OBJ      = 1.0          # objectness BCE
W_NOOBJ    = 0.5          # 负样本 objectness 权重（负样本远多于正样本）
W_VERTEX   = 5.0          # 顶点 offset L1
W_SLOPE    = 2.0          # log(slope) L1

# 评估：预测顶点与 GT 顶点距离 <= 该像素阈值(原图尺度)算命中
MATCH_DIST = 20.0
OBJ_THRES  = 0.5


# ══════════════════════════════════════════════════════════════════════════
# 标注归一化（兼容旧格式 width/height、a/b -> slope），移植自标注工具
# ══════════════════════════════════════════════════════════════════════════
def normalize_obj(obj):
    obj = dict(obj)
    if "span" not in obj and "width" in obj:
        obj["span"] = obj["width"]
    obj.setdefault("span", 100.0)
    if "slope" not in obj:
        if "a" in obj and obj.get("b"):
            obj["slope"] = round(obj["a"] / obj["b"], 2)
        elif "height" in obj:
            # 旧抛物线/矩形派生：height 是从顶点到臂端的竖直高度
            half_w = max(1.0, obj["span"] / 2.0)
            # y_edge = y_v + a*(sqrt(1+(half_w/b)^2)-1), a=y_v, 反解 slope 较繁；
            # 这里用一阶近似 slope ≈ height / half_w（够作训练初值）。
            obj["slope"] = round(max(0.1, obj["height"] / half_w), 2)
        else:
            obj["slope"] = 1.0
    obj["slope"] = float(max(0.05, obj["slope"]))
    return obj


# ══════════════════════════════════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════════════════════════════════
class HyperbolaParamDataset(Dataset):
    """返回 (image_tensor, target_grid, gt_list)。

    target_grid: [4, GRID, GRID]  通道 = [obj, dx, dy, log_slope]
        obj      : 该格是否为某顶点所在格 (0/1)
        dx,dy    : 顶点在格内的相对偏移 (0~1)
        log_slope: ln(slope)
    gt_list: [(x_px, y_px, slope), ...]  原图像素尺度，供评估用
    """

    def __init__(self, img_dir, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.img_dir = img_dir
        self.items = []
        for name, objs in raw.items():
            path = os.path.join(img_dir, name)
            if not os.path.exists(path):
                continue
            objs = [normalize_obj(o) for o in objs]
            if objs:
                self.items.append((name, objs))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, objs = self.items[i]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        w0, h0 = img.size
        img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1)      # [3,H,W]

        sx = INPUT_SIZE / w0
        sy = INPUT_SIZE / h0

        target = torch.zeros(4, GRID, GRID, dtype=torch.float32)
        gt_list = []
        for o in objs:
            # 原图 -> 输入尺度
            xin = o["x_vertex"] * sx
            yin = o["y_vertex"] * sy
            # slope 是 a/b = 竖直半轴/水平半轴 的比值，是无量纲比例，
            # 但图像各向异性缩放会改变它：a 按 sy 缩、b 按 sx 缩，
            # 故输入尺度下 slope' = slope * (sy/sx)。评估时再换算回去。
            slope_in = o["slope"] * (sy / sx)
            if not (0 <= xin < INPUT_SIZE and 0 <= yin < INPUT_SIZE):
                continue
            gx = int(xin // STRIDE)
            gy = int(yin // STRIDE)
            gx = min(gx, GRID - 1)
            gy = min(gy, GRID - 1)
            dx = (xin - gx * STRIDE) / STRIDE
            dy = (yin - gy * STRIDE) / STRIDE
            target[0, gy, gx] = 1.0
            target[1, gy, gx] = dx
            target[2, gy, gx] = dy
            target[3, gy, gx] = math.log(max(0.05, slope_in))
            gt_list.append((o["x_vertex"], o["y_vertex"], o["slope"]))

        return img_t, target, gt_list, (w0, h0)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    gts = [b[2] for b in batch]
    sizes = [b[3] for b in batch]
    return imgs, targets, gts, sizes


# ══════════════════════════════════════════════════════════════════════════
# 模型：轻量卷积主干 + 参数回归头（无 torchvision 依赖）
# ══════════════════════════════════════════════════════════════════════════
def conv_bn(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ParamRegressNet(nn.Module):
    """输入 [B,3,640,640] -> 输出 [B,4,40,40]（stride=16）。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3, 32), conv_bn(32, 32))
        self.d1 = nn.Sequential(conv_bn(32, 64, stride=2), conv_bn(64, 64))     # /2
        self.d2 = nn.Sequential(conv_bn(64, 128, stride=2), conv_bn(128, 128))  # /4
        self.d3 = nn.Sequential(conv_bn(128, 256, stride=2), conv_bn(256, 256)) # /8
        self.d4 = nn.Sequential(conv_bn(256, 256, stride=2), conv_bn(256, 256)) # /16
        self.head = nn.Sequential(
            conv_bn(256, 256),
            nn.Conv2d(256, 4, 1),   # [obj, dx, dy, log_slope]
        )
        # objectness 偏置初始化为负，训练初期抑制大量负样本的梯度爆炸
        self.head[-1].bias.data[0] = -4.0

    def forward(self, x):
        x = self.stem(x)
        x = self.d1(x); x = self.d2(x); x = self.d3(x); x = self.d4(x)
        return self.head(x)     # [B,4,GRID,GRID]


# ══════════════════════════════════════════════════════════════════════════
# 损失
# ══════════════════════════════════════════════════════════════════════════
def compute_loss(pred, target):
    obj_t = target[:, 0]                      # [B,G,G]
    pos = obj_t > 0.5
    n_pos = pos.sum().clamp(min=1)

    obj_logit = pred[:, 0]
    bce = F.binary_cross_entropy_with_logits(obj_logit, obj_t, reduction="none")
    # 正/负样本分开求均值再加权：否则 ~2 个正样本会被 ~1600 个简单负样本
    # 稀释掉，objectness 永远学不起来（P/R 恒为 0）。
    neg = ~pos
    loss_pos = bce[pos].mean() if pos.any() else torch.zeros((), device=bce.device)
    loss_neg = bce[neg].mean() if neg.any() else torch.zeros((), device=bce.device)
    loss_obj = W_OBJ * loss_pos + W_NOOBJ * loss_neg

    if pos.any():
        dx = torch.sigmoid(pred[:, 1])[pos]
        dy = torch.sigmoid(pred[:, 2])[pos]
        ls = pred[:, 3][pos]
        loss_vertex = (F.l1_loss(dx, target[:, 1][pos]) +
                       F.l1_loss(dy, target[:, 2][pos]))
        loss_slope = F.l1_loss(ls, target[:, 3][pos])
    else:
        loss_vertex = torch.zeros((), device=pred.device)
        loss_slope = torch.zeros((), device=pred.device)

    total = loss_obj + W_VERTEX * loss_vertex + W_SLOPE * loss_slope
    return total, {"obj": loss_obj.item(),
                   "vertex": loss_vertex.detach().item(),
                   "slope": loss_slope.detach().item()}


# ══════════════════════════════════════════════════════════════════════════
# 解码与评估
# ══════════════════════════════════════════════════════════════════════════
def decode(pred, size, thres=OBJ_THRES):
    """单张预测 [4,G,G] -> [(x_px, y_px, slope), ...]（原图尺度）。"""
    w0, h0 = size
    sx = INPUT_SIZE / w0
    sy = INPUT_SIZE / h0
    obj = torch.sigmoid(pred[0])
    dx = torch.sigmoid(pred[1])
    dy = torch.sigmoid(pred[2])
    slope_in = torch.exp(pred[3])
    out = []
    ys, xs = torch.where(obj >= thres)
    for gy, gx in zip(ys.tolist(), xs.tolist()):
        xin = (gx + dx[gy, gx].item()) * STRIDE
        yin = (gy + dy[gy, gx].item()) * STRIDE
        s_in = slope_in[gy, gx].item()
        # 换回原图尺度
        x_px = xin / sx
        y_px = yin / sy
        slope = s_in * (sx / sy)
        out.append((x_px, y_px, slope, obj[gy, gx].item()))
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    vx_err, vy_err, slope_err = [], [], []
    for imgs, _, gts, sizes in loader:
        preds = model(imgs.to(device)).cpu()
        for p, gt, size in zip(preds, gts, sizes):
            dets = decode(p, size)
            # 贪心匹配（按 objectness 从高到低）
            dets = sorted(dets, key=lambda d: -d[3])
            used = [False] * len(gt)
            for dx_, dy_, ds, _ in dets:
                best, bj = MATCH_DIST, -1
                for j, (gx, gy, gs) in enumerate(gt):
                    if used[j]:
                        continue
                    d = math.hypot(dx_ - gx, dy_ - gy)
                    if d < best:
                        best, bj = d, j
                if bj >= 0:
                    used[bj] = True
                    tp += 1
                    gx, gy, gs = gt[bj]
                    vx_err.append(abs(dx_ - gx))
                    vy_err.append(abs(dy_ - gy))
                    slope_err.append(abs(ds - gs))
                else:
                    fp += 1
            fn += used.count(False)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "P": prec, "R": rec, "F1": f1,
        "vertex_mae": float(np.mean(vx_err + vy_err)) if vx_err else float("nan"),
        "slope_mae": float(np.mean(slope_err)) if slope_err else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    json_path = args.json or os.path.join(args.img_dir, "annotations.json")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = HyperbolaParamDataset(args.img_dir, json_path)
    print(f"数据集：{len(full)} 张有标注图像  |  设备={device}")
    if len(full) == 0:
        raise SystemExit("没有找到带标注的图像，检查 --img_dir / --json 路径。")

    idx = list(range(len(full)))
    random.shuffle(idx)
    n_tr = int(len(idx) * TRAIN_FRAC)
    tr = Subset(full, idx[:n_tr])
    va = Subset(full, idx[n_tr:])

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = ParamRegressNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {"obj": 0, "vertex": 0, "slope": 0}
        for imgs, targets, _, _ in tl:
            imgs, targets = imgs.to(device), targets.to(device)
            loss, parts = compute_loss(model(imgs), targets)
            opt.zero_grad(); loss.backward(); opt.step()
            for k in agg:
                agg[k] += parts[k]
        sch.step()
        n = max(1, len(tl))
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            m = evaluate(model, vl, device)
            print(f"[ep {ep:3d}] obj={agg['obj']/n:.3f} vtx={agg['vertex']/n:.3f} "
                  f"slp={agg['slope']/n:.3f} | val P={m['P']:.3f} R={m['R']:.3f} "
                  f"F1={m['F1']:.3f} vtxMAE={m['vertex_mae']:.1f}px slpMAE={m['slope_mae']:.3f}")

    tag = datetime.now().strftime("hyp_param_%m%d_%H%M")
    out = os.path.join(_HERE, tag + ".pth")
    torch.save(model.state_dict(), out)
    print("已保存模型：", out)


if __name__ == "__main__":
    main()
