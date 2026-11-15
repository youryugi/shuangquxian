"""
基线 #2（关键点回归）：顶点 + 左右端点，3 点代数反解 slope。

回答的问题：关键点检测能不能反推出斜率？
--------------------------------------------------------------------------
能——前提是"端点"必须是真的落在曲线上的点，不是随便找的 bbox 角点。
本文件的端点定义为标注 span 的两端在曲线上的取值：
    x_left  = x_vertex - span/2,  y_left  = y_c(x_left)
    x_right = x_vertex + span/2,  y_right = y_c(x_right)
（y_c 就是主脚本 y_center 的公式，见 hyperbola_param_regress.py）

hyperbola_param_regress.py 里已经证明：双曲线中心线满足
    y^2 = A*x^2 + B*x + C，  A=slope^2, x_vertex=-B/(2A), y_vertex=sqrt(C-A*x_vertex^2)
给定 3 个点(顶点+左端点+右端点)，就是 3 个方程解 3 个未知数 (A,B,C)——
可以**精确代数求解**，不需要拟合/优化。baseline_hough_fit.py 里的
_fit_quadratic() 用最小二乘解这个方程组，点数=3 时最小二乘退化成精确解，
直接复用即可（见下面 decode()）。

和 hyperbola_param_regress.py 公平对比的关键点
--------------------------------------------------------------------------
- 网络结构（backbone/grid/stride）、数据集、切分 seed、MATCH_DIST、nms()、
  阈值扫描方法论全部直接复用主脚本的同名对象，只有"预测什么"不同：
      主脚本   ：[obj, dx, dy, log_slope]        直接回归 slope
      本脚本   ：[obj, dxv,dyv, dxl,dyl, dxr,dyr]  回归 3 个点，slope 后验代数反解
- 这正是文献里"keypoint regression"一类方法的做法（先测 3 个关键点，
  再解方程得到双曲线参数），本脚本是它的一个忠实、可公平对比的实现。

用法：
    python baseline_keypoint_regress.py
    python baseline_keypoint_regress.py --backbone resnet50
"""
import os
import json
import math
import random
import argparse
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from hyperbola_param_regress import (
    INPUT_SIZE, STRIDE, GRID, BATCH_SIZE, EPOCHS, LR, SEED, TRAIN_FRAC,
    BACKBONE, PRETRAINED, MATCH_DIST, NMS_DIST, OBJ_THRES,
    THRESH_SWEEP, NMS_SWEEP, FINE_SWEEP, THRESH_FINE_DELTAS, NMS_FINE_DELTAS,
    normalize_obj, collate, conv_bn, ScratchBackbone, ResNetBackbone,
    set_seed, nms, _centerline_points, _draw_vertex,
)
from baseline_hough_fit import _fit_quadratic

_HERE = os.path.dirname(os.path.abspath(__file__))
#无超参数寻找和粗调精调
W_OBJ = 1.0
W_NOOBJ = 0.5
W_POINT = 5.0    # 顶点+两端点 L1 的权重——这里没有 shape_loss 那种耦合问题
                 # （点坐标误差本身就是深度无关的像素误差，不需要额外的曲线采样项）

# 阈值 × NMS 距离的粗调+精调网格，直接复用主脚本 hyperbola_param_regress.py
# 里的同名常量（THRESH_SWEEP/NMS_SWEEP/FINE_SWEEP/*_FINE_RANGE）——两边用
# 同一套搜索方法论、同样的搜索"力度"，调参充分程度才可比，不会一个基线
# 精调过、另一个只扫了粗网格这种不公平。
DO_SWEEP = True
DO_VISUALIZE = True
VIS_MAX_IMAGES = 20
VIS_PRED_SPAN = 300.0


# ══════════════════════════════════════════════════════════════════════════
# 数据集：target 通道 = [obj, dxv,dyv, dxl,dyl, dxr,dyr]
# ══════════════════════════════════════════════════════════════════════════
class KeypointDataset(Dataset):
    """dxv,dyv：顶点在格内的相对偏移(0~1，同主脚本)。
    dxl,dyl,dxr,dyr：左/右端点相对顶点的偏移，除以 INPUT_SIZE 归一化，
    无界直接回归（同主脚本 log_slope 的处理方式）。
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
        img_t = torch.from_numpy(arr).permute(2, 0, 1)

        sx = INPUT_SIZE / w0
        sy = INPUT_SIZE / h0

        target = torch.zeros(7, GRID, GRID, dtype=torch.float32)
        gt_list = []
        for o in objs:
            xin = o["x_vertex"] * sx
            yin = o["y_vertex"] * sy
            span_in = o["span"] * sx
            slope_in = o["slope"] * (sy / sx)
            if not (0 <= xin < INPUT_SIZE and 0 <= yin < INPUT_SIZE):
                continue

            a = max(1.0, yin)
            b = a / max(1e-3, slope_in)
            half = span_in / 2.0
            xl_in, xr_in = xin - half, xin + half
            yl_in = yin + a * (math.sqrt(1.0 + ((xl_in - xin) / b) ** 2) - 1.0)
            yr_in = yin + a * (math.sqrt(1.0 + ((xr_in - xin) / b) ** 2) - 1.0)

            gx = int(xin // STRIDE)
            gy = int(yin // STRIDE)
            gx = min(gx, GRID - 1)
            gy = min(gy, GRID - 1)
            dx = (xin - gx * STRIDE) / STRIDE
            dy = (yin - gy * STRIDE) / STRIDE

            target[0, gy, gx] = 1.0
            target[1, gy, gx] = dx
            target[2, gy, gx] = dy
            target[3, gy, gx] = (xl_in - xin) / INPUT_SIZE
            target[4, gy, gx] = (yl_in - yin) / INPUT_SIZE
            target[5, gy, gx] = (xr_in - xin) / INPUT_SIZE
            target[6, gy, gx] = (yr_in - yin) / INPUT_SIZE
            gt_list.append((o["x_vertex"], o["y_vertex"], o["slope"]))

        return img_t, target, gt_list, (w0, h0)


# ══════════════════════════════════════════════════════════════════════════
# 模型：backbone 复用主脚本，head 输出 7 通道
# ══════════════════════════════════════════════════════════════════════════
class KeypointNet(nn.Module):
    def __init__(self, backbone="scratch", pretrained=True):
        super().__init__()
        if backbone == "scratch":
            self.backbone = ScratchBackbone()
        else:
            self.backbone = ResNetBackbone(name=backbone, pretrained=pretrained)
        feat_ch = self.backbone.out_ch
        self.head = nn.Sequential(
            conv_bn(feat_ch, 256),
            nn.Conv2d(256, 7, 1),   # [obj, dxv,dyv, dxl,dyl, dxr,dyr]
        )
        self.head[-1].bias.data[0] = -4.0

    def forward(self, x):
        return self.head(self.backbone(x))


# ══════════════════════════════════════════════════════════════════════════
# 损失：obj 用同主脚本的 pos/neg 平衡 BCE；3 个点直接 L1（无需 shape_loss——
# 点坐标误差本身就是深度无关的像素误差，没有主脚本里 slope/y_vertex 耦合导致
# 的"同样误差、不同深度视觉影响天差地别"的问题）。
# ══════════════════════════════════════════════════════════════════════════
def compute_loss(pred, target):
    obj_t = target[:, 0]
    pos = obj_t > 0.5
    neg = ~pos

    bce = F.binary_cross_entropy_with_logits(pred[:, 0], obj_t, reduction="none")
    loss_pos = bce[pos].mean() if pos.any() else torch.zeros((), device=bce.device)
    loss_neg = bce[neg].mean() if neg.any() else torch.zeros((), device=bce.device)
    loss_obj = W_OBJ * loss_pos + W_NOOBJ * loss_neg

    if pos.any():
        dxv_p, dyv_p = torch.sigmoid(pred[:, 1])[pos], torch.sigmoid(pred[:, 2])[pos]
        dxl_p, dyl_p = pred[:, 3][pos], pred[:, 4][pos]
        dxr_p, dyr_p = pred[:, 5][pos], pred[:, 6][pos]

        dxv_g, dyv_g = target[:, 1][pos], target[:, 2][pos]
        dxl_g, dyl_g = target[:, 3][pos], target[:, 4][pos]
        dxr_g, dyr_g = target[:, 5][pos], target[:, 6][pos]

        loss_point = (F.l1_loss(dxv_p, dxv_g) + F.l1_loss(dyv_p, dyv_g) +
                      F.l1_loss(dxl_p, dxl_g) + F.l1_loss(dyl_p, dyl_g) +
                      F.l1_loss(dxr_p, dxr_g) + F.l1_loss(dyr_p, dyr_g))
    else:
        loss_point = torch.zeros((), device=pred.device)

    total = loss_obj + W_POINT * loss_point
    return total, {"obj": loss_obj.item(), "point": loss_point.detach().item()}


# ══════════════════════════════════════════════════════════════════════════
# 解码：先取出 3 个点（原图尺度），再用 _fit_quadratic 精确代数解出
# (x_vertex, y_vertex, slope)。3 点各自独立换算回原图尺度后再拟合，
# 各向异性缩放（sx≠sy）天然被正确处理，不需要像主脚本 decode() 那样
# 对 slope 单独做 sx/sy 校正。
# ══════════════════════════════════════════════════════════════════════════
def decode(pred, size, thres=OBJ_THRES):
    w0, h0 = size
    sx = INPUT_SIZE / w0
    sy = INPUT_SIZE / h0
    obj = torch.sigmoid(pred[0])
    dxv, dyv = torch.sigmoid(pred[1]), torch.sigmoid(pred[2])
    dxl, dyl, dxr, dyr = pred[3], pred[4], pred[5], pred[6]

    out = []
    ys, xs = torch.where(obj >= thres)
    for gy, gx in zip(ys.tolist(), xs.tolist()):
        xv_in = (gx + dxv[gy, gx].item()) * STRIDE
        yv_in = (gy + dyv[gy, gx].item()) * STRIDE
        xl_in = xv_in + dxl[gy, gx].item() * INPUT_SIZE
        yl_in = yv_in + dyl[gy, gx].item() * INPUT_SIZE
        xr_in = xv_in + dxr[gy, gx].item() * INPUT_SIZE
        yr_in = yv_in + dyr[gy, gx].item() * INPUT_SIZE

        xv, yv = xv_in / sx, yv_in / sy
        xl, yl = xl_in / sx, yl_in / sy
        xr, yr = xr_in / sx, yr_in / sy

        fit = _fit_quadratic(np.array([xv, xl, xr]), np.array([yv, yl, yr]))
        if fit is None:
            continue
        x_vertex, y_vertex, slope = fit
        out.append((x_vertex, y_vertex, slope, obj[gy, gx].item(), (xv, yv), (xl, yl), (xr, yr)))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 评估（和 hyperbola_param_regress._score_at_threshold 定义完全一致）
# ══════════════════════════════════════════════════════════════════════════
def _score_at_threshold(preds, gts, sizes, thres, nms_dist=NMS_DIST):
    tp = fp = fn = 0
    vx_err, vy_err, slope_err = [], [], []
    for p, gt, size in zip(preds, gts, sizes):
        dets = decode(p, size, thres=thres)
        dets = [(d[0], d[1], d[2], d[3]) for d in dets]  # 丢掉可视化用的原始 3 点
        dets = nms(dets, dist_thres=nms_dist)
        used = [False] * len(gt)
        for x, y, s, _score in dets:
            best, bj = MATCH_DIST, -1
            for j, (gx, gy, gs) in enumerate(gt):
                if used[j]:
                    continue
                d = math.hypot(x - gx, y - gy)
                if d < best:
                    best, bj = d, j
            if bj >= 0:
                used[bj] = True
                tp += 1
                gx, gy, gs = gt[bj]
                vx_err.append(abs(x - gx))
                vy_err.append(abs(y - gy))
                slope_err.append(abs(s - gs))
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


@torch.no_grad()
def _collect_predictions(model, loader, device):
    model.eval()
    preds, gts, sizes = [], [], []
    for imgs, _, g, sz in loader:
        p = model(imgs.to(device)).cpu()
        preds.extend(p)
        gts.extend(g)
        sizes.extend(sz)
    return preds, gts, sizes


def evaluate(model, loader, device, thres=OBJ_THRES, nms_dist=NMS_DIST):
    preds, gts, sizes = _collect_predictions(model, loader, device)
    return _score_at_threshold(preds, gts, sizes, thres, nms_dist=nms_dist)


def sweep_thresholds(model, loader, device, thresholds=THRESH_SWEEP, nms_dists=NMS_SWEEP):
    """粗网格打印 P/R/F1 表；FINE_SWEEP=True 时再在细网格里搜全局最优——
    和 hyperbola_param_regress.sweep_thresholds() 完全同一套方法论/常量，
    保证两个基线"调参力度"一致，对比才公平。"""
    preds, gts, sizes = _collect_predictions(model, loader, device)

    grid = {}
    for th in thresholds:
        for nd in nms_dists:
            grid[(th, nd)] = _score_at_threshold(preds, gts, sizes, th, nms_dist=nd)

    cell_w = 19
    print("    (每格 = P/R/F1)")
    print("    thres\\nms" + "".join(f"{nd:>{cell_w}.0f}" for nd in nms_dists))
    for th in thresholds:
        row = "".join(
            f"{grid[(th, nd)]['P']:.3f}/{grid[(th, nd)]['R']:.3f}/{grid[(th, nd)]['F1']:.3f}".rjust(cell_w)
            for nd in nms_dists
        )
        print(f"    {th:>7.2f}  {row}")

    best_th, best_nd = max(grid, key=lambda k: grid[k]["F1"])
    best_m = grid[(best_th, best_nd)]

    if FINE_SWEEP:
        # 围绕粗网格的最优点展开邻域(而不是写死绝对区间)，保证换了权重/
        # backbone 之后精调范围能跟着真正的最优点走，不会搜偏区域白跑。
        fine_thres = sorted({min(0.999, max(0.001, best_th + d)) for d in THRESH_FINE_DELTAS})
        fine_nms = sorted({max(4.0, best_nd + d) for d in NMS_FINE_DELTAS})
        fbest_th = fbest_nd = fbest_m = None
        for th in fine_thres:
            for nd in fine_nms:
                m = _score_at_threshold(preds, gts, sizes, th, nms_dist=nd)
                if fbest_m is None or m["F1"] > fbest_m["F1"]:
                    fbest_th, fbest_nd, fbest_m = th, nd, m
        print(f"    [精细搜索：围绕粗网格最优 thres={best_th:.2f} nms_dist={best_nd:.0f} 展开 "
              f"{len(fine_thres)}x{len(fine_nms)}={len(fine_thres) * len(fine_nms)} 组合] "
              f"最优 thres={fbest_th:.3f} nms_dist={fbest_nd:.1f}  "
              f"P={fbest_m['P']:.3f} R={fbest_m['R']:.3f} F1={fbest_m['F1']:.3f} "
              f"vtxMAE={fbest_m['vertex_mae']:.1f}px slpMAE={fbest_m['slope_mae']:.3f}")
        if fbest_m["F1"] > best_m["F1"]:
            best_th, best_nd, best_m = fbest_th, fbest_nd, fbest_m

    return best_th, best_nd, best_m


# ══════════════════════════════════════════════════════════════════════════
# 可视化：绿=GT，红=拟合曲线，青色小点=网络直接输出的 3 个原始关键点
# （拟合曲线和原始关键点分开画，能看出"点测得准不准"和"解出来的曲线准不准"
# 是不是一回事——3 点里哪怕只有 1 个测偏了，解出来的 slope 也可能差很远）
# ══════════════════════════════════════════════════════════════════════════
def visualize_predictions(model, subset, device, thres, nms_dist, out_dir, max_images=VIS_MAX_IMAGES):
    base = subset.dataset if isinstance(subset, Subset) else subset
    order = subset.indices if isinstance(subset, Subset) else list(range(len(subset)))
    order = list(order)[:max_images]
    if not order:
        return

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for i in order:
        name, objs = base.items[i]
        orig = Image.open(os.path.join(base.img_dir, name)).convert("RGB")
        w0, h0 = orig.size

        img_in = orig.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(img_in, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_t)[0].cpu()
        dets = decode(pred, (w0, h0), thres=thres)
        dets_simple = [(d[0], d[1], d[2], d[3]) for d in dets]
        kept = nms(dets_simple, dist_thres=nms_dist)
        kept_set = {(round(x, 3), round(y, 3)) for x, y, _s, _sc in kept}

        vis = orig.copy()
        draw = ImageDraw.Draw(vis, "RGBA")

        for x, y, s, _score, pv, pl, pr in dets:
            if (round(x, 3), round(y, 3)) not in kept_set:
                continue
            draw.line(_centerline_points(x, y, s, VIS_PRED_SPAN), fill=(255, 60, 60, 230), width=3)
            _draw_vertex(draw, x, y, (255, 60, 60, 255))
            for (px, py) in (pv, pl, pr):
                r = 4
                draw.ellipse((px - r, py - r, px + r, py + r), fill=(0, 220, 255, 255))

        for o in objs:
            draw.line(_centerline_points(o["x_vertex"], o["y_vertex"], o["slope"], o["span"]),
                      fill=(0, 220, 0, 230), width=3)
            _draw_vertex(draw, o["x_vertex"], o["y_vertex"], (0, 220, 0, 255))

        vis.save(os.path.join(out_dir, name))
    print(f"可视化对比图(绿=GT，红=拟合曲线，青=原始关键点，thres={thres:.3f} nms_dist={nms_dist:.1f})已保存到：{out_dir}")


# ══════════════════════════════════════════════════════════════════════════
# 训练（结构和 hyperbola_param_regress.main() 保持一致，方便对照）
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--img_dir", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--backbone", default=BACKBONE,
                     choices=["scratch", "resnet18", "resnet34", "resnet50", "resnet101"])
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=PRETRAINED)
    args = ap.parse_args()

    from hyperbola_param_regress import IMG_DIR as _IMG_DIR
    img_dir = args.img_dir or _IMG_DIR
    json_path = args.json or os.path.join(img_dir, "annotations.json")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = KeypointDataset(img_dir, json_path)
    print(f"数据集：{len(full)} 张有标注图像  |  设备={device}  |  backbone={args.backbone}")
    if len(full) == 0:
        raise SystemExit("没有找到带标注的图像，检查 --img_dir / --json 路径。")

    # 和 hyperbola_param_regress.main() 完全相同的切分流程，保证 val 集一致。
    idx = list(range(len(full)))
    random.shuffle(idx)
    n_tr = int(len(idx) * args.train_frac)
    tr = Subset(full, idx[:n_tr])
    va = Subset(full, idx[n_tr:])

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = KeypointNet(backbone=args.backbone, pretrained=args.pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {"obj": 0.0, "point": 0.0}
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
            print(f"[ep {ep:3d}] obj={agg['obj']/n:.3f} point={agg['point']/n:.3f} | "
                  f"val P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
                  f"vtxMAE={m['vertex_mae']:.1f}px slpMAE={m['slope_mae']:.3f}")

    tag = datetime.now().strftime("kp_%m%d_%H%M")

    if DO_SWEEP:
        print("阈值 × NMS 距离联合扫描：")
        vis_th, vis_nms, best_m = sweep_thresholds(model, vl, device)
        print(f"  -> 最佳组合 thres={vis_th:.3f} nms_dist={vis_nms:.1f}  "
              f"P={best_m['P']:.3f} R={best_m['R']:.3f} "
              f"F1={best_m['F1']:.3f} vtxMAE={best_m['vertex_mae']:.1f}px slpMAE={best_m['slope_mae']:.3f}")
    else:
        vis_th, vis_nms = OBJ_THRES, NMS_DIST
        m = evaluate(model, vl, device, thres=vis_th, nms_dist=vis_nms)
        print(f"P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
              f"vtxMAE={m['vertex_mae']:.1f}px slpMAE={m['slope_mae']:.3f}")

    if DO_VISUALIZE:
        vis_dir = os.path.join(_HERE, f"vis_{tag}")
        visualize_predictions(model, va, device, vis_th, vis_nms, vis_dir)

    out = os.path.join(_HERE, tag + ".pth")
    torch.save(model.state_dict(), out)
    print("已保存模型：", out)


if __name__ == "__main__":
    main()
