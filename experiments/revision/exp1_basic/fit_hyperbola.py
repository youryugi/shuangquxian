"""第 15 轮：用双曲线的形状把 band 分给每个检测（代替 Round 12 的"按框中心最近"Voronoi 切分）。

动机：Round 14 的分层结果显示成簇目标（official box 里套着 ≥2 条双曲线，93 / 222 个 GT）是剩下的瓶颈——
band 定框（Round 12）在这一层只有 +0.015，第二阶段框回归 +0.000；即使给**真实的** band，Voronoi 切分也只有 +0.069，
而知道"哪条 band 属于哪个目标"时是 +0.205。相邻双曲线的两翼互相交叉，按距离切必然切错；但每条双曲线自己是一条
参数曲线，沿着曲线走就能穿过交叉点。

规则（每个分数 ≥ s_seed 的检测）：
  1. HBAT 的双曲线模型：y_c(x) = y_v + a·(sqrt(1 + ((x − x_v)/b)²) − 1)，a = y_v（顶点深度），b = a / slope。
     只有顶点 (x_v, y_v) 和 slope 三个自由度。
  2. 在检测框上边中点附近搜索顶点（x：框中心 ± 0.3 框宽；y：框上边 −8 px 到 +0.5 框高），slope 取 13 个值（1–6，几何级数）；
  3. 每条候选曲线从顶点向左右走，只要曲线还落在 band 里就继续（允许 GAP 列的缺口），得到被 band 支持的一段；
     打分 = 沿这一段把"曲线所在像素到 band 上下边缘的竖直距离"（封顶 DV_CAP）加起来——曲线越贴着 band 的中心线、
     走得越远，分越高。只按"落在 band 里的长度"打分时，曲线会贴着 band 的上边缘走（验证集上顶点系统性偏高约 7 px）；
  4. 带框 = 被支持的那一段曲线的外接框，上下各加半个 band 厚度（厚度从 band 上量）。支持长度太短则保留原框。
  5. 新框 = replace / union / blend（同 Round 12）；分数不变。

和 Voronoi 规则的区别：不做像素的排他划分，两条双曲线可以共用交叉处的像素；每个检测只沿自己的曲线取范围。
这一步只有 band 形状的掩码才有意义（填实的框掩码里任何曲线都"被支持"，退化成取掩码的范围），所以对照组也用同一规则跑一遍。

本文件只提供规则本身（fit_seed / fit_refine_image）和并行预计算（prefit）；选参、写 run 目录在 infer_bandbox.py 里。
"""
from __future__ import annotations

import numpy as np

SLOPES = np.geomspace(1.0, 6.0, 13)      # 数据集 2 标注的 slope 范围 1.2–5.0
# 下面的像素常数都是在 224×224 输入上定的。输入换尺寸时（数据集 1 的 --input-size），fit_seed 按 W / REF_SIZE 等比缩放，
# 保持它们占图像的比例不变；W = 224 时缩放系数正好是 1，数据集 2 的结果逐位不变。
REF_SIZE = 224
GAP = 3                                  # 允许曲线上连续 GAP 列没有 band（预测的 band 会断）
MIN_SUPPORT = 12                         # 被支持的列数小于此值 → 保留原框
DV_CAP = 12.0                            # 打分时"到 band 上下边缘的竖直距离"的封顶值（两条 band 交叉处竖直方向很厚，不让它主导）
DEFAULT_THICK = 16.0                     # 量不出厚度时用的默认值（标注厚度中位数 16.3 px）

# slope 网格按数据集设：数据集 1 的 pipe slope 5–95% 分位 0.63–9.6、rebar 中位 7.7（dataset1/data_stats.md），
# 1–6 的网格会撞边界。步长和 ds2 相同（几何级数每格约 ×1.16）。
SLOPE_GRIDS = {"ds2": (1.0, 6.0, 13), "ds1": (0.5, 12.0, 23)}


def configure_slopes(dataset):
    global SLOPES
    lo, hi, n = SLOPE_GRIDS[dataset]
    SLOPES = np.geomspace(lo, hi, n)


def _runs(hit, lead, val):
    """hit, lead: [P, L] bool，val: [P, L] float，沿第 1 维从顶点向外。lead = 曲线还在图像上方（y < 0）的前导列，不算支持也不算中断。
    返回每条曲线连续被支持的列数（不含前导列）、最后一个被支持的列序号（没有则 -1）、这一段上 val 的和。"""
    leadc = np.cumprod(lead, axis=1).astype(bool)
    ok = np.cumprod(hit | leadc, axis=1).astype(bool)
    run = ok & ~leadc
    n = run.sum(1)
    last = np.where(n > 0, ok.sum(1) - 1, -1)
    return n, last, (val * run).sum(1)


def vertical_edge_distance(band):
    """每个 band 像素到它所在竖直连续段上、下边缘的距离的较小值（边缘像素 = 1，非 band = 0）。"""
    H = band.shape[0]
    up = np.zeros(band.shape, np.float32); dn = np.zeros(band.shape, np.float32)
    for y in range(H):
        up[y] = np.where(band[y], (up[y - 1] if y else 0) + 1, 0)
    for y in range(H - 1, -1, -1):
        dn[y] = np.where(band[y], (dn[y + 1] if y < H - 1 else 0) + 1, 0)
    return np.minimum(up, dn)


def fit_seed(band, box):
    """band: (H, W) bool（输入分辨率）；box: (x1, y1, x2, y2) 同一坐标系。
    返回 dict(xv, yv, slope, xl, xr, y_top, y_bot, thick, support) 或 None（找不到被支持的曲线）。"""
    H, W = band.shape
    q = W / REF_SIZE                                      # 像素常数的缩放系数（224 输入 = 1）
    gap = int(round(GAP * q))
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 8.0 * q), max(y2 - y1, 8.0 * q)
    # band 沿水平方向膨胀 GAP 列：等价于允许曲线上有 GAP 列的缺口
    bg = band.copy()
    for s in range(1, gap + 1):
        bg[:, s:] |= band[:, :-s]; bg[:, :-s] |= band[:, s:]
    dv = np.minimum(vertical_edge_distance(band), DV_CAP * q)
    step = max(2.0 * q, w / 24.0)
    rx = max(0.5 * w, 16.0 * q)
    xv_grid = (x1 + x2) / 2 + np.arange(-rx, rx + 1e-6, step)
    yv_grid = np.arange(y1 - 8.0 * q, y1 + max(0.5 * h, 16.0 * q) + 1e-6, max(2.0 * q, h / 24.0))
    YV, SL = np.meshgrid(yv_grid, SLOPES, indexing="ij")
    YV, SL = YV.ravel(), SL.ravel()                       # [P]
    A = np.maximum(1.0, np.abs(YV)); B = A / SL
    cols = np.arange(W, dtype=np.float64)
    best = None
    for xv in xv_grid:
        xa = int(np.clip(round(xv), 0, W - 1))
        yc = YV[:, None] + A[:, None] * (np.sqrt(1.0 + ((cols[None] - xv) / B[:, None]) ** 2) - 1.0)   # [P, W]
        yi = np.rint(yc).astype(np.int64)
        inside = (yi >= 0) & (yi < H)
        hit = np.zeros_like(inside)
        val = np.zeros(inside.shape, np.float32)
        pp, xx = np.nonzero(inside)
        hit[pp, xx] = bg[yi[pp, xx], xx]
        val[pp, xx] = dv[yi[pp, xx], xx]
        above = yi < 0
        nr, lr, sr = _runs(hit[:, xa:], above[:, xa:], val[:, xa:])
        nl, ll, sl = _runs(hit[:, xa::-1], above[:, xa::-1], val[:, xa::-1])
        xr = np.where(lr >= 0, xa + lr, xa); xl = np.where(ll >= 0, xa - ll, xa)
        cols_n = nr + nl - ((nr > 0) & (nl > 0))          # 顶点列左右各算一次，减 1
        score = sr + sl - np.where((nr > 0) & (nl > 0), val[:, xa], 0.0)
        k = int(np.argmax(score))
        if cols_n[k] > 0 and (best is None or score[k] > best["support"]):
            best = {"xv": float(xv), "yv": float(YV[k]), "slope": float(SL[k]), "xl": int(xl[k]), "xr": int(xr[k]),
                    "support": float(score[k]), "n_cols": int(cols_n[k]), "_yc": yc[k].copy()}
    if best is None or best["n_cols"] < MIN_SUPPORT * q:
        return None
    yc = best.pop("_yc")
    # 允许缺口用的是水平膨胀后的 band，所以这一段的两端会多出最多 GAP 列：按原始 band 把两端收回来
    xs = np.arange(best["xl"], best["xr"] + 1)
    on = band[np.clip(np.rint(yc[xs]).astype(int), 0, H - 1), xs] & (np.rint(yc[xs]) >= 0) & (np.rint(yc[xs]) < H)
    if not on.any():
        return None
    best["xl"], best["xr"] = int(xs[np.argmax(on)]), int(xs[len(on) - 1 - np.argmax(on[::-1])])
    xs = np.arange(best["xl"], best["xr"] + 1)
    ys = np.clip(np.rint(yc[xs]).astype(int), 0, H - 1)
    # band 厚度：沿被支持的那一段，量曲线所在位置上下连续的 band 像素数，取中位数
    thick = []
    for x, y in zip(xs[:: max(1, len(xs) // 15)], ys[:: max(1, len(xs) // 15)]):
        if not band[y, x]:
            continue
        col = band[:, x]
        up = y
        while up > 0 and col[up - 1]:
            up -= 1
        dn = y
        while dn < H - 1 and col[dn + 1]:
            dn += 1
        thick.append(dn - up + 1)
    th = float(np.median(thick)) if thick else DEFAULT_THICK * q
    th = float(np.clip(th, 4.0 * q, 48.0 * q))
    y_top = float(min(yc[xs].min(), best["yv"] if best["xl"] <= best["xv"] <= best["xr"] else np.inf)) - th / 2
    y_bot = float(yc[xs].max()) + th / 2
    best.update(thick=th, y_top=y_top, y_bot=y_bot)
    return best


def fit_box(fit, H, W):
    return (float(np.clip(fit["xl"], 0, W)), float(np.clip(fit["y_top"], 0, H)),
            float(np.clip(fit["xr"] + 1, 0, W)), float(np.clip(fit["y_bot"], 0, H)))


def fit_image(dets, band_bin, scale, s_min):
    """对一张图里分数 ≥ s_min 的检测各拟合一次。返回 {检测序号: 带框 (x1,y1,x2,y2，输入分辨率) 或 None}。"""
    H, W = band_bin.shape
    sx, sy = scale
    out = {}
    if band_bin.sum() == 0:
        return out
    for i, d in enumerate(dets):
        if d[4] < s_min:
            continue
        f = fit_seed(band_bin, (d[0] * sx, d[1] * sy, d[2] * sx, d[3] * sy))
        out[i] = None if f is None else {"box": fit_box(f, H, W), "xv": f["xv"], "yv": f["yv"], "slope": f["slope"], "thick": f["thick"], "support": f["support"]}
    return out


def apply_fits(dets, fits, scale, s_seed, mode):
    """fits: fit_image 的输出（键可以是 int 或 str）。与 refine_with_band.refine_image 相同的三种 mode。"""
    sx, sy = scale
    out = [list(d) for d in dets]
    for i, d in enumerate(dets):
        f = fits.get(i, fits.get(str(i)))
        if d[4] < s_seed or f is None:
            continue
        bx1, by1, bx2, by2 = f["box"]
        if bx2 - bx1 < 2 or by2 - by1 < 2:
            continue
        x1, y1, x2, y2 = d[0] * sx, d[1] * sy, d[2] * sx, d[3] * sy
        if mode == "replace":
            nb = (bx1, by1, bx2, by2)
        elif mode == "union":
            nb = (min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2))
        else:
            nb = ((x1 + bx1) / 2, (y1 + by1) / 2, (x2 + bx2) / 2, (y2 + by2) / 2)
        out[i] = [nb[0] / sx, nb[1] / sy, nb[2] / sx, nb[3] / sy, float(d[4])]
    return out
