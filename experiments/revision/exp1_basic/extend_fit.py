"""第 20 轮：形状外推，在 Round 15 的双曲线拟合（fit_hyperbola.py）之后补翼尖。

Round 18 的诊断：预测 band 沿曲线的覆盖率从顶点段 0.56 降到翼尖段 0.27–0.32，拟合出的曲线本身（顶点 + slope）
在翼尖处已经知道，缺的只是"沿曲线走多远"。拟合框只取"曲线还落在 band 里"的那一段，所以翼尖画不全，框就偏窄、偏浅。

两条规则，都只在已拟合的曲线上操作（不重新搜索顶点和 slope）：
  1. 沿曲线续走（track）：拟合用的是高阈值 t 的二值 band（背景干净，找曲线稳）。曲线定下来以后，只在曲线上看预测概率，
     从被支持段的两端向外走：曲线中心线上下 r = thick/4 像素内的最大概率 > t_low 算命中，允许连续 gap 列不命中，
     走到图像边缘或曲线出图为止。曲线外的背景误报不再有影响，所以 t_low 可以比 t 低得多。
  2. 对称补翼（mirror）：HBAT 标注的 band 关于顶点严格对称（span 两侧各一半，见 tools/annotation/hbat.py），
     所以把两翼中较长那一翼的半宽镜像到另一侧，再裁到图内。没有参数。
补完以后按同一条曲线重算框：上边 = 顶点（顶点在范围内）或两端较高者，下边 = 两端较低者，上下各加半个 band 厚度
（与 fit_hyperbola.fit_seed 的取框方式一致；不外推时逐位还原原拟合框）。

本文件只提供规则（extend_fit / extend_image）；参数选择、写 run 目录在 infer_bandbox.py（*_fittrack / *_fitmirror / ...）。
"""
from __future__ import annotations

import numpy as np


def curve_y(xv, yv, slope, xs):
    """与 fit_hyperbola.fit_seed 相同的中心线：a = max(1, |y_v|)，b = a / slope。"""
    a = max(1.0, abs(yv)); b = a / max(slope, 1e-3)
    return yv + a * (np.sqrt(1.0 + ((np.asarray(xs, float) - xv) / b) ** 2) - 1.0)


def _walk(start, step, hit, inside, gap):
    last, miss, x, W = start, 0, start + step, len(hit)
    while 0 <= x < W and inside[x]:
        if hit[x]:
            last, miss = x, 0
        else:
            miss += 1
            if miss > gap:
                break
        x += step
    return last


def track_extent(f, prob, t_low, gap):
    """f: fit_image 的一个检测结果（box = 被支持段的外接框，xl = box[0]，xr = box[2] − 1）；prob: (H, W) 预测概率。返回 (xl, xr)。"""
    H, W = prob.shape
    xl, xr = int(round(f["box"][0])), int(round(f["box"][2])) - 1
    xs = np.arange(W)
    yi = np.rint(curve_y(f["xv"], f["yv"], f["slope"], xs)).astype(np.int64)
    r = max(2, int(round(f["thick"] / 4)))
    rows = yi[None, :] + np.arange(-r, r + 1)[:, None]
    valid = (rows >= 0) & (rows < H)
    vals = np.where(valid, prob[np.clip(rows, 0, H - 1), xs[None, :]], 0.0)
    hit = vals.max(0) > t_low
    inside = yi < H
    return _walk(xl, -1, hit, inside, gap), _walk(xr, +1, hit, inside, gap)


def box_from_extent(f, xl, xr, H, W):
    th = f["thick"]
    yl, yr = curve_y(f["xv"], f["yv"], f["slope"], [xl, xr])
    top = f["yv"] if xl <= f["xv"] <= xr else min(yl, yr)
    return (float(np.clip(xl, 0, W)), float(np.clip(top - th / 2, 0, H)),
            float(np.clip(xr + 1, 0, W)), float(np.clip(max(yl, yr) + th / 2, 0, H)))


def extend_fit(f, prob, t_low=None, gap=3, mirror=False):
    """返回外推后的新拟合结果（只改 box，其余字段不变）。t_low = None：不续走；mirror：对称补翼。"""
    if t_low is None and not mirror:
        return f
    H, W = prob.shape
    xl, xr = int(round(f["box"][0])), int(round(f["box"][2])) - 1
    if t_low is not None:
        xl, xr = track_extent(f, prob, t_low, gap)
    if mirror:
        hw = max(f["xv"] - xl, xr - f["xv"])
        xl, xr = max(0, int(np.floor(f["xv"] - hw))), min(W - 1, int(np.ceil(f["xv"] + hw)))
    g = dict(f)
    g["box"] = box_from_extent(f, xl, xr, H, W)
    return g


def extend_image(fits_img, prob, t_low=None, gap=3, mirror=False):
    """fits_img: {检测序号: fit 或 None}（fit_image / fits_for 的一张图）。"""
    return {k: (None if f is None else extend_fit(f, prob, t_low, gap, mirror)) for k, f in fits_img.items()}
