"""
基线 #1：边缘检测 + 最小二乘拟合（经典/非深度学习路线）。

和 hyperbola_param_regress.py 公平对比的关键点
--------------------------------------------------------------------------
1. 推断的是**完全同一组物理参数** (x_vertex, y_vertex, slope)，不是靠 bbox
   宽高这种混杂了多个量的代理——数学上可以证明两者等价，见下面推导。
2. 用**同一份** annotations.json / 同一个 img_dir，直接 import 主脚本里的
   HyperbolaParamDataset，保证 GT 一模一样。
3. 用**同一个 SEED/TRAIN_FRAC** 复现出完全相同的 train/val 切分，只在
   held-out 的验证集上评估——和主脚本训练时看到的图像互斥。
4. 用**同一个** MATCH_DIST 匹配标准、同一个 nms() 函数（直接 import），
   P/R/F1/vtxMAE/slpMAE 的定义和主脚本 _score_at_threshold 完全一致。

物理模型等价性推导
--------------------------------------------------------------------------
主脚本的中心线模型（a=y_vertex, b=y_vertex/slope）：
    y_c(x) = y_v + a * ( sqrt(1 + ((x-x_v)/b)^2) - 1 )
化简（a=y_v 时 y_v + a*sqrt(...) - a = a*sqrt(...) = y_v*sqrt(...)）：
    y_c(x) = y_v * sqrt(1 + ((x-x_v)/b)^2)
两边平方，代入 b = y_v/slope（故 y_v^2/b^2 = slope^2）：
    y_c(x)^2 = y_v^2 + slope^2 * (x-x_v)^2
            = slope^2 * x^2  -  2*slope^2*x_v * x  +  (slope^2*x_v^2 + y_v^2)
            =    A    * x^2  +        B       * x  +          C
这是 y^2 关于 x 的**二次多项式**——经典最小二乘（Dou et al. 2016 一路的
"t^2 = t0^2 + (4/v^2)(x-x0)^2" 做法）可以直接用 np.polyfit 线性拟合，
再代数反解出：
    slope    = sqrt(A)
    x_vertex = -B / (2A)
    y_vertex = sqrt(C - A * x_vertex^2)
不需要非线性优化、不需要 scipy，纯 numpy 闭式解。

检测流程（避免"整张图只拟合一条曲线"，要能处理一图多条 / 数量不定）
--------------------------------------------------------------------------
1. 灰度化 + Canny 边缘检测，得到边缘像素图。
2. 形态学膨胀，把同一条双曲线带的上下两条边缘桥接成一个连通域
   （膨胀太狠会把相邻双曲线粘连，太弱边缘容易断裂——CANNY_*/DILATE_ITER
   都在下面的超参数区，可调）。
3. 连通域分析（cv2.connectedComponentsWithStats），每个连通域是一个
   "候选双曲线"；按最小宽度/最小点数过滤掉噪声碎片。
4. 对每个候选连通域的边缘点做上述最小二乘拟合，解出 (x,y,slope)，
   用连通域点数当"置信度"（点越多、边缘越清晰/越长，越可信）。
5. 和主脚本一样，对置信度阈值做扫描（不是拍脑袋定一个值），
   NMS 去重复检测，再和 GT 匹配算 P/R/F1。

用法：
    python baseline_hough_fit.py
    python baseline_hough_fit.py --img_dir ... --json ...
需要 opencv-python（pip install opencv-python），懒加载，不装的话只有用到
检测函数时才会报错，不影响其它部分。
"""
import os
import math
import random
import argparse

import numpy as np
from PIL import Image

from hyperbola_param_regress import (
    HyperbolaParamDataset, IMG_DIR, HYP_JSON, SEED, TRAIN_FRAC,
    MATCH_DIST, NMS_DIST, nms, _centerline_points, _draw_vertex,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════
# 超参数（检测专用；数据/切分/匹配标准全部复用主脚本的常量，见上面 import）
# ══════════════════════════════════════════════════════════════════════════
CANNY_LO = 50            # Canny 边缘检测低阈值
CANNY_HI = 150           # Canny 边缘检测高阈值
DILATE_ITER = 2          # 膨胀迭代次数：桥接同一条带的上下边缘
MIN_COMPONENT_WIDTH = 40     # 连通域最小水平跨度（像素），太窄的多半是噪声
MIN_COMPONENT_POINTS = 30    # 连通域最少边缘点数
MAX_COMPONENT_WIDTH_FRAC = 0.95  # 连通域最大宽度不超过图宽的这个比例，避免整图噪声当成一条线

CONF_SWEEP = [10, 20, 30, 50, 80, 120, 200, 300]  # 置信度(连通域点数)阈值扫描

# 检测前端(Canny/膨胀/连通域过滤)超参数的粗调→精调网格。这几个参数直接决定
# "能不能提取出候选双曲线"，如果这里没调好，后面置信度阈值怎么扫都没用
# （表现为 P/R/F1 在整个 CONF_SWEEP range 上完全不变——候选集合本身没变过）。
DO_TUNE = True
COARSE_CANNY_HI    = [80, 150, 220]      # canny_lo 固定用 canny_hi/2（Canny 推荐 1:2~1:3 比例）
COARSE_DILATE_ITER = [1, 2, 3, 4]
COARSE_MIN_POINTS  = [10, 30, 60, 100]
COARSE_MIN_WIDTH   = [20, 40, 60]
FINE_HI_DELTAS   = [-30, -15, 0, 15, 30]     # 精调阶段围绕粗调最优 canny_hi 展开
FINE_POINTS_DELTAS = [-15, -8, 0, 8, 15]     # 精调阶段围绕粗调最优 min_points 展开

DO_VISUALIZE = True
VIS_MAX_IMAGES = 20


def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise SystemExit(
            "边缘检测基线需要 opencv-python（当前环境未安装）。\n"
            "  pip install opencv-python"
        ) from e


# ══════════════════════════════════════════════════════════════════════════
# 核心：边缘检测 + 连通域 + 最小二乘拟合
# ══════════════════════════════════════════════════════════════════════════
def _fit_quadratic(xs, ys):
    """对一组边缘点做 y^2 = A x^2 + B x + C 最小二乘拟合，
    反解 (x_vertex, y_vertex, slope)。拟合失败（退化/开口方向不对）返回 None。
    """
    xs = xs.astype(np.float64)
    ys2 = ys.astype(np.float64) ** 2
    # [x^2, x, 1] @ [A,B,C]^T = y^2，最小二乘用 lstsq 解
    M = np.stack([xs ** 2, xs, np.ones_like(xs)], axis=1)
    coef, *_ = np.linalg.lstsq(M, ys2, rcond=None)
    A, B, C = coef
    if A <= 1e-6:
        return None  # A<=0：不是"开口向下/顶点在上"的双曲线，判定拟合无效
    x_vertex = -B / (2.0 * A)
    y_vertex_sq = C - A * x_vertex ** 2
    if y_vertex_sq <= 0:
        return None
    y_vertex = math.sqrt(y_vertex_sq)
    slope = math.sqrt(A)
    return x_vertex, y_vertex, slope


def detect_hyperbolas(gray, canny_lo=CANNY_LO, canny_hi=CANNY_HI, dilate_iter=DILATE_ITER,
                       min_width=MIN_COMPONENT_WIDTH, min_points=MIN_COMPONENT_POINTS,
                       max_width_frac=MAX_COMPONENT_WIDTH_FRAC, stats_out=None):
    """输入灰度图 (H,W) uint8 ndarray，返回 [(x,y,slope,score), ...]，
    score 是连通域边缘点数（越大越可信），原图像素尺度。

    检测超参数都做成参数而不是读全局常量，方便 tune_detection_hparams()
    在不改全局状态的前提下逐组尝试。stats_out（可选，传一个 dict 进来）
    用于调参时收集诊断信息：候选连通域一路被各级过滤器筛掉了多少个。
    """
    cv2 = _import_cv2()
    h, w = gray.shape
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    kernel = np.ones((3, 3), np.uint8)
    edges_d = cv2.dilate(edges, kernel, iterations=dilate_iter)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(edges_d, connectivity=8)

    dets = []
    n_total = n_labels - 1
    n_size_ok = n_fit_ok = n_final = 0
    for lbl in range(1, n_labels):  # 0 是背景
        x0, y0, cw, ch, area = stats[lbl]
        if cw < min_width or cw > w * max_width_frac:
            continue
        ys, xs = np.where(labels == lbl)
        if len(xs) < min_points:
            continue
        n_size_ok += 1
        fit = _fit_quadratic(xs, ys)
        if fit is None:
            continue
        n_fit_ok += 1
        x_vertex, y_vertex, slope = fit
        # 顶点应该落在图像范围内（外推太离谱的拟合大概率是噪声/误拟合）
        if not (-w * 0.2 <= x_vertex <= w * 1.2 and 0 <= y_vertex <= h):
            continue
        n_final += 1
        dets.append((x_vertex, y_vertex, slope, float(len(xs))))

    if stats_out is not None:
        stats_out["n_total"] = stats_out.get("n_total", 0) + n_total
        stats_out["n_size_ok"] = stats_out.get("n_size_ok", 0) + n_size_ok
        stats_out["n_fit_ok"] = stats_out.get("n_fit_ok", 0) + n_fit_ok
        stats_out["n_final"] = stats_out.get("n_final", 0) + n_final
    return dets


# ══════════════════════════════════════════════════════════════════════════
# 评分：和 hyperbola_param_regress._score_at_threshold 逻辑完全一致
# ══════════════════════════════════════════════════════════════════════════
def _score_at_conf(all_dets, all_gts, min_score, nms_dist=NMS_DIST):
    tp = fp = fn = 0
    vx_err, vy_err, slope_err = [], [], []
    for dets, gt in zip(all_dets, all_gts):
        cand = [d for d in dets if d[3] >= min_score]
        cand = nms(cand, dist_thres=nms_dist)
        used = [False] * len(gt)
        for x, y, s, _score in cand:
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


def sweep_confidence(all_dets, all_gts, conf_sweep=CONF_SWEEP):
    """扫描置信度阈值，打印 P/R/F1 表，返回最优阈值和对应指标。"""
    print("    (置信度 = 连通域边缘点数；同一套 MATCH_DIST/nms 标准，和主模型可比)")
    print("    conf       P       R      F1   vtxMAE   slpMAE")
    best_c, best_m = None, None
    for c in conf_sweep:
        m = _score_at_conf(all_dets, all_gts, c)
        print(f"    {c:>4d}  {m['P']:6.3f}  {m['R']:6.3f}  {m['F1']:6.3f}  "
              f"{m['vertex_mae']:6.1f}  {m['slope_mae']:6.3f}")
        if best_m is None or m["F1"] > best_m["F1"]:
            best_c, best_m = c, m
    return best_c, best_m


# ══════════════════════════════════════════════════════════════════════════
# 检测前端超参数调优：先粗网格找大致区域，再在最优点附近精调
#
# 置信度阈值只是"过滤已检测到的候选"，不能创造新的候选——如果 Canny/膨胀/
# 连通域过滤这一步本身没调好（候选集合本身就是错的/太少的），置信度阈值
# 扫多少个值结果都不会变（表现正是上一轮跑出来的：P/R/F1 在整个 CONF_SWEEP
# 上完全一样）。所以要调的是检测前端参数，不是光扫置信度。
# ══════════════════════════════════════════════════════════════════════════
def _detect_all(base, val_indices, stats_out=None, **det_kwargs):
    all_dets, all_gts = [], []
    for i in val_indices:
        name, objs = base.items[i]
        gray = np.array(Image.open(os.path.join(base.img_dir, name)).convert("L"))
        dets = detect_hyperbolas(gray, stats_out=stats_out, **det_kwargs)
        gt = [(o["x_vertex"], o["y_vertex"], o["slope"]) for o in objs]
        all_dets.append(dets)
        all_gts.append(gt)
    return all_dets, all_gts


def _best_for_hparams(base, val_indices, conf_sweep=CONF_SWEEP, **det_kwargs):
    """给定一组检测前端超参数：跑一遍检测 + 置信度扫描，返回这组超参数能
    达到的最优 (conf, metrics, all_dets, all_gts)。"""
    all_dets, all_gts = _detect_all(base, val_indices, **det_kwargs)
    best_c, best_m = None, {"F1": -1.0, "P": 0.0, "R": 0.0, "vertex_mae": float("nan"), "slope_mae": float("nan")}
    for c in conf_sweep:
        m = _score_at_conf(all_dets, all_gts, c)
        if m["F1"] > best_m["F1"]:
            best_c, best_m = c, m
    return best_c, best_m, all_dets, all_gts


def tune_detection_hparams(base, val_indices):
    """粗调：CANNY_HI x DILATE_ITER x MIN_POINTS x MIN_WIDTH 网格搜索；
    精调：围绕粗调最优点，在 canny_hi / min_points 上做更细的邻域搜索
    （dilate_iter/min_width 已是小范围离散整数，粗网格基本覆盖，不用再细分）。
    """
    n_coarse = len(COARSE_CANNY_HI) * len(COARSE_DILATE_ITER) * len(COARSE_MIN_POINTS) * len(COARSE_MIN_WIDTH)
    print(f"[粗调] {n_coarse} 组检测前端超参数，逐组跑检测+置信度扫描……")
    best = None  # (f1, params, conf, metrics)
    for hi in COARSE_CANNY_HI:
        for dil in COARSE_DILATE_ITER:
            for mp in COARSE_MIN_POINTS:
                for mw in COARSE_MIN_WIDTH:
                    params = dict(canny_lo=hi / 2.0, canny_hi=hi, dilate_iter=dil,
                                  min_points=mp, min_width=mw)
                    c, m, _, _ = _best_for_hparams(base, val_indices, **params)
                    if best is None or m["F1"] > best[0]:
                        best = (m["F1"], params, c, m)
    _f1, params, conf, m = best
    print(f"[粗调] 最优：{params}")
    print(f"       conf={conf}  P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
          f"vtxMAE={m['vertex_mae']:.1f}px slpMAE={m['slope_mae']:.3f}")

    # 精调：围绕粗调最优的 canny_hi / min_points 展开更细的邻域网格
    fine_hi = sorted({max(10.0, params["canny_hi"] + d) for d in FINE_HI_DELTAS})
    fine_mp = sorted({max(5, params["min_points"] + d) for d in FINE_POINTS_DELTAS})
    print(f"[精调] 围绕 canny_hi={params['canny_hi']}, min_points={params['min_points']} "
          f"展开 {len(fine_hi)}x{len(fine_mp)} 邻域网格……")
    for hi in fine_hi:
        for mp in fine_mp:
            p2 = dict(params, canny_lo=hi / 2.0, canny_hi=hi, min_points=mp)
            c, m, _, _ = _best_for_hparams(base, val_indices, **p2)
            if m["F1"] > best[0]:
                best = (m["F1"], p2, c, m)
    _f1, params, conf, m = best
    print(f"[精调] 最优：{params}")
    print(f"       conf={conf}  P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
          f"vtxMAE={m['vertex_mae']:.1f}px slpMAE={m['slope_mae']:.3f}")

    # 诊断：最优参数下，候选连通域一路被筛掉了多少——帮助判断是"检测不到候选"
    # 还是"候选拟合出来的参数不准"。
    diag2 = {}
    _detect_all(base, val_indices, stats_out=diag2, **params)
    n_imgs = len(val_indices)
    print(f"[诊断] 平均每图连通域候选数：{diag2.get('n_total', 0) / n_imgs:.1f} -> "
          f"过滤后 {diag2.get('n_size_ok', 0) / n_imgs:.1f}(尺寸合格) -> "
          f"{diag2.get('n_fit_ok', 0) / n_imgs:.1f}(拟合成功) -> "
          f"{diag2.get('n_final', 0) / n_imgs:.1f}(顶点在图像范围内)")

    return params, conf, m


# ══════════════════════════════════════════════════════════════════════════
# 可视化：复用主脚本的画图函数，风格和主模型的可视化保持一致（绿=GT，红=预测）
# ══════════════════════════════════════════════════════════════════════════
def visualize(base, val_indices, all_dets, best_conf, out_dir, max_images=VIS_MAX_IMAGES):
    from PIL import ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    for i, dets in zip(val_indices[:max_images], all_dets[:max_images]):
        name, objs = base.items[i]
        orig = Image.open(os.path.join(base.img_dir, name)).convert("RGB")
        vis = orig.copy()
        draw = ImageDraw.Draw(vis, "RGBA")

        cand = [d for d in dets if d[3] >= best_conf]
        cand = nms(cand, dist_thres=NMS_DIST)
        for x, y, s, _score in cand:
            draw.line(_centerline_points(x, y, s, 300.0), fill=(255, 60, 60, 230), width=3)
            _draw_vertex(draw, x, y, (255, 60, 60, 255))
        for o in objs:
            draw.line(_centerline_points(o["x_vertex"], o["y_vertex"], o["slope"], o["span"]),
                      fill=(0, 220, 0, 230), width=3)
            _draw_vertex(draw, o["x_vertex"], o["y_vertex"], (0, 220, 0, 255))

        vis.save(os.path.join(out_dir, name))
    print(f"可视化对比图(绿=GT，红=预测，conf>={best_conf})已保存到：{out_dir}")


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=IMG_DIR)
    ap.add_argument("--json", default=None)
    ap.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    args = ap.parse_args()

    json_path = args.json or os.path.join(args.img_dir, "annotations.json")
    base = HyperbolaParamDataset(args.img_dir, json_path)
    print(f"数据集：{len(base)} 张有标注图像")
    if len(base) == 0:
        raise SystemExit("没有找到带标注的图像，检查 --img_dir / --json 路径。")

    # 和 hyperbola_param_regress.main() 完全相同的切分流程/seed，
    # 保证这里的 held-out 验证集和 CNN 训练时看到的验证集逐张对应一致。
    random.seed(SEED)
    idx = list(range(len(base)))
    random.shuffle(idx)
    n_tr = int(len(idx) * args.train_frac)
    val_indices = idx[n_tr:]
    print(f"train={n_tr}  val={len(val_indices)}（切分与 hyperbola_param_regress.py 一致）")

    if DO_TUNE:
        params, best_c, best_m = tune_detection_hparams(base, val_indices)
        print(f"最终采用的检测前端超参数：{params}")
    else:
        params = dict(canny_lo=CANNY_LO, canny_hi=CANNY_HI, dilate_iter=DILATE_ITER,
                      min_width=MIN_COMPONENT_WIDTH, min_points=MIN_COMPONENT_POINTS)
        all_dets, all_gts = _detect_all(base, val_indices, **params)
        best_c, best_m = sweep_confidence(all_dets, all_gts)
        print(f"  -> 最佳置信度阈值 conf={best_c}  P={best_m['P']:.3f} R={best_m['R']:.3f} "
              f"F1={best_m['F1']:.3f} vtxMAE={best_m['vertex_mae']:.1f}px slpMAE={best_m['slope_mae']:.3f}")

    if DO_VISUALIZE:
        all_dets, _all_gts = _detect_all(base, val_indices, **params)
        vis_dir = os.path.join(_HERE, "vis_baseline_hough_fit")
        visualize(base, val_indices, all_dets, best_c, vis_dir)


if __name__ == "__main__":
    main()
