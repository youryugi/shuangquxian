"""Round 3 诊断：band 在什么情况下提供框给不了的信息（不重训，只用 CPU）。

    python diag_band_value.py --runs runs_cv          # 约 5 分钟（hyperbola fit 约 10 ms / 检测）

1. 错误构成：baseline / band arm 在验证集阈值下的漏检和 FP 各属于哪一类（每个 seed 的平均个数）。
2. 信息冗余：HBAT box 就是 band 的外接框。只给一个 GT 框（厚度取 16 px），能把 slope 反推到多准？
   按"框是否碰到图像边缘"（clipped）分开。同一张图里各目标 slope 的一致性（slope = 2/v，同一介质应接近）。
3. 曲线参数：在 TP 检测上估 slope 和顶点，和 HBAT 标注比：
   - 框的几何（顶点 = 框上边中点下移半个厚度；slope 由框宽高反推）；
   - 借同图其他未截断检测框的 slope 中位数（没有则用数据集中位数 2.99）；
   - 数据集中位数 2.99（不看图）；
   - 在本 run 导出的 band 图（> 0.5）上做 hyperbola fit（../../exp1_basic/fit_hyperbola.py）：band arm 和 boxmask arm 各用自己的图；
   - 在真实 band 上做同样的 fit（上限）。
   每个 seed 算中位误差，报 10 个 seed 的均值和范围。
4. 速度一致性能否重打分：检测框反推的 slope 和同图其他高分检测（或标注）的差，用 AUC 区分 TP / FP。

只用 param_status 为 ok / rough 的 HBAT 目标（有 slope）。输出 <runs>/band_value/summary.md、summary.json。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1] / "exp1_basic"))
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(REPO / "tools" / "annotation"))
import hbat  # noqa: E402
from diag_strata import label_gt_boxes, load_seed, pooled_val_conf  # noqa: E402
from evaluate import box_iou  # noqa: E402
from fit_hyperbola import fit_seed  # noqa: E402

DATASETS = HERE.parents[1] / "shared" / "datasets"
HBAT_JSON = REPO / "data" / "additional_utilities" / "hbat" / "annotations.json"
SIZE = 224
TH = 16.0             # 标注 band 厚度的中位数（exp1_basic fit_hyperbola.DEFAULT_THICK）
S_MED = 2.99          # 数据集 slope 中位数（ok + rough）
LOG15 = math.log(1.15)


def clipped(b):
    return b[0] <= 0.5 or b[1] <= 0.5 or b[2] >= SIZE - 0.5 or b[3] >= SIZE - 0.5


def box_params(p, th=TH):
    """从框反推 (x_v, y_v, slope)：假设两翼对称、没被截断、顶点在框上边中点下 th/2。"""
    x1, y1, x2, y2 = p[:4]
    w, h = x2 - x1, y2 - y1
    yv = y1 + th / 2
    a, d = max(1.0, yv), h - th
    s = a / (w / 2) * math.sqrt((d / a + 1) ** 2 - 1) if d > 0 and w >= 4 else None
    return (x1 + x2) / 2, yv, s


def hbat_objects(ann, stem, item):
    out = []
    for o in ann.get(item.get("band_key") or f"{stem}.jpg", []):
        if not o.get("slope") or o.get("param_status") not in ("ok", "rough"):
            continue
        b = hbat.hyperbola_to_bbox(o, (SIZE, SIZE))
        if b:
            out.append(([b["x1"], b["y1"], b["x1"] + b["width"], b["y1"] + b["height"]], o))
    return out


def obj_for_gt(objs, g):
    for b, o in objs:
        if max(abs(u - v) for u, v in zip(b, g)) < 0.06:
            return o
    return None


def greedy(preds, gts, conf):
    """score >= conf 的预测按分数贪心匹配（IoU 0.5）。返回 (每个 GT 匹配到的预测, [(预测, 是否 TP)])。"""
    kept = sorted([p for p in preds if p[4] >= conf], key=lambda p: -p[4])
    m, lab = [None] * len(gts), []
    for p in kept:
        bi, bj = 0.0, -1
        for j, g in enumerate(gts):
            if m[j] is None:
                iou = box_iou(p, g)
                if iou > bi:
                    bi, bj = iou, j
        if bj >= 0 and bi >= 0.5:
            m[bj] = p
            lab.append((p, 1))
        else:
            lab.append((p, 0))
    return m, lab


def auc(y, x):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if not n1 or not n0:
        return float("nan")
    r = rankdata(x)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ioa(p, b):
    ix = max(0.0, min(p[2], b[2]) - max(p[0], b[0]))
    iy = max(0.0, min(p[3], b[3]) - max(p[1], b[1]))
    return ix * iy / max(1e-9, (p[2] - p[0]) * (p[3] - p[1]))


# ───────────────────────── 1. 错误构成 ─────────────────────────
FN_CATS = ["a. 有 IoU≥0.5 的框，但被相邻 GT 占了", "b. near miss（最佳 IoU 0.3–0.5）",
           "c. 有 IoU≥0.5 的框，但分数低于阈值", "d. 任何分数都没有 IoU≥0.5 的框"]
FP_CATS = ["1. 重复：贴着一个已匹配的 GT（IoU≥0.5）", "2. 在某个 GT 旁边（IoU 0.3–0.5）", "3. 和 GT 部分重叠（IoU 0.1–0.3）",
           "4. 没有 HBAT GT，但在 official box 里", "5. 背景"]


def error_anatomy(runs, exp, seeds, labels):
    fn, fng, fp, ngt = Counter(), defaultdict(Counter), Counter(), Counter()
    for seed in seeds:
        val_folds, pt, gt = load_seed(exp, seed, runs)
        conf = pooled_val_conf(val_folds)
        for stem, item in gt.items():
            gts, preds = item["boxes"], pt.get(stem, [])
            m, lab = greedy(preds, gts, conf)
            kept = [p for p, _ in lab]
            for j, g in enumerate(gts):
                grp = "grouped" if labels[(stem, j)]["grouped"].startswith("grouped") else "alone"
                ngt[grp] += 1
                if m[j] is not None:
                    continue
                bk = max([box_iou(p, g) for p in kept], default=0.0)
                ba = max([box_iou(p, g) for p in preds], default=0.0)
                c = FN_CATS[0] if bk >= 0.5 else FN_CATS[1] if bk >= 0.3 else FN_CATS[2] if ba >= 0.5 else FN_CATS[3]
                fn[c] += 1
                fng[grp][c] += 1
            off = item.get("boxes_official", [])
            for p, tp in lab:
                if tp:
                    continue
                mi = max([box_iou(p, g) for g in gts], default=0.0)
                c = (FP_CATS[0] if mi >= 0.5 else FP_CATS[1] if mi >= 0.3 else FP_CATS[2] if mi >= 0.1
                     else FP_CATS[3] if any(ioa(p, b) > 0.5 for b in off) else FP_CATS[4])
                fp[c] += 1
    k = len(seeds)
    return {"fn": {c: fn[c] / k for c in FN_CATS}, "fn_grouped": {c: fng["grouped"][c] / k for c in FN_CATS},
            "fn_alone": {c: fng["alone"][c] / k for c in FN_CATS}, "fp": {c: fp[c] / k for c in FP_CATS},
            "miss_rate": {g: sum(fng[g].values()) / ngt[g] for g in ("grouped", "alone")}}


# ───────────────────────── 2. 信息冗余 ─────────────────────────
def redundancy(ann):
    rows = defaultdict(list)
    per_img = {}
    for img, objs in ann.items():
        good = []
        for o in objs:
            if not o.get("slope") or o.get("param_status") not in ("ok", "rough"):
                continue
            good.append(o)
            b = hbat.hyperbola_to_bbox(o, (SIZE, SIZE))
            if b is None:
                continue
            box = [b["x1"], b["y1"], b["x1"] + b["width"], b["y1"] + b["height"]]
            _, _, s = box_params(box)
            rows["clipped" if clipped(box) else "unclipped"].append(abs(math.log(s / o["slope"])) if s else float("nan"))
        per_img[img] = good
    out = {"box_to_slope": {k: {"n": len(v), "median_abs_log": float(np.nanmedian(v)), "within15": float(np.nanmean(np.array(v) < LOG15))}
                            for k, v in rows.items()}}
    for status, key in ((("ok",), "ok"), (("ok", "rough"), "ok+rough")):
        multi = {k: [o["slope"] for o in v if o["param_status"] in status] for k, v in per_img.items()}
        multi = {k: v for k, v in multi.items() if len(v) >= 2}
        same = [abs(math.log(a / b)) for v in multi.values() for i, a in enumerate(v) for b in v[i + 1:]]
        pool = [(k, s) for k, v in multi.items() for s in v]
        rng, diff = random.Random(0), []
        while len(diff) < 20000:
            (k1, a), (k2, b) = rng.sample(pool, 2)
            if k1 != k2:
                diff.append(abs(math.log(a / b)))
        ls = np.concatenate([np.log(v) for v in multi.values()])
        wv = np.mean(np.concatenate([(np.log(v) - np.mean(np.log(v))) ** 2 for v in multi.values()]))
        out[f"slope_consistency_{key}"] = {"images": len(multi), "objects": len(ls), "same_image_median_abs_log": float(np.median(same)),
                                           "diff_image_median_abs_log": float(np.median(diff)), "within_image_var_share": float(wv / np.var(ls)),
                                           "same_within15": float(np.mean(np.array(same) < LOG15)), "diff_within15": float(np.mean(np.array(diff) < LOG15))}
    return out


# ───────────────────────── 3. 曲线参数 ─────────────────────────
EST = ["数据集中位数 2.99", "框的几何", "借同图未截断框", "fit：本 arm 的预测图", "fit：真实 band"]


def param_errors(runs, exp, seed, ann):
    val_folds, _, _ = load_seed(exp, seed, runs)
    conf = pooled_val_conf(val_folds)
    err = defaultdict(list)    # (clipped/unclipped, 量, 估计方法) -> [误差]
    for f in range(5):
        gtj = json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"]
        rd = runs / "ds2" / exp / f"f{f}_seed{seed}"
        P = json.load(open(rd / "preds_test.json", encoding="utf-8"))
        for stem, item in gtj.items():
            objs = hbat_objects(ann, stem, item)
            m, lab = greedy(P.get(stem, []), item["boxes"], conf)
            kept = [p for p, _ in lab]
            pmap = cv2.imread(str(rd / "band_test" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 > 0.5
            gmap = cv2.imread(str(DATASETS / f"ds2_f{f}_s100_hbatbox" / "masks_band" / "test" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 127
            for g, p in zip(item["boxes"], m):
                o = obj_for_gt(objs, g) if p is not None else None
                if o is None:
                    continue
                k = "clipped" if clipped(g) else "unclipped"
                xv, yv, sb = box_params(p)
                nb = [box_params(q)[2] for q in kept if q is not p and not clipped(q)]
                nb = [s for s in nb if s]
                fits = {EST[3]: fit_seed(pmap, tuple(p[:4])), EST[4]: fit_seed(gmap, tuple(p[:4]))}
                slopes = {EST[0]: S_MED, EST[1]: sb, EST[2]: float(np.median(nb)) if nb else S_MED,
                          **{n: (ft["slope"] if ft else None) for n, ft in fits.items()}}
                for n, s in slopes.items():
                    err[(k, "slope", n)].append(abs(math.log(s / o["slope"])) if s else float("nan"))
                apex = {EST[1]: (xv, yv), **{n: ((ft["xv"], ft["yv"]) if ft else None) for n, ft in fits.items()}}
                for n, a in apex.items():
                    err[(k, "apex", n)].append(math.hypot(a[0] - o["x_vertex"], a[1] - o["y_vertex"]) if a else float("nan"))
    return {key: (float(np.nanmedian(v)) if key[1] == "apex" else float(np.nanmedian(v)),
                  float(np.nanmean(np.array(v) < (LOG15 if key[1] == "slope" else 5.0))), len(v)) for key, v in err.items()}


# ───────────────────────── 4. 速度一致性重打分 ─────────────────────────
def consistency_auc(runs, seed, ann):
    _, pt, gt = load_seed("y8s_pre_none", seed, runs)
    Y, S, Cd, Cg = [], [], [], []
    for stem, item in gt.items():
        _, lab = greedy(pt.get(stem, []), item["boxes"], 0.05)
        objs = [o for _, o in hbat_objects(ann, stem, item) if o["param_status"] == "ok"]
        s_img = float(np.median([o["slope"] for o in objs])) if objs else S_MED
        for i, (p, y) in enumerate(lab):
            if clipped(p):
                continue
            sp = box_params(p)[2]
            if not sp:
                continue
            nb = [box_params(q)[2] for j, (q, _) in enumerate(lab) if j != i and q[4] >= 0.5 and not clipped(q)]
            nb = [s for s in nb if s]
            Y.append(y); S.append(p[4])
            Cd.append(-abs(math.log(sp / (float(np.median(nb)) if nb else S_MED))))
            Cg.append(-abs(math.log(sp / s_img)))
    Y, S, Cd, Cg = map(np.array, (Y, S, Cd, Cg))
    amb = (S >= 0.2) & (S < 0.6)
    return {"score_amb": auc(Y[amb], S[amb]), "cons_det_amb": auc(Y[amb], Cd[amb]), "cons_ann_amb": auc(Y[amb], Cg[amb]),
            "score_all": auc(Y, S), "cons_det_all": auc(Y, Cd), "cons_ann_all": auc(Y, Cg), "n_amb": int(amb.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()
    runs, seeds = Path(args.runs), args.seeds
    ann = json.load(open(HBAT_JSON, encoding="utf-8"))
    res = {}
    L = [f"# band 能提供什么框给不了的信息（{len(seeds)} seeds，5 折合并测试集）", ""]

    gt_all = {}
    for f in range(5):
        gt_all.update(json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"])
    labels, _ = label_gt_boxes(gt_all)
    L += ["## 1. 错误构成（每个 seed 的平均个数，验证集阈值）", ""]
    for exp in ("y8s_pre_none", "y8s_pre_band"):
        r = error_anatomy(runs, exp, seeds, labels)
        res[f"errors_{exp}"] = r
        nfn, nfp = sum(r["fn"].values()), sum(r["fp"].values())
        L += [f"**{exp}**：漏检 {nfn:.1f} / 222（grouped 漏检率 {r['miss_rate']['grouped']:.2f}，alone {r['miss_rate']['alone']:.2f}），FP {nfp:.1f}", "",
              "| 漏检类型 | 个数 | 占比 | grouped / alone |", "|---|---|---|---|"]
        L += [f"| {c} | {r['fn'][c]:.1f} | {r['fn'][c] / nfn:.0%} | {r['fn_grouped'][c]:.1f} / {r['fn_alone'][c]:.1f} |" for c in FN_CATS]
        L += ["", "| FP 类型 | 个数 | 占比 |", "|---|---|---|"]
        L += [f"| {c} | {r['fp'][c]:.1f} | {r['fp'][c] / nfp:.0%} |" for c in FP_CATS] + [""]

    r = redundancy(ann)
    res["redundancy"] = r
    L += ["## 2. 信息冗余：一个 GT 框能还原多少 band", "",
          "| GT 框 | n | 由框反推 slope 的误差（中位数，×） | ±15% 以内 |", "|---|---|---|---|"]
    for k in ("unclipped", "clipped"):
        v = r["box_to_slope"][k]
        L.append(f"| {k} | {v['n']} | ×{math.exp(v['median_abs_log']):.2f} | {v['within15']:.2f} |")
    L += ["", "| 同一张图里的 slope（= 2/v） | 图 / 目标 | 同图两两差（×） | 不同图两两差（×） | 同图 ±15% 以内 | 不同图 ±15% 以内 | 图内方差占比 |",
          "|---|---|---|---|---|---|---|"]
    for key in ("ok", "ok+rough"):
        v = r[f"slope_consistency_{key}"]
        L.append(f"| param_status {key} | {v['images']} / {v['objects']} | ×{math.exp(v['same_image_median_abs_log']):.2f} | "
                 f"×{math.exp(v['diff_image_median_abs_log']):.2f} | {v['same_within15']:.2f} | {v['diff_within15']:.2f} | {v['within_image_var_share']:.2f} |")

    L += ["", "## 3. 曲线参数：TP 检测上的 slope 和顶点误差（每个 seed 的中位数，10 seeds 均值 [最小, 最大]）", ""]
    for exp in ("y8s_pre_band", "y8s_pre_boxmask"):
        per = defaultdict(list)
        for seed in seeds:
            for key, v in param_errors(runs, exp, seed, ann).items():
                per[key].append(v)
        res[f"params_{exp}"] = {"|".join(k): v for k, v in per.items()}
        L += [f"**检测来自 {exp}**", "", "| 目标 | 量 | 估计方法 | 中位误差 | ±15%（slope）/ 5 px（顶点）以内 | TP / seed |", "|---|---|---|---|---|---|"]
        for k in ("clipped", "unclipped"):
            for q in ("slope", "apex"):
                for n in EST:
                    v = per.get((k, q, n))
                    if not v:
                        continue
                    a = np.array(v)
                    med = (f"×{math.exp(a[:, 0].mean()):.2f} [×{math.exp(a[:, 0].min()):.2f}, ×{math.exp(a[:, 0].max()):.2f}]" if q == "slope"
                           else f"{a[:, 0].mean():.1f} px [{a[:, 0].min():.1f}, {a[:, 0].max():.1f}]")
                    L.append(f"| {k} | {q} | {n} | {med} | {a[:, 1].mean():.2f} | {a[:, 2].mean():.0f} |")
        L.append("")

    aucs = [consistency_auc(runs, s, ann) for s in seeds]
    res["consistency_auc"] = aucs
    g = lambda k: np.array([a[k] for a in aucs])  # noqa: E731
    L += ["## 4. 速度一致性能否区分 TP / FP（baseline 检测，未截断框，conf ≥ 0.05；AUC，0.5 = 没有信息）", "",
          "| 信号 | 全部 | 模糊区（0.2 ≤ score < 0.6） |", "|---|---|---|",
          f"| score | {g('score_all').mean():.3f} | {g('score_amb').mean():.3f} |",
          f"| 框反推的 slope 和同图高分检测一致 | {g('cons_det_all').mean():.3f} | {g('cons_det_amb').mean():.3f} [{g('cons_det_amb').min():.3f}, {g('cons_det_amb').max():.3f}] |",
          f"| 框反推的 slope 和同图标注 slope 一致（上限） | {g('cons_ann_all').mean():.3f} | {g('cons_ann_amb').mean():.3f} [{g('cons_ann_amb').min():.3f}, {g('cons_ann_amb').max():.3f}] |",
          "", f"模糊区每个 seed 约 {g('n_amb').mean():.0f} 个检测。"]

    out = runs / "band_value"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(res, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("\n".join(L))
    print(f"\n-> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
