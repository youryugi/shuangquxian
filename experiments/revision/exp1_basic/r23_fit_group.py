"""第 23 轮：fit 和 grouped 的矛盾——三种 Round 22 没试过的去重 / 定框规则（只改推理，用 bandbox_cache 的缓存，不重训）。

实验 4 的诊断（exp4_attention/diag_weak_grouped.py）：band fusion 的 grouped recall 0.482，fit 之后掉到 0.439，丢的几乎全是
"框被配给了相邻目标"。fit 后被 NMS 删掉的 grouped 目标（每 seed 7.1 个）：原框不同但拟合到同一条曲线 3.9；
原框不同、曲线也不同，只是 union 把框撑大后 NMS 删掉 1.4；原框本来就重叠 1.7。Round 22 的"撞曲线保留原框"（α = 1）0.583，没有提升。

规则（t、s_seed、mode、NMS 阈值固定用 hbatbox_flipfix_abs_fitbox 已选中的 (0.7, 0.1, union, 0.5)）：
  A curve-aware NMS  (delta, tau)：两个框 fit 之后 IoU > NMS 阈值时，只有"拟合到同一条曲线（顶点距离 < delta px，224 输入）"
                      或"原框 IoU > tau"才删；曲线不同、原框也不重叠的保留（tau = 1.01 即只看曲线）
  B crowd-gated fit   (tau_c, s_gate)：检测的原框和另一个分数 ≥ s_gate 的检测原框 IoU > tau_c，就不做 fit、保留原框；其余照常 fit + NMS
  C soft-NMS          (sigma)：fit 之后不删，分数按 exp(−IoU² / sigma) 衰减（Gaussian soft-NMS）
  BC gate + soft-NMS  (tau_c, s_gate, sigma)：B 的定框（拥挤的不 fit），去重换成 C 的 soft-NMS
参数在 5 折合并验证集上选（每组参数对 10 个 seed 算验证 F1，置信度阈值也在验证集选，取均值最大的一组）；测试集不参与。
另报 grouped / alone recall（测试集，验证集选的阈值）。

    python r23_fit_group.py --jobs 10      # → runs_cv/r23/summary.md、summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infer_bandbox as ib  # noqa: E402
import diag_strata as ds  # noqa: E402
from evaluate import box_iou, clip_box, prf_at, select_conf  # noqa: E402
from aggregate_cv import paired  # noqa: E402

OUT = HERE / "runs_cv" / "r23"
SRCS = ("hbatbox_flipfix_abs", "hbatbox_flipfix_boxmask")
_GT = {}


def grid():
    g = [("base", None)]
    g += [("A_curve_nms", c) for c in itertools.product((4, 8, 12), (0.3, 0.5, 0.7, 1.01))]
    g += [("B_crowd_gate", c) for c in itertools.product((0.05, 0.1, 0.2, 0.3, 0.5), (0.1, 0.3, 0.5))]
    g += [("C_soft_nms", (s,)) for s in (0.05, 0.1, 0.2, 0.3, 0.5)]
    g += [("BC_gate_soft", c) for c in itertools.product((0.1, 0.2, 0.3), (0.3, 0.5), (0.05, 0.1, 0.2, 0.3, 0.5))]
    return g


def vdist(fa, fb):
    if fa is None or fb is None:
        return math.inf
    return math.hypot(fa["xv"] - fb["xv"], fa["yv"] - fb["yv"])


def soft_nms(ref, sigma):
    boxes = [list(r) for r in ref]
    out, rest = [], list(range(len(boxes)))
    while rest:
        rest.sort(key=lambda i: -boxes[i][4])
        i = rest.pop(0)
        out.append(boxes[i])
        for k in rest:
            io = box_iou(boxes[i], boxes[k])
            boxes[k][4] *= math.exp(-(io * io) / sigma)
    return out


def image(d, fits, scale, base, rule, cfg):
    t, s_seed, mode, nms_thr = base
    fit_of = {i: (fits.get(i, fits.get(str(i))) if d[i][4] >= s_seed else None) for i in range(len(d))}
    if rule in ("B_crowd_gate", "BC_gate_soft"):
        tau_c, s_gate = cfg[:2]
        crowded = {i for i in range(len(d)) for k in range(len(d))
                   if k != i and d[k][4] >= s_gate and box_iou(d[i], d[k]) > tau_c}
        fits_use = {str(i): f for i, f in fit_of.items() if f is not None and i not in crowded}
        ref = ib.apply_fits(d, fits_use, scale, s_seed, mode)
        return ib.nms(ref, nms_thr) if rule == "B_crowd_gate" else soft_nms(ref, cfg[2])
    ref = ib.apply_fits(d, fits, scale, s_seed, mode)
    if rule == "base":
        return ib.nms(ref, nms_thr)
    order = sorted(range(len(ref)), key=lambda i: -ref[i][4])
    if rule == "A_curve_nms":
        delta, tau = cfg
        keep = []
        for i in order:
            if all(not (box_iou(ref[i], ref[k]) > nms_thr and (vdist(fit_of[i], fit_of[k]) < delta or box_iou(d[i], d[k]) > tau))
                   for k in keep):
                keep.append(i)
        return [ref[i] for i in keep]
    if rule == "C_soft_nms":
        return soft_nms(ref, cfg[0])
    raise ValueError(rule)


def gts(phase):
    if phase not in _GT:
        _GT[phase] = {f: json.load(open(ib.data_dir(f) / "gt_boxes.json"))[phase] for f in ib.FOLDS}
    return _GT[phase]


def system(src, seed, phase, rule, cfg):
    base = ib.base_fit_cfg(src)
    out = {}
    for f in ib.FOLDS:
        dets, metas, _ = ib.load_cache(src, f, seed, phase)
        fits = ib.fits_for(src, f, seed, phase, base[0])
        out[f] = {stem: image(d, fits[stem], metas[stem]["scale"], base, rule, cfg) for stem, d in dets.items()}
    return out


def eval_seed(job):
    src, seed, G = job
    gt_all = {}
    for f in ib.FOLDS:
        gt_all.update(gts("test")[f])
    labels = ds.label_gt_boxes(gt_all)[0]
    res = []
    for rule, cfg in G:
        sv, tv, nv = ib.pooled_f1(system(src, seed, "val", rule, cfg), gts("val"))
        conf, f1v = select_conf(sv, tv, nv)
        pt = system(src, seed, "test", rule, cfg)
        st, tt, nt = ib.pooled_f1(pt, gts("test"))
        c = Counter()
        for f in ib.FOLDS:
            for stem, item in gts("test")[f].items():
                W, H = item["width"], item["height"]
                m = ds.match_detail([list(clip_box(p, W, H)) for p in pt[f].get(stem, [])], item["boxes"], conf)[0]
                for j in range(len(item["boxes"])):
                    g = "g" if labels[(stem, j)]["grouped"].startswith("grouped") else "a"
                    c[g + "n"] += 1; c[g + "h"] += m[j] is not None
        res.append({"val": f1v, "test": prf_at(st, tt, nt, conf)["F1"], "conf": conf,
                    "rg": c["gh"] / c["gn"], "ra": c["ah"] / c["an"]})
    return src, seed, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--srcs", nargs="+", default=list(SRCS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    G = grid()
    seeds = list(range(10))
    R = {s: {} for s in args.srcs}
    with Pool(args.jobs) as pool:
        for src, seed, res in pool.imap_unordered(eval_seed, [(s, k, G) for s in args.srcs for k in seeds]):
            R[src][seed] = res
            print(f"  done {src} seed{seed}", flush=True)
    arr = lambda src, j, key: [R[src][k][j][key] for k in seeds]  # noqa: E731
    out = {"grid": [[r, c] for r, c in G], "results": {s: {str(k): v for k, v in R[s].items()} for s in R}}
    L = ["# 第 23 轮：fit 和 grouped 的矛盾（只改推理）", "",
         "每个来源、每类规则在验证集上选一组参数；测试 F1 按 seed 和原 fit（base）配对。grouped / alone = 测试集召回率。", "",
         "| 来源 | 规则 | 选中参数 | val F1 | test F1 | Δ base [95% CI]，胜/平/负 | grouped | alone |", "|---|---|---|---|---|---|---|---|"]
    out["chosen"] = {}
    for src in args.srcs:
        for rule in ("base", "A_curve_nms", "B_crowd_gate", "C_soft_nms", "BC_gate_soft"):
            idx = [j for j, (r, _) in enumerate(G) if r == rule]
            j = max(idx, key=lambda j: np.mean(arr(src, j, "val")))
            p = paired(arr(src, 0, "test"), arr(src, j, "test"))
            out["chosen"][f"{src}|{rule}"] = {"cfg": G[j][1], "pair_vs_base": p,
                                              **{k: float(np.mean(arr(src, j, k))) for k in ("val", "test", "rg", "ra")}}
            L.append(f"| {src.replace('hbatbox_flipfix_', '')} | {rule} | {G[j][1]} | {np.mean(arr(src, j, 'val')):.3f} | "
                     f"{np.mean(arr(src, j, 'test')):.3f} | {p['delta_mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]，"
                     f"{p['wins']}/{p['ties']}/{p['losses']} | {np.mean(arr(src, j, 'rg')):.3f} | {np.mean(arr(src, j, 'ra')):.3f} |")
    # 敏感性：每类规则全部参数（测试列只看敏感性，不用于选择）
    L += ["", "## 全部参数（band 臂；测试列只看敏感性）", "", "| 规则 | 参数 | val F1 | test F1 | grouped | alone |", "|---|---|---|---|---|---|"]
    for j, (r, c) in enumerate(G):
        s = args.srcs[0]
        L.append(f"| {r} | {c} | {np.mean(arr(s, j, 'val')):.3f} | {np.mean(arr(s, j, 'test')):.3f} | "
                 f"{np.mean(arr(s, j, 'rg')):.3f} | {np.mean(arr(s, j, 'ra')):.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(out, open(OUT / "summary.json", "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
