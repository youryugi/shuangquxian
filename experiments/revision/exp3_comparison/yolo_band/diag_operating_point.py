"""Round 2 诊断：band 带来的 recall 是"检测器变好了"还是"工作点（阈值）移动了"；预测的 band 图能不能在框内区分 TP / FP。

    python diag_operating_point.py --runs runs_cv --seeds 0 1 2 3 4 5 6 7 8 9

1. 工作点：每个 seed 把 5 折测试集预测合并（和 aggregate_cv.py 一样，评估前裁到图内），算
   - maxR：所有预测都算上时的 recall（检测器能不能提出这个目标，与阈值无关）；
   - R@P0.7 / 0.8 / 0.9：precision 不低于该值时能达到的最高 recall（同一 precision 下比 recall）；
   - val_conf：aggregate_cv.py 在合并验证集上选的阈值。
   按 seed 配对，报均值差、bootstrap 95% CI、胜/平/负。
2. 框内 band 图：测试集每个预测框（conf >= 0.05）在本 run 导出的 band 图（band_test/*.png）里取框内均值，
   用 AUC 看它区分 TP / FP（IoU 0.5 贪心匹配）的能力；"模糊区"只看 0.2 <= score < 0.6 的预测。
   boxmask arm 的图是填实框监督出来的，作对照。
3. 阈值敏感性：三个 arm 用同一个固定 score 阈值；score 阈值的几种选法（含不合规的测试集最优）；判定命中的 IoU 阈值 0.3–0.7。

输出：<runs>/diag/summary.md、summary.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
from evaluate import ap50, clip_box, collect, match_image, prf_at, select_conf  # noqa: E402

DATASETS = HERE.parents[1] / "shared" / "datasets"
EXPS = ["y8s_pre_none", "y8s_pre_band", "y8s_pre_boxmask"]
PAIRS = [("y8s_pre_none", "y8s_pre_band"), ("y8s_pre_none", "y8s_pre_boxmask"), ("y8s_pre_boxmask", "y8s_pre_band")]
BAND_EXPS = ["y8s_pre_band", "y8s_pre_boxmask"]


def auc(y, x):
    """Mann–Whitney AUC：随机取一个 TP 和一个 FP，TP 的 x 更大的概率。"""
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if not n1 or not n0:
        return float("nan")
    r = rankdata(x)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(d, rng, n=5000):
    d = np.asarray(d, float)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def operating_point(runs, gts, cv, exp, seed):
    S, T, N = [], [], 0
    for f in range(5):
        p = json.load(open(runs / "ds2" / exp / f"f{f}_seed{seed}" / "preds_test.json", encoding="utf-8"))
        s, t, n = collect(p, gts[f]["test"])
        S += list(s); T += list(t); N += n
    o = np.argsort(-np.asarray(S))
    t = np.asarray(T, float)[o]
    tp, fp = np.cumsum(t), np.cumsum(1 - t)
    P, R = tp / (tp + fp), tp / N
    out = {"maxR": float(R[-1])}
    for pt in (0.7, 0.8, 0.9):
        ok = P >= pt
        out[f"R@P{pt}"] = float(R[ok].max()) if ok.any() else 0.0
    out["val_conf"] = cv[exp][str(seed)]["primary"]["val_conf"]
    return out


def band_in_box(runs, gts, exp, seed, conf_min=0.05):
    rows = []
    for f in range(5):
        rd = runs / "ds2" / exp / f"f{f}_seed{seed}"
        P = json.load(open(rd / "preds_test.json", encoding="utf-8"))
        for stem, item in gts[f]["test"].items():
            dets = sorted([d for d in P.get(stem, []) if d[4] >= conf_min], key=lambda d: -d[4])
            w, h = item["width"], item["height"]
            dets = [clip_box(d, float(w), float(h)) for d in dets]
            m, _ = match_image(dets, item["boxes"])
            band = cv2.imread(str(rd / "band_test" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE).astype(float) / 255
            for d, (s, tp) in zip(dets, m):
                x1, y1, x2, y2 = [int(round(v)) for v in d[:4]]
                crop = band[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
                rows.append((s, tp, crop.mean()))
    r = np.asarray(rows, float)
    s, tp, bm = r[:, 0], r[:, 1], r[:, 2]
    amb = (s >= 0.2) & (s < 0.6)
    return {"auc_score": auc(tp, s), "auc_map": auc(tp, bm), "auc_score_amb": auc(tp[amb], s[amb]),
            "auc_map_amb": auc(tp[amb], bm[amb]), "n_amb": int(amb.sum()), "tp_rate_amb": float(tp[amb].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()
    runs = Path(args.runs)
    gts = {f: json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8")) for f in range(5)}
    cv = json.load(open(runs / "cv_summary.json", encoding="utf-8"))["per_exp"]
    rng = np.random.default_rng(0)

    op = {e: [operating_point(runs, gts, cv, e, s) for s in args.seeds] for e in EXPS}
    keys = list(op[EXPS[0]][0])
    L = [f"# 工作点和框内 band 图（{len(args.seeds)} seeds，5 折合并测试集 222 个目标）", "",
         "## 1. 工作点", "", "| exp | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for e in EXPS:
        L.append(f"| {e} | " + " | ".join(f"{np.mean([r[k] for r in op[e]]):.3f}" for k in keys) + " |")
    L += ["", "| A → B | 指标 | Δ [95% CI] | 胜/平/负 |", "|---|---|---|---|"]
    pairs_out = {}
    for a, b in PAIRS:
        for k in keys:
            d = np.array([op[b][i][k] - op[a][i][k] for i in range(len(args.seeds))])
            m, lo, hi = boot_ci(d, rng)
            w = f"{int((d > 1e-9).sum())}/{int((abs(d) <= 1e-9).sum())}/{int((d < -1e-9).sum())}"
            L.append(f"| {a} → {b} | {k} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {w} |")
            pairs_out[f"{a}:{b}:{k}"] = {"mean": m, "ci": [lo, hi], "win_tie_loss": w}

    L += ["", "## 2. 框内 band 图能否区分 TP / FP（AUC，0.5 = 没有信息）", "",
          "| exp | score（全部） | 框内均值（全部） | score（模糊区） | 框内均值（模糊区） | 模糊区预测数 / seed | 模糊区 TP 比例 |",
          "|---|---|---|---|---|---|---|"]
    bb = {}
    for e in BAND_EXPS:
        rs = [band_in_box(runs, gts, e, s) for s in args.seeds]
        bb[e] = rs
        g = lambda k: np.array([r[k] for r in rs])  # noqa: E731
        L.append(f"| {e} | {g('auc_score').mean():.3f} | {g('auc_map').mean():.3f} [{g('auc_map').min():.3f}, {g('auc_map').max():.3f}] | "
                 f"{g('auc_score_amb').mean():.3f} | {g('auc_map_amb').mean():.3f} [{g('auc_map_amb').min():.3f}, {g('auc_map_amb').max():.3f}] | "
                 f"{g('n_amb').mean():.0f} | {g('tp_rate_amb').mean():.2f} |")
    L += ["", "方括号 = 各 seed 的最小 / 最大值。全部 = conf >= 0.05 的预测；模糊区 = 0.2 <= score < 0.6。"]

    # ── 3. 阈值敏感性：换 score 阈值的选法、换 IoU 阈值，结论变不变
    def pooled_si(exp, seed, phase, iou=0.5):
        S, T, N = [], [], 0
        for f in range(5):
            p = json.load(open(runs / "ds2" / exp / f"f{f}_seed{seed}" / f"preds_{phase}.json", encoding="utf-8"))
            s, t, n = collect(p, gts[f][phase], iou_thr=iou)
            S.append(s); T.append(t); N += n
        return np.concatenate(S), np.concatenate(T), N

    def pair_cells(f1):
        cells = []
        for a, b in (("y8s_pre_none", "y8s_pre_band"), ("y8s_pre_boxmask", "y8s_pre_band")):
            d = np.array(f1[b]) - np.array(f1[a])
            m, lo, hi = boot_ci(d, rng)
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] {int((d > 1e-9).sum())}/{int((abs(d) <= 1e-9).sum())}/{int((d < -1e-9).sum())}")
        return cells

    data = {(e, s): (pooled_si(e, s, "val"), pooled_si(e, s, "test")) for e in EXPS for s in args.seeds}
    vconf = {k: select_conf(*v[0])[0] for k, v in data.items()}
    L += ["", "## 3. 阈值敏感性", "", "### 3a. 三个 arm 用同一个固定 score 阈值（合并测试集）", "",
          "| score 阈值 | F1：none / band / boxmask | recall：none / band | precision：none / band | band − none | band − boxmask |",
          "|---|---|---|---|---|---|"]
    thr_rows = []
    for c in (0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
        r = {e: [prf_at(*data[(e, s)][1], c) for s in args.seeds] for e in EXPS}
        f1 = {e: [x["F1"] for x in r[e]] for e in EXPS}
        mr = lambda e, k: np.mean([x[k] for x in r[e]])  # noqa: E731
        cells = pair_cells(f1)
        L.append(f"| {c:.2f} | {mr('y8s_pre_none', 'F1'):.3f} / {mr('y8s_pre_band', 'F1'):.3f} / {mr('y8s_pre_boxmask', 'F1'):.3f} | "
                 f"{mr('y8s_pre_none', 'R'):.3f} / {mr('y8s_pre_band', 'R'):.3f} | {mr('y8s_pre_none', 'P'):.3f} / {mr('y8s_pre_band', 'P'):.3f} | {cells[0]} | {cells[1]} |")
        thr_rows.append({"conf": c, **{e: float(np.mean(f1[e])) for e in EXPS}})
    rules = [("各 arm 在验证集上选（协议）", lambda e, s: prf_at(*data[(e, s)][1], vconf[(e, s)])["F1"]),
             ("都用 baseline 在验证集上选的阈值", lambda e, s: prf_at(*data[(e, s)][1], vconf[("y8s_pre_none", s)])["F1"]),
             ("固定 0.25（ultralytics 默认）", lambda e, s: prf_at(*data[(e, s)][1], 0.25)["F1"]),
             ("固定 0.5", lambda e, s: prf_at(*data[(e, s)][1], 0.5)["F1"]),
             ("各 arm 在测试集上挑最好的（不合规，只看上限）", lambda e, s: select_conf(*data[(e, s)][1])[1]),
             ("AP50（不用阈值）", lambda e, s: ap50(*data[(e, s)][1]))]
    L += ["", "### 3b. score 阈值的不同选法", "", "| 选法 | F1：none / band / boxmask | band − none | band − boxmask |", "|---|---|---|---|"]
    for name, fn in rules:
        f1 = {e: [fn(e, s) for s in args.seeds] for e in EXPS}
        cells = pair_cells(f1)
        L.append(f"| {name} | {np.mean(f1['y8s_pre_none']):.3f} / {np.mean(f1['y8s_pre_band']):.3f} / {np.mean(f1['y8s_pre_boxmask']):.3f} | {cells[0]} | {cells[1]} |")
    L += ["", "### 3c. 判定命中的 IoU 阈值（每个 IoU 下重新在验证集上选 score 阈值）", "",
          "| IoU | F1：none / band / boxmask | band − none | band − boxmask |", "|---|---|---|---|"]
    for iou in (0.3, 0.4, 0.5, 0.6, 0.7):
        f1 = {}
        for e in EXPS:
            f1[e] = []
            for s in args.seeds:
                v, t = pooled_si(e, s, "val", iou), pooled_si(e, s, "test", iou)
                f1[e].append(prf_at(*t, select_conf(*v)[0])["F1"])
        cells = pair_cells(f1)
        L.append(f"| {iou:.1f} | {np.mean(f1['y8s_pre_none']):.3f} / {np.mean(f1['y8s_pre_band']):.3f} / {np.mean(f1['y8s_pre_boxmask']):.3f} | {cells[0]} | {cells[1]} |")
    L += ["", "Δ 列：均值 [bootstrap 95% CI] 胜/平/负（按 seed 配对）。"]

    out = runs / "diag"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"seeds": args.seeds, "operating_point": op, "pairs": pairs_out, "band_in_box": bb},
              open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    print("\n".join(L))
    print(f"\n-> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
