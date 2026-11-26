"""实验 0 的汇总：每个 arm、每个 seed 把 5 折合并（与 exp1_basic/aggregate_cv.py 相同口径），再按 seed 配对。

    python aggregate_curve.py --exps curve_local curve_hough ... --pairs curve_local:curve_hough ...

合并：5 折验证集预测合在一起选一个阈值，5 折测试集（= 全部 131 张）在该阈值下算 F1；另给 AP50。
--ref 把 exp1 的 run（runs_cv/ds2/<exp>）也读进来作参照行（同一 fold / seed 划分，backbone 初始化相同）。
曲线指标：测试集上按框 IoU ≥ 0.5 一对一匹配到的 GT（阈值 = 合并验证集选的），顶点误差（px）和 |log slope 比|，
只对有 HBAT 参数的 GT 计算。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "exp1_basic"
DATASETS = HERE.parent / "shared" / "datasets"
sys.path.insert(0, str(EXP1))
sys.path.insert(0, str(HERE.parent / "shared"))
from evaluate import ap50, collect, prf_at, select_conf, box_iou, clip_box  # noqa: E402
from aggregate_cv import paired  # noqa: E402


def run_dirs(root, exp, seed):
    out = {}
    for f in range(5):
        rd = root / "ds2" / exp / f"f{f}_seed{seed}"
        if (rd / "preds_test.json").exists() and (rd / "preds_val.json").exists():
            out[f] = rd
    return out


def pooled(rds):
    sv, tv, nv, st, tt, nt = [], [], 0, [], [], 0
    for f, rd in sorted(rds.items()):
        gt = json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))
        s, t, n = collect(json.load(open(rd / "preds_val.json")), gt["val"]); sv.append(s); tv.append(t); nv += n
        s, t, n = collect(json.load(open(rd / "preds_test.json")), gt["test"]); st.append(s); tt.append(t); nt += n
    sv, tv, st, tt = map(np.concatenate, (sv, tv, st, tt))
    conf, f1v = select_conf(sv, tv, nv)
    r = prf_at(st, tt, nt, conf)
    return {"val_conf": float(conf), "val_F1": float(f1v), "test_F1": float(r["F1"]), "P": float(r["P"]), "R": float(r["R"]),
            "AP50": float(ap50(st, tt, nt)), "n_test_gt": int(nt)}


def curve_errors(rds, conf):
    """测试集：阈值以上的预测（按分数降序）与 GT 框贪心一对一匹配（IoU ≥ 0.5），对命中且有 HBAT 参数的 GT 算误差。"""
    sys.path.insert(0, str(HERE))
    import train_curve as tc
    ap_err, sl_err = [], []
    for f, rd in sorted(rds.items()):
        cpath = rd / "curves_test.json"
        if not cpath.exists():
            return None
        curves = json.load(open(cpath))
        boxes = json.load(open(rd / "preds_test.json"))
        ds = tc.CurveDataset(DATASETS / f"ds2_f{f}_s100_hbatbox", "test")
        for stem in ds.stems:
            gts = [c for c in ds.params[stem]]
            gtb = [tc.curve_box(c["xv"], c["yv"], c["s"], c["w"], c["th"]) for c in gts]
            dets = [(b, c) for b, c in zip(boxes.get(stem, []), curves.get(stem, [])) if b[4] >= conf]
            dets.sort(key=lambda d: -d[0][4])
            used = [False] * len(gts)
            for b, c in dets:
                bi, bj = 0.0, -1
                for j, g in enumerate(gtb):
                    if not used[j] and g is not None:
                        iou = box_iou(clip_box(b[:4], 224, 224), g)
                        if iou > bi:
                            bi, bj = iou, j
                if bi >= 0.5:
                    used[bj] = True
                    g = gts[bj]
                    if g["src"] == "band":
                        ap_err.append(math.hypot(c[0] - g["xv"], c[1] - g["yv"]))
                        sl_err.append(abs(math.log(c[2] / g["s"])))
    return {"apex_px_median": float(np.median(ap_err)), "slope_abslog_median": float(np.median(sl_err)), "n": len(ap_err)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--ref", nargs="*", default=[], help="exp1_basic/runs_cv 里的参照 exp")
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--curve-metrics", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    roots = {e: HERE / "runs" for e in args.exps}
    roots.update({e: EXP1 / "runs_cv" for e in args.ref})
    res = {}
    for e, root in roots.items():
        per = {}
        for s in args.seeds:
            rds = run_dirs(root, e, s)
            if len(rds) == 5:
                per[s] = pooled(rds)
                if args.curve_metrics and root == HERE / "runs":
                    per[s]["curve"] = curve_errors(rds, per[s]["val_conf"])
        res[e] = per
    lines = ["| exp | n seeds | val F1 | test F1 | P | R | AP50 | apex err px | slope |log| |", "|---|---|---|---|---|---|---|---|---|"]
    for e, per in res.items():
        if not per:
            lines.append(f"| {e} | 0 | | | | | | | |"); continue
        g = lambda k: np.mean([v[k] for v in per.values()])
        cv = [v["curve"] for v in per.values() if v.get("curve")]
        ce = (f"{np.mean([c['apex_px_median'] for c in cv]):.1f}", f"{np.mean([c['slope_abslog_median'] for c in cv]):.3f}") if cv else ("", "")
        sd = np.std([v["test_F1"] for v in per.values()], ddof=1) if len(per) > 1 else 0.0
        lines.append(f"| {e} | {len(per)} | {g('val_F1'):.3f} | {g('test_F1'):.3f} ± {sd:.3f} | {g('P'):.3f} | {g('R'):.3f} | {g('AP50'):.3f} | {ce[0]} | {ce[1]} |")
    lines += ["", "| A → B | n | ΔF1 [95% CI] | Wilcoxon p | 胜/平/负 | ΔAP50 [95% CI] |", "|---|---|---|---|---|---|"]
    out_pairs = {}
    for pr in args.pairs:
        a, b = pr.split(":")
        common = sorted(set(res.get(a, {})) & set(res.get(b, {})))
        if len(common) < 2:
            continue
        pf = paired([res[a][s]["test_F1"] for s in common], [res[b][s]["test_F1"] for s in common])
        pa = paired([res[a][s]["AP50"] for s in common], [res[b][s]["AP50"] for s in common])
        out_pairs[pr] = {"F1": pf, "AP50": pa}
        lines.append(f"| {a} → {b} | {pf['n']} | {pf['delta_mean']:+.3f} [{pf['ci95'][0]:+.3f}, {pf['ci95'][1]:+.3f}] | {pf['wilcoxon_p']:.3f} | "
                     f"{pf['wins']}/{pf['ties']}/{pf['losses']} | {pa['delta_mean']:+.3f} [{pa['ci95'][0]:+.3f}, {pa['ci95'][1]:+.3f}] |")
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        Path(args.out).with_suffix(".md").write_text(txt + "\n", encoding="utf-8")
        json.dump({"per_seed": res, "pairs": out_pairs}, open(Path(args.out).with_suffix(".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
