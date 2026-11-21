"""第 17 轮分析：参数化头 vs 修好翻转的 baseline vs 旧 baseline。只读 runs_cv 的预测，不训练。

    python r17_analysis.py [--runs runs_cv] [--out runs_cv/r17]

协议与 aggregate_cv.py 相同（每个 seed：5 折验证集合并选阈值，5 折测试集合并 = 131 张 / 222 个 HBAT box；
evaluate.py 的新口径，预测先裁到图内）。输出：
  1. 各 exp 的合并 F1 / AP50 / F1@IoU0.7（均值 ± sd）
  2. 按 seed 配对的差值（bootstrap 95% CI、Wilcoxon p、胜/平/负）
  3. 框中心的水平偏差（命中框的 预测 − GT，px）：验证翻转 bug 修好后偏差消失
  4. 参数误差（命中的、有真实参数的 GT）：顶点距离、|log(slope 比)|、span 相对误差。
     param / pseudo 用网络输出的参数；box head 用预测框反推（pseudo_params），即"框检测器能给出的参数"
写 <out>/summary.md、summary.json。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "detector_comparison"))
from aggregate_cv import paired, pooled_metrics  # noqa: E402
from evaluate import clip_box, collect, prf_at, select_conf  # noqa: E402
import train_param as tp  # noqa: E402

DATASETS = HERE.parent / "detector_comparison" / "datasets"
EXPS = ["hbatbox_aug_none", "hbatbox_augfix_none", "hbatbox_flipfix_none","hbatbox_augfix_pseudo", "hbatbox_augfix_param",
        "hbatbox_aug_abs", "hbatbox_aug_abs_bandbox", "hbatbox_aug_none_s2plain", "hbatbox_aug_abs_fitbox"]
PAIRS = ["hbatbox_aug_none:hbatbox_augfix_none",        # 只修翻转
         "hbatbox_flipfix_none:hbatbox_augfix_none",    # 交叉检查：train_basic.py 修翻转（另一会话）vs train_param.py box head
         "hbatbox_augfix_none:hbatbox_augfix_pseudo",   # 参数化输出 + GIoU（只用框标注）
         "hbatbox_augfix_none:hbatbox_augfix_param",    # 参数化输出 + 真实 HBAT 参数
         "hbatbox_augfix_pseudo:hbatbox_augfix_param",  # 真实参数 vs 反推参数 = band 标注的几何信息
         "hbatbox_aug_none_s2plain:hbatbox_augfix_none",
         "hbatbox_aug_none_s2plain:hbatbox_augfix_param",
         "hbatbox_aug_abs_bandbox:hbatbox_augfix_param"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def run_dirs(runs, exp, seed):
    d = {f: runs / "ds2" / exp / f"f{f}_seed{seed}" for f in range(5)}
    return d if all((p / "metrics.json").exists() for p in d.values()) else None


def pooled_iou07(rds):
    sv, tv, nv, st, tt, nt = [], [], 0, [], [], 0
    for f, rd in rds.items():
        gt = json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))
        pv = json.load(open(rd / "preds_val.json")); pt = json.load(open(rd / "preds_test.json"))
        s, t, n = collect(pv, gt["val"]); sv.append(s); tv.append(t); nv += n
        s, t, n = collect(pt, gt["test"], iou_thr=0.7); st.append(s); tt.append(t); nt += n
    conf, _ = select_conf(np.concatenate(sv), np.concatenate(tv), nv)
    return prf_at(np.concatenate(st), np.concatenate(tt), nt, conf)["F1"]


def matched(rds, conf, gt_inst):
    """在阈值 conf 下逐图贪心匹配（分数降序、IoU ≥ 0.5、预测先裁到图内），返回 [(gt_inst, det_idx, det, fold, stem)]。"""
    out = []
    for f, rd in rds.items():
        pt = json.load(open(rd / "preds_test.json"))
        prm = json.load(open(rd / "preds_test_params.json")) if (rd / "preds_test_params.json").exists() else None
        for stem, insts in gt_inst[f].items():
            dets = [(i, clip_box(d, tp.INPUT, tp.INPUT)) for i, d in enumerate(pt.get(stem, [])) if d[4] >= conf]
            dets.sort(key=lambda x: -x[1][4])
            used = [False] * len(insts)
            for i, d in dets:
                best, bj = 0.5, -1
                for j, g in enumerate(insts):
                    if not used[j]:
                        v = iou(d, g["box"])
                        if v >= best:
                            best, bj = v, j
                if bj >= 0:
                    used[bj] = True
                    out.append((insts[bj], d, prm[stem][i] if prm is not None else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()
    runs = Path(args.runs)
    out = Path(args.out) if args.out else runs / "r17"
    out.mkdir(parents=True, exist_ok=True)

    gt_inst = {f: tp.ParamDataset(DATASETS / f"ds2_f{f}_s100_hbatbox", "test", "param").inst for f in range(5)}

    per = {}
    for exp in EXPS:
        rows = {}
        for s in args.seeds:
            rds = run_dirs(runs, exp, s)
            if rds is None:
                continue
            pm = pooled_metrics(rds)
            r = {"F1": pm["test"]["F1"], "P": pm["test"]["P"], "R": pm["test"]["R"], "AP50": pm["AP50"],
                 "F1_iou07": pooled_iou07(rds), "conf": pm["val_conf"]}
            m = matched(rds, pm["val_conf"], gt_inst)
            dx = [((d[0] + d[2]) - (g["box"][0] + g["box"][2])) / 2 for g, d, _ in m]
            r["center_dx_mean"] = float(np.mean(dx)) if dx else None
            ev, es, ep = [], [], []
            for g, d, p in m:
                if g["real"] is None:
                    continue
                xv, yv, sl, sp, _ = g["real"]
                if p is not None:
                    pxv, pyv, psl, psp = p[0], p[1], p[2], p[3]
                else:
                    pxv, pyv, psl, psp, _ = tp.pseudo_params(d[:4])
                ev.append(math.hypot(pxv - xv, pyv - yv)); es.append(abs(math.log(psl / sl))); ep.append(abs(psp - sp) / sp)
            r["vertex_err_px"] = float(np.median(ev)) if ev else None
            r["slope_abs_logratio"] = float(np.median(es)) if es else None
            r["span_rel_err"] = float(np.median(ep)) if ep else None
            r["n_matched_real"] = len(ev)
            rows[s] = r
        if rows:
            per[exp] = rows
        print(exp, len(rows), "seeds", flush=True)

    def agg(exp, k):
        v = [r[k] for r in per[exp].values() if r.get(k) is not None]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (None, None)

    pairs = {}
    for pr in PAIRS:
        a, b = pr.split(":")
        if a not in per or b not in per:
            continue
        common = sorted(set(per[a]) & set(per[b]))
        if len(common) < 2:
            continue
        pairs[pr] = {k: paired([per[a][s][k] for s in common], [per[b][s][k] for s in common]) for k in ("F1", "AP50", "F1_iou07")}

    json.dump({"per_seed": per, "pairs": pairs}, open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    L = ["# 第 17 轮：参数化头（新口径：预测裁到图内）", "",
         "| exp | seeds | F1 | AP50 | F1@IoU0.7 | 框中心 dx (px) | 顶点误差中位数 (px) | slope |log 比| 中位数 | span 相对误差中位数 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for exp in per:
        f = lambda k, d=3: ("—" if agg(exp, k)[0] is None else f"{agg(exp, k)[0]:.{d}f}")
        F, Fs = agg(exp, "F1")
        L.append(f"| {exp} | {len(per[exp])} | {F:.3f} ± {Fs:.3f} | {f('AP50')} | {f('F1_iou07')} | {f('center_dx_mean', 2)} | "
                 f"{f('vertex_err_px', 1)} | {f('slope_abs_logratio')} | {f('span_rel_err')} |")
    L += ["", "| A → B | n | ΔF1 [95% CI] | p | 胜/平/负 | ΔAP50 | ΔF1@0.7 |", "|---|---|---|---|---|---|---|"]
    for pr, d in pairs.items():
        p = d["F1"]
        L.append(f"| {pr.replace(':', ' → ')} | {p['n']} | {p['delta_mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | "
                 f"{p['wilcoxon_p']:.3f} | {p['wins']}/{p['ties']}/{p['losses']} | {d['AP50']['delta_mean']:+.3f} | {d['F1_iou07']['delta_mean']:+.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
