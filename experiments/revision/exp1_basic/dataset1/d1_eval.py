"""数据集 1 的分类别评估：同一个阈值（5 折合并验证集、全部类别一起选，和 aggregate_cv.py 相同）下，
测试集 F1 / P / R / AP50 分别按全部、pipe、rebar 报告，并按 seed 配对比较。

    python d1_eval.py --exps d1_s224_none d1_s448_none ...            # 每个实验一行（含合并验证 F1，选 config 用）
    python d1_eval.py --pairs d1_s448gb_none:d1_s448gb_abs ... --out r2_main.md

为什么要分类别：pipe 145 张图 / 297 个目标，rebar 25 张图 / 303 个目标。按目标计数的合并 F1 里两者各占一半，
几张密集 rebar 图就能左右总数；而 rebar 的 band 几乎是框状（dataset1/data_stats.md），band 形状的作用只能在 pipe 上看。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent
sys.path.insert(0, str(EXP1))
sys.path.insert(0, str(EXP1.parent / "shared"))
from aggregate_cv import data_dir_of, paired  # noqa: E402
from evaluate import ap50, collect, prf_at, select_conf  # noqa: E402

CATS = ("all", "pipe", "rebar")
FOLDS = (0, 1, 2, 3, 4)


def find_runs(runs_root, exp):
    """{seed: {fold: run_dir}}，只保留 5 折都跑完的 seed。"""
    tree = defaultdict(dict)
    for m in (runs_root / "ds1" / exp).glob("f*_seed*/metrics.json"):
        f, s = m.parent.name.split("_seed")
        tree[int(s)][int(f[1:])] = m.parent
    return {s: d for s, d in sorted(tree.items()) if set(d) == set(FOLDS)}


def _filter(gt, cat):
    return gt if cat == "all" else {k: v for k, v in gt.items() if v.get("category") == cat}


def seed_metrics(run_dirs, gt_name="gt_boxes.json"):
    pools = {ph: {c: ([], [], 0) for c in CATS} for ph in ("val", "test")}
    for fold, rd in sorted(run_dirs.items()):
        ri = json.load(open(rd / "run_info.json", encoding="utf-8"))
        gt = json.load(open(data_dir_of(ri, fold) / gt_name, encoding="utf-8"))
        for ph in ("val", "test"):
            preds = json.load(open(rd / f"preds_{ph}.json", encoding="utf-8"))
            for c in CATS:
                s, t, n = collect(preds, _filter(gt[ph], c))
                a, b, k = pools[ph][c]
                pools[ph][c] = (a + [s], b + [t], k + n)
    cat = lambda x: (np.concatenate(x[0]), np.concatenate(x[1]), x[2])   # noqa: E731
    sv, tv, nv = cat(pools["val"]["all"])
    conf, f1v = select_conf(sv, tv, nv)
    out = {"val_conf": float(conf), "val_F1": float(f1v)}
    for c in CATS:
        st, tt, nt = cat(pools["test"][c])
        m = prf_at(st, tt, nt, conf)
        out[c] = {"F1": m["F1"], "P": m["P"], "R": m["R"], "AP50": float(ap50(st, tt, nt)), "n_gt": int(nt)}
    return out


def exp_metrics(runs_root, exp):
    return {s: seed_metrics(rds) for s, rds in find_runs(runs_root, exp).items()}


def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else f"{np.mean(v):.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(EXP1 / "runs_cv"))
    ap.add_argument("--exps", nargs="*", default=[])
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="同时写到 dataset1/results/<out>")
    args = ap.parse_args()
    runs = Path(args.runs)
    exps = list(dict.fromkeys(args.exps + [e for p in args.pairs for e in p.split(":")]))
    R = {e: exp_metrics(runs, e) for e in exps}

    lines = ["| 实验 | seeds | 合并验证 F1（选参用） | 测试 F1 全部 | pipe | rebar | AP50 全部 | pipe | rebar | R pipe | R rebar |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in exps:
        r = R[e]
        if not r:
            lines.append(f"| {e} | 0 | | | | | | | | | |")
            continue
        g = lambda c, k: [x[c][k] for x in r.values()]   # noqa: E731
        lines.append(f"| {e} | {len(r)} | {ms([x['val_F1'] for x in r.values()])} | {ms(g('all', 'F1'))} | {ms(g('pipe', 'F1'))} | "
                     f"{ms(g('rebar', 'F1'))} | {np.mean(g('all', 'AP50')):.3f} | {np.mean(g('pipe', 'AP50')):.3f} | "
                     f"{np.mean(g('rebar', 'AP50')):.3f} | {np.mean(g('pipe', 'R')):.3f} | {np.mean(g('rebar', 'R')):.3f} |")
    if args.pairs:
        lines += ["", "| A → B | n | 类别 | F1 A | F1 B | ΔF1 [95% CI] | p | 胜/平/负 | ΔAP50 |", "|---|---|---|---|---|---|---|---|---|"]
        for pr in args.pairs:
            a, b = pr.split(":")
            seeds = sorted(set(R[a]) & set(R[b]))
            if not seeds:
                continue
            for c in CATS:
                d = paired([R[a][s][c]["F1"] for s in seeds], [R[b][s][c]["F1"] for s in seeds])
                dap = paired([R[a][s][c]["AP50"] for s in seeds], [R[b][s][c]["AP50"] for s in seeds])
                lines.append(f"| {a} → {b} | {d['n']} | {c} | {d['mean_A']:.3f} | {d['mean_B']:.3f} | {d['delta_mean']:+.3f} "
                             f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {d['wilcoxon_p']:.3f} | {d['wins']}/{d['ties']}/{d['losses']} | "
                             f"{dap['delta_mean']:+.3f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / args.out).write_text(text, encoding="utf-8")
        json.dump({e: {str(s): v for s, v in r.items()} for e, r in R.items()},
                  open(HERE / "results" / (Path(args.out).stem + ".json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
