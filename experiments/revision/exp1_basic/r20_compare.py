"""第 20 轮的配对比较（只读 run 目录；写 runs_cv/r20/summary.md，不动共享的 cv_summary.*）。

    python r20_compare.py --pairs A:B C:D ...

每个 pair 按 seed 配对（5 折合并，验证集选阈值），ΔF1 / ΔAP50 的均值、bootstrap 95% CI、Wilcoxon p、胜/平/负，
口径与 aggregate_cv.py 相同（直接调用它的 pooled_metrics / paired）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_cv import paired, pooled_metrics

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_cv" / "ds2"
FOLDS, SEEDS = range(5), range(10)

_CACHE = {}


def per_seed(exp):
    if exp not in _CACHE:
        out = {}
        for s in SEEDS:
            rds = {f: RUNS / exp / f"f{f}_seed{s}" for f in FOLDS}
            if all((rd / "metrics.json").exists() for rd in rds.values()):
                m = pooled_metrics(rds)
                out[s] = {"F1": m["test"]["F1"], "AP50": m["AP50"]}
        _CACHE[exp] = out
    return _CACHE[exp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--title", default="第 20 轮配对比较")
    ap.add_argument("--out", default=str(HERE / "runs_cv" / "r20" / "summary.md"))
    args = ap.parse_args()
    lines = [f"# {args.title}", "", "按 seed 配对（5 折合并，验证集选阈值），B − A。", "",
             "| A → B | n | F1 A | F1 B | ΔF1 [95% CI] | p | 胜/平/负 | ΔAP50 [95% CI] | 胜 |", "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for pr in args.pairs:
        a, b = pr.split(":")
        A, B = per_seed(a), per_seed(b)
        seeds = sorted(set(A) & set(B))
        if not seeds:
            lines.append(f"| {a} → {b} | 0 | | | | | | | |"); continue
        f = paired([A[s]["F1"] for s in seeds], [B[s]["F1"] for s in seeds])
        p = paired([A[s]["AP50"] for s in seeds], [B[s]["AP50"] for s in seeds])
        res[pr] = {"F1": f, "AP50": p}
        lines.append(f"| {a} → {b} | {f['n']} | {f['mean_A']:.3f} | {f['mean_B']:.3f} | {f['delta_mean']:+.3f} [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}] | "
                     f"{f['wilcoxon_p']:.3f} | {f['wins']}/{f['ties']}/{f['losses']} | {p['delta_mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['wins']}/{p['n']} |")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(res, open(out.with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(lines)); print(f"-> {out}")


if __name__ == "__main__":
    main()
