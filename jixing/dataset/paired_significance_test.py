"""
Paired significance test for the invert_aug (true polarity-inversion
augmentation) ablation results produced by run_invert_ablation.py.

For each backbone condition (pretrained / scratch) and each metric
(mAP50, mAP50-95), pairs the ON/OFF runs by seed (same seed = same
train/val split and model init/batch order, only invert_aug differs)
and runs:
  - paired t-test (scipy.stats.ttest_rel)
  - Wilcoxon signed-rank test (scipy.stats.wilcoxon) -- distribution-free,
    more appropriate given n=5 and no normality assumption
  - Cohen's d for paired samples (mean diff / std of the differences)

Run inside the `gpr` conda env:

    conda activate gpr
    python paired_significance_test.py
"""
import csv
import os
import statistics

from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))

RESULT_FILES = {
    "pretrained (yolov8n.pt)": os.path.join(_HERE, "yolo_runs", "invert_ablation", "results.csv"),
    "scratch (yolov8n.yaml)": os.path.join(_HERE, "yolo_runs", "invert_ablation_scratch", "results.csv"),
}
METRICS = ["map50", "map50_95"]


def load_paired(path, metric):
    """Return (on_values, off_values), both ordered by seed ascending."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    on = {int(r["seed"]): float(r[metric]) for r in rows if r["invert_aug"] == "True"}
    off = {int(r["seed"]): float(r[metric]) for r in rows if r["invert_aug"] == "False"}
    seeds = sorted(set(on) & set(off))
    missing = sorted(set(on) ^ set(off))
    if missing:
        raise ValueError(f"{path}: seeds {missing} are missing an ON or OFF run, cannot pair")
    return [on[s] for s in seeds], [off[s] for s in seeds], seeds


def cohens_d_paired(diffs):
    sd = statistics.pstdev(diffs)
    if sd == 0:
        return float("inf") if statistics.mean(diffs) != 0 else 0.0
    return statistics.mean(diffs) / sd


def main():
    for label, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue
        print(f"\n=== {label} ===")
        for metric in METRICS:
            on_vals, off_vals, seeds = load_paired(path, metric)
            diffs = [o - f for o, f in zip(on_vals, off_vals)]
            n = len(diffs)
            mean_diff = statistics.mean(diffs)
            d = cohens_d_paired(diffs)

            t_res = stats.ttest_rel(on_vals, off_vals)
            # Wilcoxon needs at least one non-zero difference and n>=1;
            # with all-positive/negative diffs and small n it can hit the
            # "exact" branch's zero-method edge cases, so fall back gracefully.
            try:
                w_res = stats.wilcoxon(on_vals, off_vals)
                w_stat, w_p = w_res.statistic, w_res.pvalue
            except ValueError as e:
                w_stat, w_p = float("nan"), float("nan")
                print(f"  [wilcoxon skipped: {e}]")

            n_pos = sum(1 for d_ in diffs if d_ > 0)
            print(f"  {metric}: seeds={seeds}")
            print(f"    per-seed diff (ON-OFF): {[round(d_, 4) for d_ in diffs]}  "
                  f"({n_pos}/{n} seeds favor ON)")
            print(f"    mean diff={mean_diff:+.4f}  Cohen's d (paired)={d:.2f}")
            print(f"    paired t-test:      t={t_res.statistic:.3f}  p={t_res.pvalue:.4f}")
            print(f"    Wilcoxon signed-rank: W={w_stat:.3f}  p={w_p:.4f}"
                  f"  (n={n}: exact test's smallest possible two-sided p is 2/2^n={2/2**n:.4f})")


if __name__ == "__main__":
    main()
