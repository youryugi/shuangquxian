"""
Locate which built-in ultralytics augmentation(s) flip invert_aug (true
polarity-inversion augmentation) from helpful to harmful.

Background: run_invert_ablation.py (all built-in augs off) found
invert_aug significantly *improves* mAP50-95 (paired t-test p=0.011).
run_invert_ablation_default_aug.py (all of ultralytics' own default augs
on) found invert_aug significantly *hurts* mAP50-95 (p=0.015) instead.
Something in the default augmentation set interacts badly with
invert_aug on this small (186 train image) dataset.

The rendered radargrams are pixel-exact grayscale (R=G=B everywhere --
verified directly), so hsv_h (hue) and hsv_s (saturation) are no-ops here
regardless of their value: they're excluded from consideration. That
leaves 5 real candidates that differ from the clean baseline in the
default recipe: hsv_v, translate, scale, fliplr, mosaic.

This script runs a leave-one-out design: start from the full default-aug
recipe and remove exactly one of those 5 at a time (set it back to 0),
re-running the invert_aug on/off x 5-seed ablation each time. Whichever
removal restores a positive/insignificant invert_aug effect identifies
the culprit augmentation. `mosaic` is the prime suspect (it splices 4
images together, so an inverted-polarity tile can land next to a
normal-polarity tile in the same mosaic -- unlike the other 4, which are
simple per-image photometric/affine transforms).

Each condition is 2 (invert_aug on/off) x 5 seeds = 10 runs; there are 5
conditions, so a full sweep is 50 runs (~3+ hours at ~4 min/run). Pass
--condition to run a subset instead of everything.

Run inside the `gpr` conda env:

    conda activate gpr
    python run_invert_ablation_aug_interaction.py --condition no_mosaic
    python run_invert_ablation_aug_interaction.py --condition all       # all 5, ~3h+
    python run_invert_ablation_aug_interaction.py --list                # show condition names, don't train
"""
import argparse
import csv
import os
import statistics
from types import SimpleNamespace

from train_yolo_bbox import (
    DATA_DIR, WORK_DIR, MODEL, EPOCHS, IMGSZ, BATCH, VAL_RATIO,
    POLARITY_DIR, INVERT_DIR, train,
)
from run_invert_ablation_default_aug import YOLO_DEFAULT_AUG

SEEDS = [0, 1, 2]
WORK_ROOT = os.path.join(WORK_DIR, "invert_ablation_aug_interaction")

# Each condition = the full default-aug recipe with exactly one of the 5
# real (non-no-op) augmentations zeroed back out.
REAL_CANDIDATES = ["mosaic", "hsv_v", "translate", "scale", "fliplr"]
CONDITIONS = {
    f"no_{name}": {**YOLO_DEFAULT_AUG, name: 0.0}
    for name in REAL_CANDIDATES
}


def run_one(condition_name, invert_aug, seed):
    aug_kwargs = CONDITIONS[condition_name]
    work_dir = os.path.join(WORK_ROOT, condition_name, f"invert_{'on' if invert_aug else 'off'}_seed{seed}")
    args = SimpleNamespace(
        data_dir=DATA_DIR,
        work_dir=work_dir,
        model=MODEL,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        val_ratio=VAL_RATIO,
        seed=seed,
        polarity_dir=POLARITY_DIR,
        polarity_aug=False,
        invert_dir=INVERT_DIR,
        invert_aug=invert_aug,
        **aug_kwargs,
    )
    _best, map50, map5095 = train(args)
    return float(map50), float(map5095)


def run_condition(condition_name, seeds=None):
    seeds = SEEDS if seeds is None else seeds
    condition_root = os.path.join(WORK_ROOT, condition_name)
    os.makedirs(condition_root, exist_ok=True)
    results_csv = os.path.join(condition_root, "results.csv")
    rows = []
    for invert_aug in (True, False):
        for seed in seeds:
            print(f"\n=== [{condition_name}] invert_aug={invert_aug} seed={seed} "
                  f"(aug={CONDITIONS[condition_name]}) ===")
            map50, map5095 = run_one(condition_name, invert_aug, seed)
            rows.append({"invert_aug": invert_aug, "seed": seed, "map50": map50, "map50_95": map5095})
            with open(results_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["invert_aug", "seed", "map50", "map50_95"])
                writer.writeheader()
                writer.writerows(rows)

    print(f"\n=== summary [{condition_name}] ===")
    for invert_aug in (True, False):
        group = [r for r in rows if r["invert_aug"] == invert_aug]
        map50s = [r["map50"] for r in group]
        map5095s = [r["map50_95"] for r in group]
        label = "invert_aug=ON " if invert_aug else "invert_aug=OFF"
        print(f"{label}  mAP50={statistics.mean(map50s):.4f}+/-{statistics.pstdev(map50s):.4f}  "
              f"mAP50-95={statistics.mean(map5095s):.4f}+/-{statistics.pstdev(map5095s):.4f}")
    print(f"[ablation] per-run results -> {results_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["all", *CONDITIONS.keys()], default=None,
                    help="Which leave-one-out condition to run. 'all' runs all 5 (~3h+).")
    p.add_argument("--list", action="store_true", help="Print condition names/aug values and exit.")
    p.add_argument("--num-seeds", type=int, default=None,
                    help=f"Override seed count, uses seeds 0..N-1 (default: module SEEDS={SEEDS}).")
    args = p.parse_args()

    if args.list or args.condition is None:
        print("Available conditions (each = default-aug recipe with one aug zeroed out):")
        for name, kwargs in CONDITIONS.items():
            changed = {k: v for k, v in kwargs.items() if k in REAL_CANDIDATES}
            print(f"  {name:15s} {changed}")
        if args.condition is None:
            print("\nPass --condition <name> or --condition all to run.")
            return

    seeds = list(range(args.num_seeds)) if args.num_seeds is not None else None
    names = list(CONDITIONS) if args.condition == "all" else [args.condition]
    for name in names:
        run_condition(name, seeds=seeds)


if __name__ == "__main__":
    main()
