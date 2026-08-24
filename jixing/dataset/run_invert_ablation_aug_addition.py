"""
Add-one-at-a-time counterpart to run_invert_ablation_aug_interaction.py.

That script started from the full default-aug recipe (all of ultralytics'
built-in augmentations on) and removed one augmentation at a time
(leave-one-out) to find which one flips invert_aug from helpful to
harmful. With only 3-10 seeds per condition, none of the 5 removals
significantly restored a positive invert_aug effect -- so no single
augmentation, removed from the full set, explains the reversal. That
could mean the interaction only shows up when *multiple* augmentations
are stacked (leave-one-out can't isolate that), or that removing from an
already-crowded recipe is just too noisy to read.

This script instead starts from the clean all-off baseline (same as
run_invert_ablation.py) and adds exactly one of the 5 real
(non-no-op-on-grayscale) candidate augmentations at a time -- mosaic,
hsv_v, translate, scale, fliplr -- at its ultralytics default value, then
runs the same invert_aug on/off x N-seed comparison. This isolates each
augmentation's effect on invert_aug in the simplest possible setting
(one variable added at a time to an otherwise-clean recipe), which may
separate the individual factors more clearly than leave-one-out did.

Each condition is 2 (invert_aug on/off) x N seeds; there are 5
conditions. Default N=5 (50 runs total, ~3+ hours). Pass --condition to
run a subset and --num-seeds to change N.

Run inside the `gpr` conda env:

    conda activate gpr
    python run_invert_ablation_aug_addition.py --condition only_mosaic
    python run_invert_ablation_aug_addition.py --condition all --num-seeds 5
    python run_invert_ablation_aug_addition.py --condition all --scratch  # yolov8n.yaml, random init
    python run_invert_ablation_aug_addition.py --list
"""
import argparse
import csv
import os
import statistics
from types import SimpleNamespace

from train_yolo_bbox import (
    AUG_PARAMS, DATA_DIR, WORK_DIR, MODEL, EPOCHS, IMGSZ, BATCH, VAL_RATIO,
    POLARITY_DIR, INVERT_DIR, train,
)
from run_invert_ablation_default_aug import YOLO_DEFAULT_AUG

SEEDS = [0, 1, 2, 3, 4]
SCRATCH_MODEL = "yolov8n.yaml"

ALL_OFF = {name: default for name, (default, _help) in AUG_PARAMS.items()}
REAL_CANDIDATES = ["mosaic", "hsv_v", "translate", "scale", "fliplr"]
CONDITIONS = {
    f"only_{name}": {**ALL_OFF, name: YOLO_DEFAULT_AUG[name]}
    for name in REAL_CANDIDATES
}


def run_one(condition_name, invert_aug, seed, model, work_root):
    aug_kwargs = CONDITIONS[condition_name]
    work_dir = os.path.join(work_root, condition_name, f"invert_{'on' if invert_aug else 'off'}_seed{seed}")
    args = SimpleNamespace(
        data_dir=DATA_DIR,
        work_dir=work_dir,
        model=model,
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


def run_condition(condition_name, model, work_root, seeds=None):
    seeds = SEEDS if seeds is None else seeds
    condition_root = os.path.join(work_root, condition_name)
    os.makedirs(condition_root, exist_ok=True)
    results_csv = os.path.join(condition_root, "results.csv")
    rows = []
    for invert_aug in (True, False):
        for seed in seeds:
            print(f"\n=== [{condition_name}, model={model}] invert_aug={invert_aug} seed={seed} "
                  f"(aug={CONDITIONS[condition_name]}) ===")
            map50, map5095 = run_one(condition_name, invert_aug, seed, model, work_root)
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
                    help="Which add-one condition to run. 'all' runs all 5.")
    p.add_argument("--list", action="store_true", help="Print condition names/aug values and exit.")
    p.add_argument("--num-seeds", type=int, default=None,
                    help=f"Override seed count, uses seeds 0..N-1 (default: module SEEDS={SEEDS}).")
    p.add_argument("--scratch", action="store_true",
                    help="Train from yolov8n.yaml (random init) instead of pretrained yolov8n.pt.")
    args = p.parse_args()

    if args.list or args.condition is None:
        print("Available conditions (each = clean all-off baseline with one aug added at its ultralytics default):")
        for name, kwargs in CONDITIONS.items():
            changed = {k: v for k, v in kwargs.items() if k in REAL_CANDIDATES}
            print(f"  {name:15s} {changed}")
        if args.condition is None:
            print("\nPass --condition <name> or --condition all to run.")
            return

    model = SCRATCH_MODEL if args.scratch else MODEL
    work_root = os.path.join(WORK_DIR, "invert_ablation_aug_addition_scratch" if args.scratch
                              else "invert_ablation_aug_addition")
    seeds = list(range(args.num_seeds)) if args.num_seeds is not None else None
    names = list(CONDITIONS) if args.condition == "all" else [args.condition]
    for name in names:
        run_condition(name, model, work_root, seeds=seeds)


if __name__ == "__main__":
    main()
