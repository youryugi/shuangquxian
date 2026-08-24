"""
Cross-architecture reproduction of run_invert_ablation.py: same clean
ablation (all built-in ultralytics augmentations off, only the true
polarity-inversion augmentation INVERT_AUG varies) but on the larger
pretrained yolov8s.pt backbone instead of yolov8n.pt, to check whether
the invert_aug effect found on yolov8n also holds on a different model
size/capacity.

Trains with INVERT_AUG on and off, 5 random seeds each (10 runs total),
and prints/saves mAP50 / mAP50-95 per run plus the mean +/- std for
each condition.

Run inside the `gpr` conda env:

    conda activate gpr
    python run_invert_ablation_yolov8s.py
"""
import csv
import os
import statistics
from types import SimpleNamespace

from train_yolo_bbox import (
    AUG_PARAMS, DATA_DIR, WORK_DIR, EPOCHS, IMGSZ, BATCH, VAL_RATIO,
    POLARITY_DIR, INVERT_DIR, train,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(_HERE, "yolov8s.pt")

SEEDS = [0, 1, 2, 3, 4]
WORK_ROOT = os.path.join(WORK_DIR, "invert_ablation_yolov8s")
RESULTS_CSV = os.path.join(WORK_ROOT, "results.csv")


def run_one(invert_aug, seed):
    work_dir = os.path.join(WORK_ROOT, f"invert_{'on' if invert_aug else 'off'}_seed{seed}")
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
        **{name: default for name, (default, _help) in AUG_PARAMS.items()},  # all built-in augs off
    )
    _best, map50, map5095 = train(args, cleanup_dataset=True)
    return float(map50), float(map5095)


def main():
    os.makedirs(WORK_ROOT, exist_ok=True)
    rows = []
    for invert_aug in (True, False):
        for seed in SEEDS:
            print(f"\n=== [yolov8s] invert_aug={invert_aug} seed={seed} ===")
            map50, map5095 = run_one(invert_aug, seed)
            rows.append({"invert_aug": invert_aug, "seed": seed, "map50": map50, "map50_95": map5095})
            with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["invert_aug", "seed", "map50", "map50_95"])
                writer.writeheader()
                writer.writerows(rows)

    print("\n=== summary (yolov8s.pt, all built-in augs off) ===")
    for invert_aug in (True, False):
        group = [r for r in rows if r["invert_aug"] == invert_aug]
        map50s = [r["map50"] for r in group]
        map5095s = [r["map50_95"] for r in group]
        label = "invert_aug=ON " if invert_aug else "invert_aug=OFF"
        print(f"{label}  mAP50={statistics.mean(map50s):.4f}+/-{statistics.pstdev(map50s):.4f}  "
              f"mAP50-95={statistics.mean(map5095s):.4f}+/-{statistics.pstdev(map5095s):.4f}")
    print(f"\n[ablation] per-run results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
