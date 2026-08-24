"""
Ablation: does the true-polarity-inversion augmentation (INVERT_AUG in
train_yolo_bbox.py -- adds each train image's sign-flipped data -> -data
render as an extra training sample) help YOLO detection?

Trains with INVERT_AUG on and off, 5 random seeds each (10 runs total),
and prints/saves a summary of val mAP50 / mAP50-95 per run plus the
mean +/- std for each condition.

Run inside the `gpr` conda env:

    conda activate gpr
    python run_invert_ablation.py                  # default: pretrained yolov8n.pt
    python run_invert_ablation.py --scratch         # yolov8n.yaml, random init, no ImageNet weights
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

SEEDS = [0, 1, 2, 3, 4]
SCRATCH_MODEL = "yolov8n.yaml"  # architecture only, random init (no ImageNet pretraining)


def run_one(invert_aug, seed, model, work_root):
    work_dir = os.path.join(work_root, f"invert_{'on' if invert_aug else 'off'}_seed{seed}")
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
        **{name: default for name, (default, _help) in AUG_PARAMS.items()},
    )
    _best, map50, map5095 = train(args, cleanup_dataset=True)
    return float(map50), float(map5095)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", action="store_true",
                    help="Train from yolov8n.yaml (random init) instead of pretrained yolov8n.pt.")
    args = p.parse_args()

    model = SCRATCH_MODEL if args.scratch else MODEL
    work_root = os.path.join(WORK_DIR, "invert_ablation_scratch" if args.scratch else "invert_ablation")
    results_csv = os.path.join(work_root, "results.csv")

    os.makedirs(work_root, exist_ok=True)
    rows = []
    for invert_aug in (True, False):
        for seed in SEEDS:
            print(f"\n=== model={model} invert_aug={invert_aug} seed={seed} ===")
            map50, map5095 = run_one(invert_aug, seed, model, work_root)
            rows.append({"invert_aug": invert_aug, "seed": seed, "map50": map50, "map50_95": map5095})
            with open(results_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["invert_aug", "seed", "map50", "map50_95"])
                writer.writeheader()
                writer.writerows(rows)

    print("\n=== summary ===")
    for invert_aug in (True, False):
        group = [r for r in rows if r["invert_aug"] == invert_aug]
        map50s = [r["map50"] for r in group]
        map5095s = [r["map50_95"] for r in group]
        label = "invert_aug=ON " if invert_aug else "invert_aug=OFF"
        print(f"{label}  mAP50={statistics.mean(map50s):.4f}+/-{statistics.pstdev(map50s):.4f}  "
              f"mAP50-95={statistics.mean(map5095s):.4f}+/-{statistics.pstdev(map5095s):.4f}")
    print(f"\n[ablation] per-run results -> {results_csv}")


if __name__ == "__main__":
    main()
