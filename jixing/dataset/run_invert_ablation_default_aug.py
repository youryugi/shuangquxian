"""
Ablation: with ultralytics' own default built-in augmentations turned ON
(hsv jitter, translate/scale, fliplr, mosaic -- the "normal" YOLO training
recipe, not the clean all-off baseline used by run_invert_ablation.py),
does adding the true-polarity-inversion augmentation (INVERT_AUG) still
help?

Trains with INVERT_AUG on and off, 5 random seeds each (10 runs total),
using the pretrained yolov8n.pt backbone, and prints/saves mAP50 /
mAP50-95 per run plus the mean +/- std for each condition.

Run inside the `gpr` conda env:

    conda activate gpr
    python run_invert_ablation_default_aug.py             # pretrained yolov8n.pt
    python run_invert_ablation_default_aug.py --scratch    # yolov8n.yaml, random init
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

SEEDS = [0, 1, 2, 3, 4]
SCRATCH_MODEL = "yolov8n.yaml"

# Ultralytics' own out-of-the-box defaults (see the "ultralytics default ..."
# notes in train_yolo_bbox.AUG_PARAMS). flipud is kept 0 -- vertically
# flipping a radargram flips the depth axis, which isn't a valid augmentation
# for this domain -- but that's also ultralytics' own default, so no
# compromise is needed there.
YOLO_DEFAULT_AUG = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


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
        **YOLO_DEFAULT_AUG,
    )
    _best, map50, map5095 = train(args, cleanup_dataset=True)
    return float(map50), float(map5095)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", action="store_true",
                    help="Train from yolov8n.yaml (random init) instead of pretrained yolov8n.pt.")
    args = p.parse_args()

    model = SCRATCH_MODEL if args.scratch else MODEL
    work_root = os.path.join(WORK_DIR, "invert_ablation_default_aug_scratch" if args.scratch else "invert_ablation_default_aug")
    results_csv = os.path.join(work_root, "results.csv")

    os.makedirs(work_root, exist_ok=True)
    rows = []
    for invert_aug in (True, False):
        for seed in SEEDS:
            print(f"\n=== [default-aug, model={model}] invert_aug={invert_aug} seed={seed} ===")
            map50, map5095 = run_one(invert_aug, seed, model, work_root)
            rows.append({"invert_aug": invert_aug, "seed": seed, "map50": map50, "map50_95": map5095})
            with open(results_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["invert_aug", "seed", "map50", "map50_95"])
                writer.writeheader()
                writer.writerows(rows)

    print("\n=== summary (default ultralytics augmentations ON) ===")
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
