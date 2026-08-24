"""
Cross-architecture reproduction of the invert_aug (true polarity-inversion
augmentation) ablation: same clean protocol as run_invert_ablation.py
(all built-in ultralytics augmentations off, only INVERT_AUG varies) but
swept over several *different model architectures*, all trained from a
random initialization -- no ImageNet/COCO pretrained weights -- via each
model's .yaml architecture config instead of a .pt checkpoint. This
matches the earlier finding that the invert_aug effect is largest and
statistically significant specifically in the no-pretraining regime, so
that's the regime used here to check whether the effect generalizes
across architectures.

Models covered (all ultralytics-API compatible: YOLO or RTDETR classes,
both expose .train()/.val() with the same argument names):
  - yolov8n  (YOLO,   yolov8n.yaml)   -- already covered by
    run_invert_ablation.py's scratch mode; included here too so all
    cross-architecture results live in one place.
  - yolov8s  (YOLO,   yolov8s.yaml)   -- larger CNN, same generation.
  - yolo11n  (YOLO,   yolo11n.yaml)   -- newer YOLO generation/backbone.
  - rtdetr-l (RTDETR, rtdetr-l.yaml)  -- transformer-decoder DETR-style
    detector, architecturally unrelated to the YOLO CNN family. Note:
    RT-DETR is known to converge slowly without pretraining and on very
    small datasets (186 train images here) -- treat its from-scratch
    numbers with extra caution, they may simply reflect under-convergence
    rather than an architecture-specific augmentation effect.

Trains with INVERT_AUG on and off, 5 random seeds each, per model
(4 models x 2 conditions x 5 seeds = 40 runs total). Saves one results
CSV per model plus a combined CSV, and prints a summary table.

Run inside the `gpr` conda env:

    conda activate gpr
    python run_invert_ablation_multi_model.py
"""
import csv
import os
import statistics
from types import SimpleNamespace

from train_yolo_bbox import (
    AUG_PARAMS, DATA_DIR, WORK_DIR, EPOCHS, IMGSZ, BATCH, VAL_RATIO,
    POLARITY_DIR, INVERT_DIR, train,
)

SEEDS = [0, 1, 2, 3, 4]
WORK_ROOT = os.path.join(WORK_DIR, "invert_ablation_multi_model")

# name -> (ultralytics class name, architecture yaml). Class is resolved
# lazily in run_one() so importing this module doesn't require RTDETR.
MODELS = {
    "yolov8n":  ("YOLO",   "yolov8n.yaml"),
    "yolov8s":  ("YOLO",   "yolov8s.yaml"),
    "yolo11n":  ("YOLO",   "yolo11n.yaml"),
    "rtdetr-l": ("RTDETR", "rtdetr-l.yaml"),
}


def resolve_model_cls(class_name):
    import ultralytics
    return getattr(ultralytics, class_name)


def run_one(model_name, invert_aug, seed):
    class_name, model_yaml = MODELS[model_name]
    model_cls = resolve_model_cls(class_name)
    work_dir = os.path.join(WORK_ROOT, model_name, f"invert_{'on' if invert_aug else 'off'}_seed{seed}")
    args = SimpleNamespace(
        data_dir=DATA_DIR,
        work_dir=work_dir,
        model=model_yaml,
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
    _best, map50, map5095 = train(args, model_cls=model_cls)
    return float(map50), float(map5095)


def main():
    os.makedirs(WORK_ROOT, exist_ok=True)
    all_rows = []
    for model_name in MODELS:
        model_root = os.path.join(WORK_ROOT, model_name)
        os.makedirs(model_root, exist_ok=True)
        results_csv = os.path.join(model_root, "results.csv")
        rows = []
        for invert_aug in (True, False):
            for seed in SEEDS:
                print(f"\n=== [{model_name}, scratch] invert_aug={invert_aug} seed={seed} ===")
                map50, map5095 = run_one(model_name, invert_aug, seed)
                row = {"model": model_name, "invert_aug": invert_aug, "seed": seed,
                       "map50": map50, "map50_95": map5095}
                rows.append(row)
                all_rows.append(row)
                with open(results_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["invert_aug", "seed", "map50", "map50_95"])
                    writer.writeheader()
                    writer.writerows({k: v for k, v in r.items() if k != "model"} for r in rows)
        with open(os.path.join(WORK_ROOT, "results_all.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "invert_aug", "seed", "map50", "map50_95"])
            writer.writeheader()
            writer.writerows(all_rows)

    print("\n=== summary (all models, scratch, all built-in augs off) ===")
    for model_name in MODELS:
        group_all = [r for r in all_rows if r["model"] == model_name]
        for invert_aug in (True, False):
            group = [r for r in group_all if r["invert_aug"] == invert_aug]
            map50s = [r["map50"] for r in group]
            map5095s = [r["map50_95"] for r in group]
            label = "ON " if invert_aug else "OFF"
            print(f"{model_name:>9s}  invert_aug={label}  "
                  f"mAP50={statistics.mean(map50s):.4f}+/-{statistics.pstdev(map50s):.4f}  "
                  f"mAP50-95={statistics.mean(map5095s):.4f}+/-{statistics.pstdev(map5095s):.4f}")
    print(f"\n[ablation] per-model results -> {WORK_ROOT}/<model>/results.csv")
    print(f"[ablation] combined results   -> {os.path.join(WORK_ROOT, 'results_all.csv')}")


if __name__ == "__main__":
    main()
