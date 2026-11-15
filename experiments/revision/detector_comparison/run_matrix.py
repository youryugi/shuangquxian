"""Run the whole experiment matrix sequentially (skips runs that already have metrics.json).

    python run_matrix.py --datasets ds2 ds1 --seeds 0 1 2 --exps yolov8s_pre yolov8s_pre_band        # pilot
    python run_matrix.py --datasets ds2 ds1 --seeds 0-9                                             # everything
    python run_matrix.py --smoke                                                                     # 2-epoch sanity run
    python run_matrix.py --list                                                                      # show the matrix

Experiment names (--exps):
    yolov8s_pre / yolov8s_scratch          YOLOv8s, COCO-pretrained / random init
    yolov8s_pre_band / yolov8s_scratch_band   + hyperbola-band auxiliary supervision (the plug-in)
    yolov8s_pre_boxmask                    ablation: same aux head, filled official boxes instead of bands
    yolo11s_pre                            YOLO11s (optional extra row)
    rtdetr_l_pre                           RT-DETR-L (transformer detector)
    frcnn_r50_pre / frcnn_r50_scratch      Faster R-CNN ResNet-50-FPN v2
Band-loss weight scan: yolov8s_pre_band_g0.5 / _g2 / _g5 (gain 0.5 / 2 / 5).
The dataset directory must already exist (prepare_data.py); pass --data-suffix to pick fold/subset.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

EXPS = {
    "yolov8s_pre":            dict(script="train_ultralytics.py", args=["--model", "yolov8s.pt"]),
    "yolov8s_scratch":        dict(script="train_ultralytics.py", args=["--model", "yolov8s.yaml"]),
    "yolov8s_pre_band":       dict(script="train_ultralytics.py", args=["--model", "yolov8s.pt", "--band"]),
    "yolov8s_scratch_band":   dict(script="train_ultralytics.py", args=["--model", "yolov8s.yaml", "--band"]),
    "yolov8s_pre_boxmask":    dict(script="train_ultralytics.py", args=["--model", "yolov8s.pt", "--band", "--mask-dir", "masks_box"]),
    "yolov8s_pre_band_g0.5":  dict(script="train_ultralytics.py", args=["--model", "yolov8s.pt", "--band", "--band-gain", "0.5"]),
    "yolov8s_pre_band_g2":    dict(script="train_ultralytics.py", args=["--model", "yolov8s.pt", "--band", "--band-gain", "2"]),
    "yolov8s_pre_band_g5":    dict(script="train_ultralytics.py", args=["--model", "yolov8s.pt", "--band", "--band-gain", "5"]),
    "yolov8n_pre":            dict(script="train_ultralytics.py", args=["--model", "yolov8n.pt"]),
    "yolov8n_pre_band":       dict(script="train_ultralytics.py", args=["--model", "yolov8n.pt", "--band"]),
    "yolo11s_pre":            dict(script="train_ultralytics.py", args=["--model", "yolo11s.pt"]),
    "rtdetr_l_pre":           dict(script="train_ultralytics.py", args=["--model", "rtdetr-l.pt", "--batch", "8"]),
    "frcnn_r50_pre":          dict(script="train_frcnn.py", args=[]),
    "frcnn_r50_scratch":      dict(script="train_frcnn.py", args=["--scratch"]),
}
DEFAULT_EXPS = ["yolov8s_pre", "yolov8s_pre_band", "yolov8s_scratch", "yolov8s_scratch_band",
                "yolov8s_pre_boxmask", "frcnn_r50_pre", "rtdetr_l_pre"]


def parse_seeds(tokens):
    seeds = []
    for t in tokens:
        if "-" in t:
            a, b = t.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(t))
    return seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["ds2", "ds1"])
    ap.add_argument("--seeds", nargs="+", default=["0-9"])
    ap.add_argument("--exps", nargs="+", default=DEFAULT_EXPS)
    ap.add_argument("--data-suffix", default="f0_s100", help="datasets/<ds>_<suffix>")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for every run")
    ap.add_argument("--extra", nargs="*", default=[], help="extra CLI args appended to every run")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, seed 0, ds2, yolov8n baseline + band, runs_smoke/")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.datasets, args.seeds, args.epochs = ["ds2"], ["0"], 2
        args.exps = ["yolov8n_pre", "yolov8n_pre_band"]
        args.runs = str(HERE / "runs_smoke")
    seeds = parse_seeds(args.seeds)
    runs_root = Path(args.runs)
    log = runs_root / "run_matrix.log"

    jobs = []
    for ds in args.datasets:
        data_dir = HERE / "datasets" / f"{ds}_{args.data_suffix}"
        for exp in args.exps:
            if exp not in EXPS:
                raise SystemExit(f"unknown exp {exp}; known: {list(EXPS)}")
            for seed in seeds:
                run_dir = runs_root / ds / exp / f"seed{seed}"
                done = (run_dir / "metrics.json").exists()
                cmd = [PY, "-X", "utf8", str(HERE / EXPS[exp]["script"]), "--data", str(data_dir), "--exp", exp,
                       "--seed", str(seed), "--runs", str(runs_root), *EXPS[exp]["args"]]
                if args.epochs is not None:
                    cmd += ["--epochs", str(args.epochs)]
                cmd += args.extra
                jobs.append((ds, exp, seed, cmd, done, data_dir))

    print(f"{len(jobs)} runs, {sum(j[4] for j in jobs)} already done")
    if args.list or args.dry_run:
        for ds, exp, seed, cmd, done, _ in jobs:
            print(("DONE " if done else "TODO ") + f"{ds}/{exp}/seed{seed}: " + " ".join(cmd[3:]))
        return
    runs_root.mkdir(parents=True, exist_ok=True)
    for ds, exp, seed, cmd, done, data_dir in jobs:
        if done:
            continue
        if not data_dir.exists():
            raise SystemExit(f"dataset dir missing: {data_dir} (run prepare_data.py first)")
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {ds}/{exp}/seed{seed}\n  {' '.join(cmd)}\n")
        rc = subprocess.call(cmd, cwd=str(HERE))
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END   {ds}/{exp}/seed{seed} rc={rc} "
                    f"({(time.time() - t0) / 60:.1f} min)\n")
        if rc != 0:
            print(f"!! {ds}/{exp}/seed{seed} failed with rc={rc}; see {log}")
    # summary
    from evaluate import summary
    summary(runs_root)


if __name__ == "__main__":
    main()
