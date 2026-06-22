"""
用与你的方法一致的置信度阈值(conf=0.30 ≈ HM_THRESH)重新评估已训练好的 YOLO，
而不是 ultralytics 默认的 conf=0.001。不重新训练，只加载 best.pt 重新算 bbox_recall / mAP50。
"""
import os
import csv
import glob
import json
import importlib.util

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("yc", os.path.join(_HERE, "yolo_common.py"))
yc = importlib.util.module_from_spec(spec); spec.loader.exec_module(yc)

from ultralytics import YOLO

CONF = 0.30   # 与你的方法 HM_THRESH 对齐（原来是 0.001，对 YOLO 过于宽松）


def latest_workdir(tag):
    for d in sorted(glob.glob(os.path.join(_HERE, tag + "_*")), reverse=True):
        if os.path.exists(os.path.join(d, "seed00", "weights", "best.pt")):
            return d
    return None


def main():
    yc.CONF_EVAL = CONF   # evaluate_yolo 读模块级 CONF_EVAL
    names = yc._image_names()
    n_total = len(names)
    img_paths = [os.path.join(yc.IMG_DIR, n) for n in names]
    with open(yc.RECT_JSON, "r", encoding="utf-8") as f:
        rect = json.load(f)

    print(f"重新评估 YOLO，conf={CONF}（原 0.001）；IoU>=0.5\n")
    for tag in ["yolo_scratch", "yolo_pretrained"]:
        work = latest_workdir(tag)
        if work is None:
            print(f"[{tag}] 找不到已训练模型，跳过"); continue
        print(f"=== {tag}  ({os.path.basename(work)}) ===")
        rows = []
        for seed in yc.SEEDS:
            best = os.path.join(work, f"seed{seed:02d}", "weights", "best.pt")
            _, _, test_idx = yc.exp.make_split(n_total, seed)
            test_paths = [img_paths[i] for i in test_idx]
            recall, map50, n_gt = yc.evaluate_yolo(YOLO(best), test_paths, rect)
            print(f"   seed {seed}  n_gt={n_gt}  recall={recall:.4f}  mAP50={map50:.4f}")
            rows.append({"seed": seed, "n_gt": n_gt, "bbox_recall": recall, "bbox_mAP50": map50})
        rec = [r["bbox_recall"] for r in rows]; mp = [r["bbox_mAP50"] for r in rows]
        print(f"   MEAN recall={np.mean(rec):.4f}±{np.std(rec):.4f}  mAP50={np.mean(mp):.4f}±{np.std(mp):.4f}")
        out = os.path.join(work, f"{tag}_results_conf{CONF}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"   saved -> {out}\n")

    print("对照（你的方法，conf 口径 HM_THRESH=0.30）：bbox_recall=0.7833  bbox_mAP50=0.6281")


if __name__ == "__main__":
    main()
