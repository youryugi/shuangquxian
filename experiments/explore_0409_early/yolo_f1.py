"""
算 YOLO 的检测级 precision / recall / F1（基于 bbox IoU>=0.5 匹配），conf 与你的方法对齐(0.30)。
加载已训练 best.pt，不重新训练。
"""
import os
import glob
import json
import importlib.util

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("yc", os.path.join(_HERE, "yolo_common.py"))
yc = importlib.util.module_from_spec(spec); spec.loader.exec_module(yc)
from ultralytics import YOLO

CONF = 0.30
IOU_THR = 0.5


def latest_workdir(tag):
    for d in sorted(glob.glob(os.path.join(_HERE, tag + "_*")), reverse=True):
        if os.path.exists(os.path.join(d, "seed00", "weights", "best.pt")):
            return d
    return None


def eval_prf(model, test_paths, rect):
    TP = FP = FN = 0
    for p in test_paths:
        name = os.path.basename(p)
        res = model.predict(p, imgsz=yc.IMGSZ, conf=CONF, max_det=yc.MAX_DET, verbose=False)
        b = res[0].boxes
        boxes = b.xyxy.cpu().numpy().tolist() if len(b) else []
        confs = b.conf.cpu().numpy().tolist() if len(b) else []
        gt = [[r["x1"], r["y1"], r["x1"] + r["width"], r["y1"] + r["height"]]
              for r in rect.get(name, []) if r.get("label", "") == "hyperbola"]
        matched = [False] * len(gt)
        for box, _ in sorted(zip(boxes, confs), key=lambda z: -z[1]):
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt):
                if matched[j]:
                    continue
                iou = yc.bbox_iou(box, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= IOU_THR and best_j >= 0:
                matched[best_j] = True; TP += 1
            else:
                FP += 1
        FN += len(gt) - sum(matched)
    P = TP / max(TP + FP, 1e-9)
    R = TP / max(TP + FN, 1e-9)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return P, R, F1


def main():
    names = yc._image_names()
    n_total = len(names)
    img_paths = [os.path.join(yc.IMG_DIR, n) for n in names]
    with open(yc.RECT_JSON, "r", encoding="utf-8") as f:
        rect = json.load(f)

    print(f"YOLO 检测级 P/R/F1  (conf={CONF}, IoU>={IOU_THR})\n")
    for tag in ["yolo_scratch", "yolo_pretrained"]:
        work = latest_workdir(tag)
        if work is None:
            print(f"[{tag}] 找不到模型，跳过"); continue
        print(f"=== {tag} ===")
        Ps, Rs, Fs = [], [], []
        for seed in yc.SEEDS:
            best = os.path.join(work, f"seed{seed:02d}", "weights", "best.pt")
            _, _, test_idx = yc.exp.make_split(n_total, seed)
            P, R, F1 = eval_prf(YOLO(best), [img_paths[i] for i in test_idx], rect)
            print(f"   seed {seed}  P={P:.4f}  R={R:.4f}  F1={F1:.4f}")
            Ps.append(P); Rs.append(R); Fs.append(F1)
        print(f"   MEAN  P={np.mean(Ps):.4f}±{np.std(Ps):.4f}  "
              f"R={np.mean(Rs):.4f}±{np.std(Rs):.4f}  F1={np.mean(Fs):.4f}±{np.std(Fs):.4f}\n")


if __name__ == "__main__":
    main()
