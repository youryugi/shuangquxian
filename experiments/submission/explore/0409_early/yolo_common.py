"""
YOLO baseline 共享逻辑（供 yolo_scratch.py / yolo_pretrained.py 调用）。

公平性保证（与你的方法 / 朴素 CNN baseline 完全一致）：
  - 相同 315 张图（annotations.json 的 keys，相同排序）
  - 相同 5-seed 划分（复用 0616-1.py 的 make_split）
  - 相同 bbox 评估口径（bbox_iou + compute_ap50，IoU>=0.5）
GT 框来自 annotations_rect.json。

依赖：pip install ultralytics
"""
import os
import re
import csv
import json
import shutil
import importlib.util
from datetime import datetime

import numpy as np
from PIL import Image

# ── 复用 0616-1.py（make_split / compute_ap50 / 超参）──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
def _load_exp(path):
    spec = importlib.util.spec_from_file_location("exp0616", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
exp = _load_exp(os.path.join(_HERE, "0616-1.py"))

SEEDS        = exp.SEEDS
num_epochs   = exp.num_epochs
batch_size   = exp.batch_size
IMGSZ        = exp.input_size[0]
MAX_DET      = exp.max_det
BBOX_IOU_THR = 0.5
CONF_EVAL    = 0.001   # 评估时用低 conf 取全部预测，再贪心匹配（检出率口径）

IMG_DIR   = exp.data_sources[0]["image_dir"]
HYP_JSON  = exp.data_sources[0]["annotation_json"]
RECT_JSON = os.path.join(os.path.dirname(HYP_JSON), "annotations_rect.json")


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter  = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _image_names():
    """与 0616-1 的 HyperbolaDataset 完全相同的 315 张 + 排序，保证 make_split 对齐。"""
    with open(HYP_JSON, "r", encoding="utf-8") as f:
        hyp = json.load(f)
    return sorted(hyp.keys(), key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0])


def prepare_yolo_data(work):
    """图片 + YOLO 格式标注只准备一次：work/images/all、work/labels/all。"""
    with open(RECT_JSON, "r", encoding="utf-8") as f:
        rect = json.load(f)
    names = _image_names()
    img_all = os.path.join(work, "images", "all")
    lab_all = os.path.join(work, "labels", "all")
    os.makedirs(img_all, exist_ok=True)
    os.makedirs(lab_all, exist_ok=True)

    paths = []
    for name in names:
        src = os.path.join(IMG_DIR, name)
        dst = os.path.join(img_all, name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        paths.append(dst)
        w, h = Image.open(src).size
        lines = []
        for r in rect.get(name, []):
            if r.get("label", "") != "hyperbola":
                continue
            cx = (r["x1"] + r["width"]  / 2.0) / w
            cy = (r["y1"] + r["height"] / 2.0) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {r['width']/w:.6f} {r['height']/h:.6f}")
        with open(os.path.join(lab_all, os.path.splitext(name)[0] + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return names, paths, rect


def _write_split(work, seed, paths, train_idx, val_idx):
    tl = os.path.join(work, f"seed{seed:02d}_train.txt")
    vl = os.path.join(work, f"seed{seed:02d}_val.txt")
    with open(tl, "w", encoding="utf-8") as f:
        f.write("\n".join(paths[i] for i in train_idx))
    with open(vl, "w", encoding="utf-8") as f:
        f.write("\n".join(paths[i] for i in val_idx))
    yaml = os.path.join(work, f"seed{seed:02d}.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {work}\ntrain: {tl}\nval: {vl}\nnames:\n  0: hyperbola\n")
    return yaml


def evaluate_yolo(model, test_paths, rect):
    """与你的方法 / 朴素 CNN 相同口径：bbox IoU>=0.5 -> recall / mAP50。"""
    n_gt = n_det = 0
    ap_tp, ap_sc = [], []
    for p in test_paths:
        name = os.path.basename(p)
        res = model.predict(p, imgsz=IMGSZ, conf=CONF_EVAL, max_det=MAX_DET, verbose=False)
        b = res[0].boxes
        boxes = b.xyxy.cpu().numpy().tolist() if len(b) else []
        confs = b.conf.cpu().numpy().tolist() if len(b) else []
        gt = [[r["x1"], r["y1"], r["x1"] + r["width"], r["y1"] + r["height"]]
              for r in rect.get(name, []) if r.get("label", "") == "hyperbola"]
        n_gt += len(gt)
        matched = [False] * len(gt)
        for box, sc in sorted(zip(boxes, confs), key=lambda z: -z[1]):
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt):
                if matched[j]:
                    continue
                iou = bbox_iou(box, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = best_iou >= BBOX_IOU_THR and best_j >= 0
            if is_tp:
                matched[best_j] = True
            ap_tp.append(is_tp); ap_sc.append(sc)
        n_det += sum(matched)
    return n_det / max(n_gt, 1), exp.compute_ap50(ap_tp, ap_sc, n_gt), n_gt


def _patch_torch_save_for_bytesio():
    """规避 torch 2.10 + BytesIO 的间歇性 'I/O operation on closed file'：
    ultralytics 用 io.BytesIO 序列化 checkpoint（trainer.save_model），在 16GB 内存下偶发失败。
    目标是 BytesIO 时改用临时文件中转（文件路径 save 稳定，且流式写盘更省内存）；其余原样。"""
    import io
    import tempfile
    import torch
    from ultralytics.utils import patches
    orig = patches._torch_save  # ultralytics patch 之前捕获的原始 torch.save

    def safe_save(obj, f=None, *args, **kwargs):
        if isinstance(f, io.BytesIO):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmp.close()
            try:
                orig(obj, tmp.name, *args, **kwargs)
                with open(tmp.name, "rb") as r:
                    f.write(r.read())
            finally:
                os.unlink(tmp.name)
            return
        return orig(obj, f, *args, **kwargs)

    patches.torch_save = safe_save
    torch.save = safe_save


def run_experiment(model_spec, tag):
    """model_spec='yolov8n.yaml' -> from scratch；'yolov8n.pt' -> COCO 预训练。"""
    from ultralytics import YOLO
    _patch_torch_save_for_bytesio()

    now  = datetime.now()
    work = os.path.join(os.getcwd(), f"{tag}_{now.strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    print(f"=== {tag}  model_spec={model_spec}  imgsz={IMGSZ}  epochs={num_epochs} ===")

    names, paths, rect = prepare_yolo_data(work)
    n_total = len(names)
    print(f"Total samples: {n_total}")

    rows = []
    for seed in SEEDS:
        train_idx, val_idx, test_idx = exp.make_split(n_total, seed)
        yaml = _write_split(work, seed, paths, train_idx, val_idx)
        print(f"\n--- Seed {seed}  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} ---")

        model = YOLO(model_spec)
        model.train(data=yaml, epochs=num_epochs, imgsz=IMGSZ, batch=batch_size,
                    seed=seed, project=work, name=f"seed{seed:02d}", exist_ok=True,
                    verbose=True, workers=2)

        best = os.path.join(work, f"seed{seed:02d}", "weights", "best.pt")
        recall, map50, n_gt = evaluate_yolo(YOLO(best), [paths[i] for i in test_idx], rect)
        print(f"  [Seed {seed}] n_gt={n_gt}  bbox_recall={recall:.4f}  bbox_mAP50={map50:.4f}")
        rows.append({"seed": seed, "n_gt": n_gt, "bbox_recall": recall, "bbox_mAP50": map50})

    keys = ["bbox_recall", "bbox_mAP50"]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    stds  = {k: float(np.std ([r[k] for r in rows])) for k in keys}
    print("\n" + "=" * 60)
    print(f"{tag}  {len(rows)}-seed  (IoU>={BBOX_IOU_THR})")
    print("=" * 60)
    for k in keys:
        print(f"  {k:<12} = {means[k]:.4f} ± {stds[k]:.4f}")
    print("=" * 60)

    csv_path = os.path.join(work, f"{tag}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {csv_path}")
    return rows
