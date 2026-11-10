"""
和 simple_bbox_cnn 完全同口径的 YOLO baseline。

公平对齐：
  - 同 315 张图、同 make_split(70/15/15)、同 5 seed、同 bbox IoU>=0.5 评估。
  - 输出 P / R / F1 + mAP50，可与 simple_bbox_cnn 直接并排对比。
复用 yolo_common 的数据准备 / split 写出 / torch2.10 BytesIO 补丁。

用法：
  python yolo_same.py              # from scratch (yolov8n.yaml)
  python yolo_same.py yolov8n.pt   # COCO 预训练

注：mAP50 与阈值无关、最可比；P/R/F1 在固定 conf 下算（YOLO 与 CNN 的 score 尺度不同，
F1 仅作参考，正式对比以 mAP50 为主）。
"""
import os
import sys
import csv
from datetime import datetime

import numpy as np

import yolo_common as yc

exp     = yc.exp
CONF_F1 = 0.25   # 算 P/R/F1 的置信阈值（mAP50 不依赖它）


def evaluate(model, test_paths, rect):
    n_gt = 0
    TP = FP = FN = 0
    ap_tp, ap_sc = [], []
    for p in test_paths:
        name = os.path.basename(p)
        res = model.predict(p, imgsz=yc.IMGSZ, conf=yc.CONF_EVAL, max_det=yc.MAX_DET, verbose=False)
        b = res[0].boxes
        boxes = b.xyxy.cpu().numpy().tolist() if len(b) else []
        confs = b.conf.cpu().numpy().tolist() if len(b) else []
        gt = [[r["x1"], r["y1"], r["x1"] + r["width"], r["y1"] + r["height"]]
              for r in rect.get(name, []) if r.get("label", "") == "hyperbola"]
        n_gt += len(gt)

        # mAP50：用全部预测（conf=CONF_EVAL 已取全）
        matched_all = [False] * len(gt)
        for box, sc in sorted(zip(boxes, confs), key=lambda z: -z[1]):
            bi, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if matched_all[j]:
                    continue
                iou = yc.bbox_iou(box, gb)
                if iou > bi:
                    bi, bj = iou, j
            tp = bi >= yc.BBOX_IOU_THR and bj >= 0
            if tp:
                matched_all[bj] = True
            ap_tp.append(tp); ap_sc.append(sc)

        # P/R/F1 @ CONF_F1
        keep = [(box, sc) for box, sc in zip(boxes, confs) if sc >= CONF_F1]
        matched = [False] * len(gt)
        for box, sc in sorted(keep, key=lambda z: -z[1]):
            bi, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if matched[j]:
                    continue
                iou = yc.bbox_iou(box, gb)
                if iou > bi:
                    bi, bj = iou, j
            if bi >= yc.BBOX_IOU_THR and bj >= 0:
                matched[bj] = True; TP += 1
            else:
                FP += 1
        FN += len(gt) - sum(matched)

    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
    return {"bbox_P": P, "bbox_R": R, "bbox_F1": 2 * P * R / max(P + R, 1e-9),
            "mAP50": exp.compute_ap50(ap_tp, ap_sc, n_gt)}


def main():
    model_spec = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.yaml"
    tag = "yolo_scratch_same" if model_spec.endswith(".yaml") else "yolo_pretrained_same"

    from ultralytics import YOLO
    yc._patch_torch_save_for_bytesio()

    work = os.path.join(os.getcwd(), f"{tag}_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    names, paths, rect = yc.prepare_yolo_data(work)
    n_total = len(names)
    print(f"[{tag}]  n_total={n_total}  model={model_spec}  70/15/15  seeds={yc.SEEDS}")

    rows = []
    for seed in yc.SEEDS:
        tr, va, te = exp.make_split(n_total, seed, train_frac=0.70, val_frac=0.15)
        yaml = yc._write_split(work, seed, paths, tr, va)
        print(f"\n--- seed {seed}  train={len(tr)} val={len(va)} test={len(te)} ---")
        model = YOLO(model_spec)
        model.train(data=yaml, epochs=yc.num_epochs, imgsz=yc.IMGSZ, batch=yc.batch_size,
                    seed=seed, project=work, name=f"seed{seed:02d}", exist_ok=True,
                    verbose=False, workers=2)
        best = os.path.join(work, f"seed{seed:02d}", "weights", "best.pt")
        m = evaluate(YOLO(best), [paths[i] for i in te], rect)
        print(f"  seed{seed}: P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} F1={m['bbox_F1']:.4f} mAP50={m['mAP50']:.4f}")
        row = {"seed": seed}; row.update(m); rows.append(row)

    keys = ["bbox_P", "bbox_R", "bbox_F1", "mAP50"]
    print("\n" + "=" * 60)
    print(f"{'metric':>14}{'mean':>12}{'std':>12}")
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"{k:>14}{np.mean(vals):>12.4f}{np.std(vals):>12.4f}")
    print("=" * 60)

    with open(os.path.join(work, f"{tag}_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {work}")


if __name__ == "__main__":
    main()
