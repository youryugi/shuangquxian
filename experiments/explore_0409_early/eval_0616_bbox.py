"""
把"跑了很久"的 0616 HyperbolaNet 推理出的双曲线带转成外接 bbox，
算 bbox 级 Precision / Recall / F1 @IoU0.5 + mAP50（与 simple_bbox_cnn 同指标定义）。

重要：这 5 个权重是 50/25/25（TRAIN_FRAC=0.5）训练的，所以只能在其【对应的 50/25/25
test】上评估，否则训练见过的图会泄露进 test。要和 70/15/15 的 CNN/YOLO 同 split 对比，
需重训（见结尾提示）。

口径与 simple_bbox_cnn 完全一致：
  - 同 315 图、同 make_split（这里用 50/25/25 默认，因为权重就是这么训的）
  - GT bbox 来自 annotations_rect.json
  - P/R/F1 在固定 conf=HM_THRESH(0.30) 下；mAP50 扫 score、阈值无关
"""
import os
import re
import json
import csv
import importlib.util

import numpy as np
from PIL import Image
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(f, n):
    spec = importlib.util.spec_from_file_location(n, os.path.join(_HERE, f))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
exp = _load("0616-1.py", "exp0616")

device     = exp.device
input_size = exp.input_size
CKPT_DIR   = "0616-1_0617_0014"
IMG_DIR    = exp.data_sources[0]["image_dir"]
HYP_JSON   = exp.data_sources[0]["annotation_json"]
RECT_JSON  = os.path.join(os.path.dirname(HYP_JSON), "annotations_rect.json")
IOU_THR    = 0.5
CONF_F1    = exp.HM_THRESH   # 0.30，固定阈值算 P/R/F1


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def hyp_to_bbox(d):
    """双曲线参数(640 尺度) -> 外接矩形框。与 attn_cnn.hyperbola_to_bbox 同公式。"""
    hw = d["width"] / 2.0
    return [d["x_vertex"] - hw, d["y_vertex"] - d["thickness"] / 2.0,
            d["x_vertex"] + hw, d["y_vertex"] + d["height"] + d["thickness"] / 2.0]


def image_names():
    with open(HYP_JSON, "r", encoding="utf-8") as f:
        hyp = json.load(f)
    return sorted(hyp.keys(), key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0])


def compute_ap50(ap_tp, ap_sc, n_gt):
    if n_gt == 0 or not ap_tp:
        return 0.0
    order = np.argsort(ap_sc)[::-1]
    tp = np.array(ap_tp, dtype=np.float32)[order]
    fp = 1.0 - tp
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def eval_seed(model, names, test_idx, rect):
    H, W = input_size
    TP = FP = FN = 0
    n_gt = 0
    ap_tp, ap_sc = [], []
    for i in test_idx:
        name = names[i]
        path = os.path.join(IMG_DIR, name)
        ow, oh = Image.open(path).size
        sx, sy = W / ow, H / oh
        gt = [[r["x1"] * sx, r["y1"] * sy, (r["x1"] + r["width"]) * sx, (r["y1"] + r["height"]) * sy]
              for r in rect.get(name, []) if r.get("label", "") == "hyperbola"]
        n_gt += len(gt)

        _, _, dets = exp.predict_single_image(
            model, path, input_size, device,
            obj_thresh=0.01, nms_k=exp.nms_kernel, max_det=exp.max_det)   # 低阈值取全部，给 mAP
        preds = [(hyp_to_bbox(d), d["score"]) for d in dets]

        # mAP50：全部预测
        matched_all = [False] * len(gt)
        for box, sc in sorted(preds, key=lambda z: -z[1]):
            bi, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if matched_all[j]:
                    continue
                iou = bbox_iou(box, gb)
                if iou > bi:
                    bi, bj = iou, j
            tp = bi >= IOU_THR and bj >= 0
            if tp:
                matched_all[bj] = True
            ap_tp.append(tp); ap_sc.append(sc)

        # P/R/F1 @ CONF_F1
        keep = [(box, sc) for box, sc in preds if sc >= CONF_F1]
        matched = [False] * len(gt)
        for box, sc in sorted(keep, key=lambda z: -z[1]):
            bi, bj = 0.0, -1
            for j, gb in enumerate(gt):
                if matched[j]:
                    continue
                iou = bbox_iou(box, gb)
                if iou > bi:
                    bi, bj = iou, j
            if bi >= IOU_THR and bj >= 0:
                matched[bj] = True; TP += 1
            else:
                FP += 1
        FN += len(gt) - sum(matched)

    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
    return {"bbox_P": P, "bbox_R": R, "bbox_F1": 2 * P * R / max(P + R, 1e-9),
            "mAP50": compute_ap50(ap_tp, ap_sc, n_gt)}


def main():
    names = image_names()
    n = len(names)
    with open(RECT_JSON, "r", encoding="utf-8") as f:
        rect = json.load(f)
    print(f"[0616 HyperbolaNet -> bbox]  n_total={n}  (权重为 50/25/25 训练，于对应 test 评估)")

    rows = []
    for seed in exp.SEEDS:
        tr, va, te = exp.make_split(n, seed)   # 默认 50/25/25 = 训练时 split
        ckpt = os.path.join(CKPT_DIR, f"seed{seed:02d}", "checkpoints", "best_model.pth")
        if not os.path.exists(ckpt):
            print(f"  seed{seed}: 缺权重 {ckpt}，跳过"); continue
        model = exp.HyperbolaNet(in_ch=1, base_ch=32).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
        m = eval_seed(model, names, te, rect)
        print(f"  seed{seed} (test={len(te)}): P={m['bbox_P']:.4f} R={m['bbox_R']:.4f} "
              f"F1={m['bbox_F1']:.4f} mAP50={m['mAP50']:.4f}")
        row = {"seed": seed}; row.update(m); rows.append(row)

    keys = ["bbox_P", "bbox_R", "bbox_F1", "mAP50"]
    print("\n" + "=" * 60)
    print(f"{'metric':>14}{'mean':>12}{'std':>12}")
    for k in keys:
        vals = [r[k] for r in rows]
        print(f"{k:>14}{np.mean(vals):>12.4f}{np.std(vals):>12.4f}")
    print("=" * 60)

    with open(os.path.join(CKPT_DIR, "bbox_eval_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {os.path.join(CKPT_DIR, 'bbox_eval_results.csv')}")


if __name__ == "__main__":
    main()
