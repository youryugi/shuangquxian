"""
小样本对比实验 —— 共享模块。

设定：固定 seed=0；test/val 固定（用 0616-1 的 make_split(0)）；train 从训练池按百分比取。
4 种方法（你的方法 / 朴素CNN / YOLO scratch / YOLO pretrained）用相同的 train 子集、相同 test。

统一评估（4 方法可比，就是之前共识的指标）：
  - bbox 级 P / R / F1（双曲线->外接框；CNN/YOLO 本身是框；IoU>=0.5）
  - vertex_recall（你的方法用预测顶点；CNN/YOLO 用框顶边中点近似顶点；距离<=VERTEX_THRESH）
  - mask_iou（你的方法->双曲线带 mask；CNN/YOLO->实心矩形 mask；全局 IoU），体现 bbox 对薄带的不足
"""
import os
import re
import json
import random
import importlib.util

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)


def load_mod(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_PARENT, fname))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


exp = load_mod("0616-1.py", "exp0616")

SEED          = 0
input_size    = exp.input_size
VERTEX_THRESH = exp.VERTEX_THRESH
BBOX_IOU_THR  = 0.5
IMG_DIR   = exp.data_sources[0]["image_dir"]
HYP_JSON  = exp.data_sources[0]["annotation_json"]
RECT_JSON = os.path.join(os.path.dirname(HYP_JSON), "annotations_rect.json")


def image_names():
    with open(HYP_JSON, "r", encoding="utf-8") as f:
        hyp = json.load(f)
    return sorted(hyp.keys(), key=lambda n: [int(x) for x in re.findall(r'\d+', n)] or [0])


def make_fewshot_split(n_total, train_ratio, seed=SEED):
    """固定 test/val（来自 make_split(seed)），train 从训练池按 train_ratio*n_total 取。"""
    tr, va, te = exp.make_split(n_total, seed)
    pool = list(tr)                      # 训练池（不含 val/test）
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_train = max(1, int(round(n_total * train_ratio)))
    train_idx = pool[:min(n_train, len(pool))]
    return train_idx, va, te


# ── 几何工具 ──────────────────────────────────────────────────────────────────
def hyperbola_to_bbox(o):
    hw = o["width"] / 2.0
    return [o["x_vertex"] - hw, o["y_vertex"] - o["thickness"] / 2.0,
            o["x_vertex"] + hw, o["y_vertex"] + o["height"] + o["thickness"] / 2.0]


def bbox_to_vertex(box):
    """框顶边中点 ≈ 双曲线顶点（开口向下，顶点在带顶部）。"""
    return ((box[0] + box[2]) / 2.0, box[1])


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def solid_bbox_mask(boxes, h, w):
    m = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        xa, ya = max(0, int(round(x1))), max(0, int(round(y1)))
        xb, yb = min(w, int(round(x2))), min(h, int(round(y2)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = 1.0
    return m


# ── 统一评估 ──────────────────────────────────────────────────────────────────
def evaluate_method(preds, metas):
    """
    preds[i] = {"boxes": [[x1,y1,x2,y2],...], "scores": [...],
                "vertices": [(x,y),...], "mask": HxW float}
    metas[i] = 0616-1 的 meta（含 objects/双曲线参数）
    """
    H, W = input_size
    TP = FP = FN = 0
    n_gt = vtx_det = 0
    tot_inter = tot_union = 0.0

    for pred, meta in zip(preds, metas):
        gt_objs  = meta["objects"]
        gt_boxes = [hyperbola_to_bbox(o) for o in gt_objs]
        gt_verts = [(o["x_vertex"], o["y_vertex"]) for o in gt_objs]
        n_gt += len(gt_objs)

        # bbox P/R/F1（按 score 降序贪心匹配）
        order = np.argsort(pred["scores"])[::-1] if pred["scores"] else []
        matched = [False] * len(gt_boxes)
        for k in order:
            pb = pred["boxes"][k]
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if matched[j]:
                    continue
                iou = bbox_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= BBOX_IOU_THR and best_j >= 0:
                matched[best_j] = True; TP += 1
            else:
                FP += 1
        FN += len(gt_boxes) - sum(matched)

        # vertex_recall
        for gv in gt_verts:
            if pred["vertices"]:
                d = min(np.hypot(gv[0] - pv[0], gv[1] - pv[1]) for pv in pred["vertices"])
                if d <= VERTEX_THRESH:
                    vtx_det += 1

        # mask IoU（全局累计）
        gt_mask = exp.build_gt_mask_from_meta(meta, input_size) > 0
        pm = pred["mask"] > 0
        tot_inter += float(np.logical_and(pm, gt_mask).sum())
        tot_union += float(np.logical_or(pm, gt_mask).sum())

    P = TP / max(TP + FP, 1e-9)
    R = TP / max(TP + FN, 1e-9)
    return {
        "bbox_P":  P,
        "bbox_R":  R,
        "bbox_F1": 2 * P * R / max(P + R, 1e-9),
        "vertex_recall": vtx_det / max(n_gt, 1),
        "mask_iou": tot_inter / max(tot_union, 1e-9),
    }
