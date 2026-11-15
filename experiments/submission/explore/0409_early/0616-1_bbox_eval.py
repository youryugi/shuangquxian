"""
对 0616-1.py 训练好的 5 个 seed best_model 做 bbox 检出率评估。

做法：
  - 复用 0616-1.py 的模型 / 数据集 / 划分逻辑（保证 test 集与训练时完全一致）
  - 加载每个 seed 的 best_model.pth
  - 预测双曲线 -> 转成外接 bbox -> 与 annotations_rect.json 的 GT 框比 IoU>=阈值 -> 检出率
  - 同时输出 vertex-based recall 作对照（两种准则的检出率对比，正是论文要的）

bbox 与双曲线参数的精确对应（已验证）：
  x1 = x_vertex - width/2
  y1 = y_vertex - thickness/2
  box_w = width
  box_h = hyperbola_height + thickness
"""
import os
import csv
import glob
import json
import importlib.util

import numpy as np
import torch
from torch.utils.data import Subset

# ── 配置 ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
EXP_FILE     = os.path.join(_HERE, "0616-1.py")   # 复用的训练脚本
WORK_DIR     = None                               # None=自动找含完整 5 个 best_model 的最新目录
BBOX_IOU_THR = 0.5                                # bbox 检出准则
RECT_JSON_NAME = "annotations_rect.json"


# ── 复用 0616-1.py 的全部定义（文件名带连字符，用 importlib 按路径加载）──────────
def load_exp_module(path):
    spec = importlib.util.spec_from_file_location("exp0616", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_work_dir(here):
    """找含完整 5 个 seed best_model 的 0616-1_* 目录（取最新）。"""
    cands = []
    for d in glob.glob(os.path.join(here, "0616-1_*")):
        n = len(glob.glob(os.path.join(d, "seed*", "checkpoints", "best_model.pth")))
        if n >= 1:
            cands.append((os.path.getmtime(d), n, d))
    if not cands:
        raise FileNotFoundError("未找到任何 0616-1_* 训练输出目录")
    cands.sort()
    return cands[-1][2]


# ── bbox 工具 ─────────────────────────────────────────────────────────────────
def hyperbola_to_bbox(o):
    """双曲线参数 -> [x1, y1, x2, y2]（与 rect 标注的派生公式一致）。"""
    half_w = o["width"] / 2.0
    x1 = o["x_vertex"] - half_w
    y1 = o["y_vertex"] - o["thickness"] / 2.0
    x2 = o["x_vertex"] + half_w
    y2 = o["y_vertex"] + o["height"] + o["thickness"] / 2.0
    return [x1, y1, x2, y2]


def rect_to_bbox_scaled(r, sx, sy):
    """rect 标注 {x1,y1,width,height}（原图坐标）-> [x1,y1,x2,y2]（缩放到 input 尺度）。"""
    x1 = r["x1"] * sx
    y1 = r["y1"] * sy
    x2 = (r["x1"] + r["width"])  * sx
    y2 = (r["y1"] + r["height"]) * sy
    return [x1, y1, x2, y2]


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter  = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / max(union, 1e-6)


# ── 主评估 ────────────────────────────────────────────────────────────────────
def main():
    exp = load_exp_module(EXP_FILE)
    device = exp.device

    work_dir = WORK_DIR or find_work_dir(_HERE)
    print(f"Using work_dir : {work_dir}")
    print(f"bbox IoU thresh: {BBOX_IOU_THR}   vertex thresh: {exp.VERTEX_THRESH}px")

    # rect (bbox) GT
    rect_path = os.path.join(os.path.dirname(exp.data_sources[0]["annotation_json"]), RECT_JSON_NAME)
    with open(rect_path, "r", encoding="utf-8") as f:
        rect_ann = json.load(f)

    # 与训练完全相同的数据集与划分
    full_dataset = exp.HyperbolaDataset(
        image_dir=exp.data_sources[0]["image_dir"],
        annotation_json=exp.data_sources[0]["annotation_json"],
        input_size=exp.input_size,
        hm_stride=exp.HM_STRIDE,
        sigma=exp.HM_SIGMA,
    )
    n_total = len(full_dataset)
    input_h, input_w = exp.input_size

    rows = []
    for seed in exp.SEEDS:
        _, _, test_idx = exp.make_split(n_total, seed)
        test_set = Subset(full_dataset, test_idx)

        best_path = os.path.join(work_dir, f"seed{seed:02d}", "checkpoints", "best_model.pth")
        if not os.path.exists(best_path):
            print(f"  [Seed {seed}] 跳过：找不到 {best_path}")
            continue

        model = exp.HyperbolaNet(in_ch=1, base_ch=32).to(device)
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        # 累计
        n_gt = 0
        n_box_detected = 0
        n_vtx_detected = 0
        ap_tp, ap_scores = [], []

        for i in range(len(test_set)):
            meta = test_set[i][-1]
            _, _, detections = exp.predict_single_image(
                model, meta["image_path"], exp.input_size, device,
                obj_thresh=exp.HM_THRESH, nms_k=exp.nms_kernel, max_det=exp.max_det,
            )

            # GT bbox（来自 rect json，缩放到 input 尺度）
            orig_h, orig_w = meta["orig_size"]
            sx, sy = input_w / orig_w, input_h / orig_h
            gt_recs  = [r for r in rect_ann.get(meta["image_name"], []) if r.get("label", "") == "hyperbola"]
            gt_boxes = [rect_to_bbox_scaled(r, sx, sy) for r in gt_recs]
            n_gt += len(gt_boxes)

            # 预测 bbox（双曲线 -> 外接框）
            pred = sorted(detections, key=lambda d: d["score"], reverse=True)
            pred_boxes = [hyperbola_to_bbox(d) for d in pred]

            # bbox 贪心匹配（IoU>=阈值，每个 GT 最多匹配一次）
            gt_matched = [False] * len(gt_boxes)
            for pb, d in zip(pred_boxes, pred):
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gt_boxes):
                    if gt_matched[j]:
                        continue
                    iou = bbox_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                is_tp = best_iou >= BBOX_IOU_THR and best_j >= 0
                if is_tp:
                    gt_matched[best_j] = True
                ap_tp.append(is_tp)
                ap_scores.append(float(d["score"]))
            n_box_detected += sum(gt_matched)

            # vertex-based recall（对照：用双曲线 GT 顶点 + 顶点距离阈值）
            gt_objs = meta["objects"]
            for gt_obj in gt_objs:
                best_dist = float("inf")
                for d in detections:
                    best_dist = min(best_dist, exp._vertex_dist(d, gt_obj))
                if best_dist <= exp.VERTEX_THRESH:
                    n_vtx_detected += 1

        bbox_recall = n_box_detected / max(n_gt, 1)
        vtx_recall  = n_vtx_detected / max(n_gt, 1)
        bbox_map50  = exp.compute_ap50(ap_tp, ap_scores, n_gt)

        print(f"  [Seed {seed}]  n_gt={n_gt:3d}  "
              f"bbox_recall={bbox_recall:.4f}  bbox_mAP50={bbox_map50:.4f}  "
              f"(vertex_recall={vtx_recall:.4f})")
        rows.append({
            "seed": seed,
            "n_gt": n_gt,
            "bbox_recall": bbox_recall,
            "bbox_mAP50":  bbox_map50,
            "vertex_recall": vtx_recall,
        })

    # 汇总
    if rows:
        keys = ["bbox_recall", "bbox_mAP50", "vertex_recall"]
        means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        stds  = {k: float(np.std ([r[k] for r in rows])) for k in keys}
        print("\n" + "=" * 60)
        print(f"{len(rows)}-Seed bbox 检出率评估 (IoU>={BBOX_IOU_THR})")
        print("=" * 60)
        for k in keys:
            print(f"  {k:<14} = {means[k]:.4f} ± {stds[k]:.4f}")
        print("=" * 60)

        csv_path = os.path.join(work_dir, "bbox_eval_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
