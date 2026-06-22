"""
扫 HM_THRESH，找 recall/F1 最佳工作点（不重训，加载已有 best_model）。
高效做法：每图前向一次（低阈值取全部候选），再在各阈值下纯 Python 筛选 score>=t。
指标：bbox 级 P/R/F1（双曲线->外接框，IoU>=0.5）+ vertex_recall（顶点距离<=VERTEX_THRESH）。
"""
import os
import glob
import importlib.util

import numpy as np
import torch
from torch.utils.data import Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp0616", os.path.join(_HERE, "0616-1.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
device = exp.device

WORK_DIR  = None   # None=自动找含 5 个 best_model 的最新 0616-1_* 目录
THRS      = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
CAND_THR  = 0.04   # 取候选用的低阈值（低于所有扫描阈值）
CAND_MAXDET = 15   # 候选上限（每图双曲线很少，15 足够覆盖）
FINAL_MAXDET = exp.max_det
BBOX_IOU_THR = 0.5


def hyperbola_to_bbox(o):
    hw = o["width"] / 2.0
    return [o["x_vertex"] - hw, o["y_vertex"] - o["thickness"] / 2.0,
            o["x_vertex"] + hw, o["y_vertex"] + o["height"] + o["thickness"] / 2.0]


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def find_work_dir():
    for d in sorted(glob.glob(os.path.join(_HERE, "0616-1_*")), reverse=True):
        if len(glob.glob(os.path.join(d, "seed*", "checkpoints", "best_model.pth"))) >= 1:
            return d
    raise FileNotFoundError("找不到 0616-1_* 训练输出")


def main():
    work = WORK_DIR or find_work_dir()
    print(f"work_dir: {work}\n按候选阈值 {CAND_THR} 前向缓存，再扫 {THRS}\n")

    full = exp.HyperbolaDataset(
        image_dir=exp.data_sources[0]["image_dir"],
        annotation_json=exp.data_sources[0]["annotation_json"],
        input_size=exp.input_size, hm_stride=exp.HM_STRIDE, sigma=exp.HM_SIGMA)
    n_total = len(full)

    # 阶段1：前向缓存（每图一次）
    cache = {}
    for seed in exp.SEEDS:
        best = os.path.join(work, f"seed{seed:02d}", "checkpoints", "best_model.pth")
        if not os.path.exists(best):
            print(f"seed {seed} 缺 best_model，跳过"); continue
        model = exp.HyperbolaNet(in_ch=1, base_ch=32).to(device)
        model.load_state_dict(torch.load(best, map_location=device)); model.eval()
        _, _, test_idx = exp.make_split(n_total, seed)
        test = Subset(full, test_idx)
        items = []
        for i in range(len(test)):
            meta = test[i][-1]
            _, _, dets = exp.predict_single_image(
                model, meta["image_path"], exp.input_size, device,
                obj_thresh=CAND_THR, nms_k=exp.nms_kernel, max_det=CAND_MAXDET)
            items.append((dets, meta["objects"]))
        cache[seed] = items
        print(f"  cached seed {seed}: {len(items)} imgs")

    # 阶段2：多阈值评估
    print(f"\n{'thr':>6}{'P':>10}{'R':>10}{'F1':>10}{'vtx_rec':>10}")
    print("-" * 46)
    best_f1, best_t, best_line = -1, None, None
    for t in THRS:
        Ps, Rs, Fs, Vs = [], [], [], []
        for seed, items in cache.items():
            TP = FP = FN = vtx = ngt = 0
            for dets, gts in items:
                sel = sorted([d for d in dets if d["score"] >= t], key=lambda d: -d["score"])[:FINAL_MAXDET]
                gtb = [hyperbola_to_bbox(o) for o in gts]
                matched = [False] * len(gtb)
                for d in sel:
                    db = hyperbola_to_bbox(d)
                    bi, bj = 0.0, -1
                    for j, gb in enumerate(gtb):
                        if matched[j]:
                            continue
                        iou = bbox_iou(db, gb)
                        if iou > bi:
                            bi, bj = iou, j
                    if bi >= BBOX_IOU_THR and bj >= 0:
                        matched[bj] = True; TP += 1
                    else:
                        FP += 1
                FN += len(gtb) - sum(matched)
                ngt += len(gtb)
                for o in gts:
                    if sel and min(exp._vertex_dist(d, o) for d in sel) <= exp.VERTEX_THRESH:
                        vtx += 1
            P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
            Ps.append(P); Rs.append(R)
            Fs.append(2 * P * R / max(P + R, 1e-9)); Vs.append(vtx / max(ngt, 1))
        mP, mR, mF, mV = np.mean(Ps), np.mean(Rs), np.mean(Fs), np.mean(Vs)
        mark = "  <- 当前" if abs(t - exp.HM_THRESH) < 1e-6 else ""
        line = f"{t:>6.2f}{mP:>10.4f}{mR:>10.4f}{mF:>10.4f}{mV:>10.4f}{mark}"
        print(line)
        if mF > best_f1:
            best_f1, best_t, best_line = mF, t, line.strip()

    print("-" * 46)
    print(f"\n最佳 F1 在 HM_THRESH={best_t}：{best_line}")
    print(f"(当前用的是 {exp.HM_THRESH})")


if __name__ == "__main__":
    main()
