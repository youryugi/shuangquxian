"""可视化带修正前后的框：绿 = GT，红 = 原预测，蓝 = 修正后；底图叠加注意力图。
    python vis_refine.py [--seed 0] [--n 16] [--conf 0.5]
输出 runs_cv/refine/vis/seed<k>/<stem>.png，优先挑"修正后 IoU 变化最大"的图。"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "detector_comparison"))
from refine_with_band import FOLDS, OUT, data_dir, load_attn, run_dir, EXP_ABS  # noqa: E402
from evaluate import box_iou  # noqa: E402


def best_iou(box, gts):
    return max([box_iou(box, g) for g in gts], default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=16); ap.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args()
    cands = []
    for f in FOLDS:
        gt = json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["test"]
        od = OUT / EXP_ABS / f"f{f}_seed{a.seed}"
        orig = json.load(open(run_dir(EXP_ABS, f, a.seed) / "preds_test.json"))
        ref = json.load(open(od / "preds_test_refined.json"))
        attn, _ = load_attn(f, a.seed, "test")
        for stem, g in gt.items():
            o = [d for d in orig.get(stem, []) if d[4] >= a.conf]; r = [d for d in ref.get(stem, []) if d[4] >= a.conf]
            gain = sum(best_iou(rb[:4], g["boxes"]) - best_iou(ob[:4], g["boxes"]) for ob, rb in zip(o, r))
            cands.append((abs(gain), gain, f, stem, g, o, r, attn[stem]))
    cands.sort(key=lambda c: -c[0])
    out = OUT / "vis" / f"seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    for _, gain, f, stem, g, o, r, A in cands[:a.n]:
        img = cv2.imread(str(HERE.parents[2] / g["file"]))
        H, W = img.shape[:2]
        heat = cv2.applyColorMap((np.clip(cv2.resize(A, (W, H)), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        panel_a = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
        panel_b = img.copy()
        for x1, y1, x2, y2 in g["boxes"]:
            cv2.rectangle(panel_b, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        for d in o:
            cv2.rectangle(panel_b, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 0, 255), 1)
        for d in r:
            cv2.rectangle(panel_b, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 120, 0), 1)
        canvas = np.concatenate([cv2.resize(panel_a, (W * 2, H * 2)), cv2.resize(panel_b, (W * 2, H * 2))], axis=1)
        cv2.putText(canvas, f"{stem} f{f} sumIoUgain {gain:+.2f}  green=GT red=orig blue=refined", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}.png"), canvas)
    print(f"wrote {min(a.n, len(cands))} images to {out}")


if __name__ == "__main__":
    main()
