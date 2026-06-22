"""
对 attn_cnn 的 with_attn 模型(5 seed)扫 HM_THRESH，找 bbox F1 最优工作点。
每图前向一次取候选(低阈值+大 max_det)，再各阈值纯筛选 score>=thr。
"""
import os
import glob

import numpy as np
import torch

import attn_cnn as ac

THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SEEDS = [0, 1, 2, 3, 4]
FINAL_MAXDET = ac.max_det


def main():
    full = ac.AttnDataset(input_size=ac.input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    n = len(full)
    ms = sorted(glob.glob("attn_multiseed_*"))[0]
    print(f"models from {ms}")

    ac.HM_THRESH = 0.01   # 取候选用的低阈值
    ac.max_det = 50

    cache = {}
    for seed in SEEDS:
        bp = os.path.join(ms, f"seed{seed}_with_attn_best.pth")
        if not os.path.exists(bp):
            continue
        model = ac.AttnBBoxNet(in_ch=1, base_ch=32, use_attn=True).to(ac.device)
        model.load_state_dict(torch.load(bp, map_location=ac.device)); model.eval()
        _, _, te = ac.exp.make_split(n, seed)
        items = []
        for i in te:
            img = full[i][0]; meta = full[i][-1]
            boxes, scores, _ = ac.predict(model, img.unsqueeze(0))
            gt = [ac.hyperbola_to_bbox(o) for o in meta["objects"]]
            items.append((boxes, scores, gt))
        cache[seed] = items
        print(f"  cached seed {seed}: {len(items)} imgs")

    print(f"\n{'thr':>6}{'P':>9}{'R':>9}{'F1':>9}")
    print("-" * 33)
    best = (-1, None)
    for thr in THRS:
        Ps, Rs, Fs = [], [], []
        for seed, items in cache.items():
            TP = FP = FN = 0
            for boxes, scores, gt in items:
                sel = sorted([(b, s) for b, s in zip(boxes, scores) if s >= thr], key=lambda z: -z[1])[:FINAL_MAXDET]
                matched = [False] * len(gt)
                for b, s in sel:
                    bi, bj = 0.0, -1
                    for j, gb in enumerate(gt):
                        if matched[j]:
                            continue
                        iou = ac.bbox_iou(b, gb)
                        if iou > bi:
                            bi, bj = iou, j
                    if bi >= 0.5 and bj >= 0:
                        matched[bj] = True; TP += 1
                    else:
                        FP += 1
                FN += len(gt) - sum(matched)
            P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
            Ps.append(P); Rs.append(R); Fs.append(2 * P * R / max(P + R, 1e-9))
        mP, mR, mF = np.mean(Ps), np.mean(Rs), np.mean(Fs)
        mark = "  <- 当前" if abs(thr - 0.30) < 1e-6 else ""
        print(f"{thr:>6.2f}{mP:>9.4f}{mR:>9.4f}{mF:>9.4f}{mark}")
        if mF > best[0]:
            best = (mF, thr)
    print("-" * 33)
    print(f"\n最佳 F1={best[0]:.4f} @ HM_THRESH={best[1]}（当前 0.30 的 F1≈0.489）")


if __name__ == "__main__":
    main()
