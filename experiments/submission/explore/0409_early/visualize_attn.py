"""
可视化注意力图：把模型预测的注意力图叠到原图上，肉眼确认注意力是否落在双曲线带上。
输出每张测试图的四联图：原图 | 注意力热力图叠加 | GT双曲线带 | 预测框。

默认会先训练一个 with_attn 模型（单 seed）再可视化；若已有 best.pth 可通过 MODEL_PATH 直接加载。
"""
import os
from datetime import datetime

import cv2
import numpy as np
import torch

import attn_cnn as ac

N_VIS      = 100000  # 全部 test
MODEL_PATH = None    # 指定已训好的 with_attn_best.pth；None=现训一个


def overlay_heatmap(gray_u8, heat01):
    """heat01: HxW in [0,1] -> 叠加到灰度图。"""
    h, w = gray_u8.shape
    heat = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8), (w, h))
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.55, heat_color, 0.45, 0)


def main():
    now = datetime.now()
    work = os.path.join(os.getcwd(), f"attn_vis_{now.strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)

    full = ac.AttnDataset(input_size=ac.input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    tr, va, te = ac.exp.make_split(len(full), ac.SEED)

    model = ac.AttnBBoxNet(in_ch=1, base_ch=32, use_attn=True).to(ac.device)
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=ac.device)); model.eval()
        print(f"loaded {MODEL_PATH}")
    else:
        print("training a with_attn model for visualization ...")
        model = ac.train_model(True, full, tr, va, ac.num_epochs, work, "vis")

    H, W = ac.input_size
    for k, i in enumerate(te[:N_VIS]):
        img_t, _, _, _, _, band, meta = full[i]
        boxes, scores, A = ac.predict(model, img_t.unsqueeze(0))
        gray = (img_t[0].numpy() * 255).astype(np.uint8)

        panel_orig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        panel_attn = overlay_heatmap(gray, A) if A is not None else panel_orig.copy()
        # GT 双曲线带
        gt_band = np.zeros((H, W), np.float32)
        for o in meta["objects"]:
            gt_band = np.maximum(gt_band, ac.exp.rasterize_hyperbola_band_mask(
                H, W, o["x_vertex"], o["y_vertex"], o["width"], o["height"], o["thickness"]))
        panel_gt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        panel_gt[gt_band > 0.5] = (0, 200, 0)
        # 预测框
        panel_pred = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for b in boxes:
            cv2.rectangle(panel_pred, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 220), 2)

        for p, t in [(panel_orig, "Original"), (panel_attn, "Attention"), (panel_gt, "GT band"), (panel_pred, "Pred boxes")]:
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vis = np.hstack([panel_orig, panel_attn, panel_gt, panel_pred])
        out = os.path.join(work, f"{k:02d}_{os.path.splitext(meta['image_name'])[0]}.png")
        cv2.imwrite(out, vis)
    print(f"\nSaved {min(N_VIS, len(te))} visuals -> {work}")


if __name__ == "__main__":
    main()
