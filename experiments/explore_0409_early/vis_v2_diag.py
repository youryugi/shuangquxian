"""诊断：V2 注意力到底偏向 GT 带，还是落在背景？
- 量化全 test：带内 A 均值 vs 带外 A 均值（比值>1 说明偏向带，≈1 说明没区分/在背景）。
- 可视化几张：注意力热力 + GT 带绿色轮廓线，直接看红区落在哪。
"""
import os, sys
import cv2
import numpy as np
import torch

import attn_cnn as ac
import attn_cnn_v2 as v2

exp, device, input_size = v2.exp, v2.device, v2.input_size
WORK = sys.argv[1] if len(sys.argv) > 1 else "attn_cnn_v2_0621_1424"
SEED = 0
N = 6


def overlay(gray_u8, heat01):
    h, w = gray_u8.shape
    heat = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8), (w, h))
    hc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), 0.5, hc, 0.5, 0)


def main():
    full = ac.AttnDataset(input_size=input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    tr, va, te = exp.make_split(len(full), SEED)
    model = v2.AttnBBoxNetV2(in_ch=1, base_ch=32, use_attn=True).to(device)
    model.load_state_dict(torch.load(os.path.join(WORK, f"seed{SEED}_with_attn_best.pth"), map_location=device))
    model.eval()

    # 量化：全 test 带内/带外注意力均值
    in_vals, out_vals, ratios = [], [], []
    for i in te:
        img, _, _, _, _, band, meta = full[i]
        _, _, A = ac.predict(model, img.unsqueeze(0))           # A: stride4 sigmoid (Hs,Ws)
        b = band[0].numpy()
        Ar = cv2.resize(A, (b.shape[1], b.shape[0]))
        m = b > 0.5
        if m.sum() < 5:
            continue
        im, om = float(Ar[m].mean()), float(Ar[~m].mean())
        in_vals.append(im); out_vals.append(om); ratios.append(im / (om + 1e-6))
    print(f"全 test({len(in_vals)} 张有带):")
    print(f"  带内 A 均值 = {np.mean(in_vals):.3f}")
    print(f"  带外 A 均值 = {np.mean(out_vals):.3f}")
    print(f"  带内/带外 比值 = {np.mean(ratios):.2f}  (>1 偏向带, ~1 不分/在背景)")

    # 可视化：注意力 + GT 带绿轮廓
    rows = []
    for i in te[:N]:
        img, _, _, _, _, band, meta = full[i]
        gray = (img[0].numpy() * 255).astype(np.uint8)
        _, _, A = ac.predict(model, img.unsqueeze(0))
        H, W = gray.shape
        bfull = cv2.resize(band[0].numpy(), (W, H))
        cnts, _ = cv2.findContours((bfull > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        p_orig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        p_aov = overlay(gray, A)
        cv2.drawContours(p_aov, cnts, -1, (0, 255, 0), 2)       # GT 带绿轮廓叠在注意力上
        p_gtline = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(p_gtline, cnts, -1, (0, 255, 0), 2)
        for p, t in [(p_orig, "Original"), (p_aov, "Attn + GT outline"), (p_gtline, "GT band outline")]:
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack([p_orig, p_aov, p_gtline]))

    out = os.path.join(WORK, "vis_attn_diag.png")
    cv2.imwrite(out, np.vstack(rows))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
