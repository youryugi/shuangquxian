"""快速可视化 V2 with_attn 学到的注意力图：原图 | 注意力A | A叠加 | GT带 | 预测框。
拼成一张大图，直观看注意力到底落在哪。"""
import os, sys
import cv2
import numpy as np
import torch

import attn_cnn as ac
import attn_cnn_v2 as v2   # 内含 make_split 70/15/15 patch + AttnBBoxNetV2

exp, device, input_size = v2.exp, v2.device, v2.input_size
WORK = sys.argv[1] if len(sys.argv) > 1 else "attn_cnn_v2_0621_1424"
SEED = 0
N = 6


def overlay(gray_u8, heat01):
    h, w = gray_u8.shape
    heat = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8), (w, h))
    hc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), 0.55, hc, 0.45, 0)


def main():
    full = ac.AttnDataset(input_size=input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    tr, va, te = exp.make_split(len(full), SEED)
    model = v2.AttnBBoxNetV2(in_ch=1, base_ch=32, use_attn=True).to(device)
    model.load_state_dict(torch.load(os.path.join(WORK, f"seed{SEED}_with_attn_best.pth"), map_location=device))
    model.eval()

    rows = []
    for i in te[:N]:
        img, _, _, _, _, band, meta = full[i]
        gray = (img[0].numpy() * 255).astype(np.uint8)
        boxes, scores, A = ac.predict(model, img.unsqueeze(0))     # A: stride4 注意力 sigmoid
        gt = [ac.hyperbola_to_bbox(o) for o in meta["objects"]]

        p_orig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        p_Araw = overlay(np.zeros_like(gray), A)                   # 纯注意力热力（不叠原图）
        p_Aov  = overlay(gray, A)                                  # 注意力叠在原图上
        p_gt   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        bandf  = cv2.resize(band[0].numpy(), (gray.shape[1], gray.shape[0]))
        p_gt[bandf > 0.5] = (0, 200, 0)
        p_box  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for b in gt:
            cv2.rectangle(p_box, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 2)
        for b in boxes:
            cv2.rectangle(p_box, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 220), 2)

        for p, t in [(p_orig, "Original"), (p_Araw, "Attn (raw)"), (p_Aov, "Attn overlay"),
                     (p_gt, "GT band"), (p_box, "GT(g)/Pred(r) box")]:
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack([p_orig, p_Araw, p_Aov, p_gt, p_box]))

    out = os.path.join(WORK, "vis_attn_grid.png")
    cv2.imwrite(out, np.vstack(rows))
    # 统计注意力数值分布
    amax = float(A.max()); amean = float(A.mean()); ahot = float((A > 0.5).mean())
    print(f"A: max={amax:.3f} mean={amean:.3f} frac(>0.5)={ahot:.3f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
