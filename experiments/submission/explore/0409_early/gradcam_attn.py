"""
用 Grad-CAM 看网络"做检测时关注哪里"——尤其 no_attn(无注意力监督)时的隐式注意力。
对 bottleneck 特征做 Grad-CAM(目标=heatmap 最强响应)。
加载 attn_multiseed 的 seed0 with/no 模型对比。

输出五联：原图 | no_attn Grad-CAM | with_attn Grad-CAM | with_attn 显式A | GT双曲线带
"""
import os
import glob
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import attn_cnn as ac

N_VIS = 100000   # 全部 test


def grad_cam(model, x):
    """对 model.bottleneck 做 Grad-CAM，score=heatmap 最强响应。"""
    acts, grads = {}, {}
    h1 = model.bottleneck.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = model.bottleneck.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    model.zero_grad()
    hm_logit, _, _, _ = model(x)
    score = hm_logit.max()
    score.backward()
    A = acts["v"][0].detach(); g = grads["v"][0].detach()
    w = g.mean(dim=(1, 2))
    cam = F.relu((w[:, None, None] * A).sum(0)).float().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    h1.remove(); h2.remove()
    return cam


def overlay(gray_u8, heat01):
    h, w = gray_u8.shape
    heat = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8), (w, h))
    hc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), 0.55, hc, 0.45, 0)


def main():
    ms = sorted(glob.glob(os.path.join(os.getcwd(), "attn_multiseed_*")))
    if not ms:
        print("找不到 attn_multiseed_*"); return
    # 用 ×(1+gate) 的那次（最早一个，0402）
    msdir = ms[0]
    print("models from", msdir)
    full = ac.AttnDataset(input_size=ac.input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    tr, va, te = ac.exp.make_split(len(full), 0)

    m_no = ac.AttnBBoxNet(in_ch=1, base_ch=32, use_attn=False).to(ac.device)
    m_no.load_state_dict(torch.load(os.path.join(msdir, "seed0_no_attn_best.pth"), map_location=ac.device)); m_no.eval()
    m_with = ac.AttnBBoxNet(in_ch=1, base_ch=32, use_attn=True).to(ac.device)
    m_with.load_state_dict(torch.load(os.path.join(msdir, "seed0_with_attn_best.pth"), map_location=ac.device)); m_with.eval()

    work = os.path.join(os.getcwd(), f"gradcam_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    H, W = ac.input_size

    for k, i in enumerate(te[:N_VIS]):
        img_t, _, _, _, _, _, meta = full[i]
        gray = (img_t[0].numpy() * 255).astype(np.uint8)
        x = img_t.unsqueeze(0).to(ac.device)

        cam_no = grad_cam(m_no, x.clone())
        cam_with = grad_cam(m_with, x.clone())
        with torch.no_grad():
            _, _, A = ac.predict(m_with, img_t.unsqueeze(0))

        gt_band = np.zeros((H, W), np.float32)
        for o in meta["objects"]:
            gt_band = np.maximum(gt_band, ac.exp.rasterize_hyperbola_band_mask(
                H, W, o["x_vertex"], o["y_vertex"], o["width"], o["height"], o["thickness"]))
        panel_gt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR); panel_gt[gt_band > 0.5] = (0, 200, 0)

        panels = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                  overlay(gray, cam_no),
                  overlay(gray, cam_with),
                  overlay(gray, A) if A is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                  panel_gt]
        titles = ["Original", "no_attn Grad-CAM", "with_attn Grad-CAM", "with_attn A", "GT band"]
        for p, t in zip(panels, titles):
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(work, f"{k:02d}_{os.path.splitext(meta['image_name'])[0]}.png"), np.hstack(panels))
    print(f"\nSaved {min(N_VIS, len(te))} -> {work}")


if __name__ == "__main__":
    main()
