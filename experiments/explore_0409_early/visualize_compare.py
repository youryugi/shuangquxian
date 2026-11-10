"""
对比 with_attn vs no_attn 网络"关注哪里"。
no_attn 没有显式注意力图，所以对两者都用【特征激活图】(门控后/bottleneck 特征沿通道的 L2 强度)，
公平对比："有注意力监督 -> 特征聚焦双曲线带；无 -> 特征分散/背景"。

加载 attn_multiseed 已训好的 seed0 模型，不重训。
输出五联：原图 | with_attn 特征激活 | no_attn 特征激活 | GT双曲线带 | with_attn 显式注意力A
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


def gated_feat(model, x):
    """复现 forward 到注意力门控后，返回 (feat, A)。no_attn 时 A=None。"""
    _, x = model.down1(x); _, x = model.down2(x)
    f3, x = model.down3(x)
    feat = model.bottleneck(x)
    A = None
    if model.use_attn:
        a_logit = model.attn_head(f3)
        gate = F.avg_pool2d(torch.sigmoid(a_logit), 2)
        feat = feat * (1.0 + gate)
        A = torch.sigmoid(a_logit[0, 0]).float().cpu().numpy()
    return feat, A


def act_map(feat):
    """bottleneck 特征沿通道 L2 -> 归一化空间激活图 (H,W)。"""
    a = feat[0].norm(dim=0).float().cpu().numpy()
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def overlay(gray_u8, heat01):
    h, w = gray_u8.shape
    heat = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8), (w, h))
    hc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), 0.55, hc, 0.45, 0)


def main():
    ms = sorted(glob.glob(os.path.join(os.getcwd(), "attn_multiseed_*")))
    if not ms:
        print("找不到 attn_multiseed_* 目录"); return
    msdir = ms[-1]
    wp = os.path.join(msdir, "seed0_with_attn_best.pth")
    npth = os.path.join(msdir, "seed0_no_attn_best.pth")
    print(f"with: {wp}\nno  : {npth}")

    full = ac.AttnDataset(input_size=ac.input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    tr, va, te = ac.exp.make_split(len(full), 0)

    m_with = ac.AttnBBoxNet(in_ch=1, base_ch=32, use_attn=True).to(ac.device)
    m_with.load_state_dict(torch.load(wp, map_location=ac.device)); m_with.eval()
    m_no = ac.AttnBBoxNet(in_ch=1, base_ch=32, use_attn=False).to(ac.device)
    m_no.load_state_dict(torch.load(npth, map_location=ac.device)); m_no.eval()

    work = os.path.join(os.getcwd(), f"attn_compare_{datetime.now().strftime('%m%d_%H%M')}")
    os.makedirs(work, exist_ok=True)
    H, W = ac.input_size

    for k, i in enumerate(te[:N_VIS]):
        img_t, _, _, _, _, _, meta = full[i]
        gray = (img_t[0].numpy() * 255).astype(np.uint8)
        x = img_t.unsqueeze(0).to(ac.device)
        with torch.no_grad():
            fw, A = gated_feat(m_with, x)
            fn, _ = gated_feat(m_no, x)

        gt_band = np.zeros((H, W), np.float32)
        for o in meta["objects"]:
            gt_band = np.maximum(gt_band, ac.exp.rasterize_hyperbola_band_mask(
                H, W, o["x_vertex"], o["y_vertex"], o["width"], o["height"], o["thickness"]))
        panel_gt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR); panel_gt[gt_band > 0.5] = (0, 200, 0)

        panels = [cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                  overlay(gray, act_map(fw)),
                  overlay(gray, act_map(fn)),
                  panel_gt,
                  overlay(gray, A) if A is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]
        titles = ["Original", "with_attn feat", "no_attn feat", "GT band", "with_attn A"]
        for p, t in zip(panels, titles):
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(work, f"{k:02d}_{os.path.splitext(meta['image_name'])[0]}.png"), np.hstack(panels))
    print(f"\nSaved {min(N_VIS, len(te))} -> {work}")


if __name__ == "__main__":
    main()
