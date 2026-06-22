"""没用注意力监督(no_attn)时，网络的"隐式注意力"在哪？用 Grad-CAM 看检测头盯着哪。
对比：原图 | no_attn Grad-CAM | with_attn 显式注意力A | GT带轮廓。
并量化各自的「带内/带外」比值，直接和 with_attn 的 4.34× 对照。
"""
import os, sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import attn_cnn as ac
import attn_cnn_v2 as v2

exp, device, input_size = v2.exp, v2.device, v2.input_size
WORK = sys.argv[1] if len(sys.argv) > 1 else "attn_cnn_v2_0621_1424"
SEED = 0
N = 6


def gradcam(model, img):
    """对 bottleneck 输出做 Grad-CAM，target=heatmap 最大响应。返回 stride8 CAM(0-1)。"""
    acts = {}
    def fwd(m, i, o): acts["f"] = o
    def bwd(m, gi, go): acts["g"] = go[0]
    h1 = model.bottleneck.register_forward_hook(fwd)
    h2 = model.bottleneck.register_full_backward_hook(bwd)
    model.zero_grad()
    hm_logit, _, _, _ = model(img.unsqueeze(0).to(device))   # fp32，不用 autocast
    target = hm_logit.max()
    target.backward()
    f, g = acts["f"][0], acts["g"][0]            # (C,H,W)
    w = g.mean(dim=(1, 2))                       # GAP 权重
    cam = F.relu((w[:, None, None] * f).sum(0))  # (H,W)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    h1.remove(); h2.remove()
    return cam.detach().float().cpu().numpy()


def ratio_in_out(map01, band):
    Ar = cv2.resize(map01, (band.shape[1], band.shape[0]))
    m = band > 0.5
    if m.sum() < 5:
        return None
    return float(Ar[m].mean()), float(Ar[~m].mean())


def overlay(gray_u8, heat01):
    h, w = gray_u8.shape
    heat = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8), (w, h))
    hc = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), 0.5, hc, 0.5, 0)


def main():
    full = ac.AttnDataset(input_size=input_size, hm_stride=ac.HM_STRIDE, sigma=ac.HM_SIGMA)
    tr, va, te = exp.make_split(len(full), SEED)

    m_no = v2.AttnBBoxNetV2(in_ch=1, base_ch=32, use_attn=False).to(device)
    m_no.load_state_dict(torch.load(os.path.join(WORK, f"seed{SEED}_no_attn_best.pth"), map_location=device))
    m_no.eval()
    m_at = v2.AttnBBoxNetV2(in_ch=1, base_ch=32, use_attn=True).to(device)
    m_at.load_state_dict(torch.load(os.path.join(WORK, f"seed{SEED}_with_attn_best.pth"), map_location=device))
    m_at.eval()

    cam_in, cam_out, att_in, att_out = [], [], [], []
    for i in te:
        img, _, _, _, _, band, _ = full[i]
        b = band[0].numpy()
        cam = gradcam(m_no, img)
        r1 = ratio_in_out(cam, b)
        with torch.no_grad():
            _, _, A = ac.predict(m_at, img.unsqueeze(0))
        r2 = ratio_in_out(A, b)
        if r1: cam_in.append(r1[0]); cam_out.append(r1[1])
        if r2: att_in.append(r2[0]); att_out.append(r2[1])
    cr = np.mean(cam_in) / (np.mean(cam_out) + 1e-6)
    ar = np.mean(att_in) / (np.mean(att_out) + 1e-6)
    print(f"no_attn  Grad-CAM : 带内={np.mean(cam_in):.3f} 带外={np.mean(cam_out):.3f} 比值={cr:.2f}")
    print(f"with_attn 显式注意力: 带内={np.mean(att_in):.3f} 带外={np.mean(att_out):.3f} 比值={ar:.2f}")

    rows = []
    for i in te[:N]:
        img, _, _, _, _, band, _ = full[i]
        gray = (img[0].numpy() * 255).astype(np.uint8)
        H, W = gray.shape
        cam = gradcam(m_no, img)
        with torch.no_grad():
            _, _, A = ac.predict(m_at, img.unsqueeze(0))
        bfull = cv2.resize(band[0].numpy(), (W, H))
        cnts, _ = cv2.findContours((bfull > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p_o = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        p_c = overlay(gray, cam);  cv2.drawContours(p_c, cnts, -1, (0, 255, 0), 2)
        p_a = overlay(gray, A);    cv2.drawContours(p_a, cnts, -1, (0, 255, 0), 2)
        for p, t in [(p_o, "Original"), (p_c, "no_attn Grad-CAM"), (p_a, "with_attn Attn")]:
            cv2.putText(p, t, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack([p_o, p_c, p_a]))
    out = os.path.join(WORK, "vis_gradcam_compare.png")
    cv2.imwrite(out, np.vstack(rows))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
