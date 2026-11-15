import os
import importlib.util
import numpy as np
import cv2
import argparse
import torch
import torch.nn.functional as F


# =========================================================
# 加载 0607-1.py 里的模型和工具函数
# =========================================================

def _load_sibling(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location("_hyp", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = _load_sibling("0607-1.py")

HyperbolaNet            = m.HyperbolaNet
HyperbolaDataset        = m.HyperbolaDataset
build_gt_mask_from_meta = m.build_gt_mask_from_meta
PARAM_CH                = m.PARAM_CH
N_PARAM                 = m.N_PARAM

input_size   = m.input_size
HM_STRIDE    = m.HM_STRIDE
HM_SIGMA     = m.HM_SIGMA
HM_THRESH    = m.HM_THRESH
device       = m.device
data_sources = m.data_sources


# =========================================================
# Grad-CAM（推理参数版）
# =========================================================

class GradCAMParam:
    """
    bottleneck 输出做 Grad-CAM，目标信号：
      heatmap 置信度最高的顶点位置处的 param_head 预测值。
    对 width / height / thickness 三个通道分别计算，共返回 3 张 CAM。
    """

    def __init__(self, model: HyperbolaNet):
        self.model  = model
        self._feats = None
        self._grads = None
        self._h_fwd = model.bottleneck.register_forward_hook(self._fwd_hook)
        self._h_bwd = model.bottleneck.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _, __, output):
        self._feats = output

    def _bwd_hook(self, _, __, grad_output):
        self._grads = grad_output[0]

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()

    def __call__(self, x: torch.Tensor):
        """
        x: (1, 1, H, W) on device
        Returns
            cams   : list of 3 (hm_h, hm_w) float32 arrays, 归一化到 0-1
                     顺序为 width / height / thickness
            hm_np  : (hm_h, hm_w) float32  sigmoid 热力图
            peak   : (peak_y, peak_x) 最置信顶点的热力图坐标
        """
        # 一次前向，保留计算图供多次反向
        hm_logit, param_out, _ = self.model(x)
        hm = torch.sigmoid(hm_logit)
        hm_np = hm.squeeze().detach().cpu().numpy()

        # 找热力图最大值位置（推理顶点）
        peak_flat = np.argmax(hm_np)
        peak_y, peak_x = np.unravel_index(peak_flat, hm_np.shape)

        cams = []
        for ch in range(N_PARAM):
            self.model.zero_grad()
            # 以该顶点处的第 ch 个参数预测值为目标信号
            target = param_out[0, ch, int(peak_y), int(peak_x)]
            retain = (ch < N_PARAM - 1)
            target.backward(retain_graph=retain)

            weights = self._grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self._feats).sum(dim=1, keepdim=True)
            cam = F.relu(cam).squeeze().detach().cpu().numpy()
            mn, mx = cam.min(), cam.max()
            cam = (cam - mn) / (mx - mn + 1e-8)
            cams.append(cam)

        return cams, hm_np, (int(peak_y), int(peak_x))


# =========================================================
# 可视化工具
# =========================================================

def _colorize(arr: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    return cv2.applyColorMap((arr * 255).clip(0, 255).astype(np.uint8), cmap)


def _overlay(img_bgr: np.ndarray, color_map: np.ndarray, alpha=0.5) -> np.ndarray:
    return cv2.addWeighted(img_bgr, 1 - alpha, color_map, alpha, 0)


def _gt_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img_bgr.astype(np.float32)
    green = np.array([0, 200, 0], dtype=np.float32)
    a = 0.4 * np.clip(mask, 0, 1)[..., None]
    return np.clip(out * (1 - a) + green * a, 0, 255).astype(np.uint8)


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(panel, text, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _draw_peak(panel: np.ndarray, peak_y: int, peak_x: int,
               hm_h: int, hm_w: int) -> np.ndarray:
    """在面板上标出顶点位置（坐标从热力图尺寸映射到面板尺寸）"""
    ph, pw = panel.shape[:2]
    px = int(round(peak_x / hm_w * pw))
    py = int(round(peak_y / hm_h * ph))
    cv2.drawMarker(panel, (px, py), (0, 255, 255),
                   cv2.MARKER_CROSS, markerSize=12, thickness=1, line_type=cv2.LINE_AA)
    return panel


# =========================================================
# 主循环
# =========================================================

def run(model, dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gcam = GradCAMParam(model)
    model.eval()

    ih, iw = input_size
    hm_h = ih // HM_STRIDE
    hm_w = iw // HM_STRIDE

    for idx in range(len(dataset)):
        img_t, _, _, _, _, meta = dataset[idx]
        x = img_t.unsqueeze(0).to(device)

        cams, hm_np, (peak_y, peak_x) = gcam(x)

        img_u8  = (img_t.numpy()[0] * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        gt_mask = build_gt_mask_from_meta(meta, input_size)

        hm_up = cv2.resize(hm_np, (iw, ih), interpolation=cv2.INTER_LINEAR)

        # 6 张面板：Original | Pred Heatmap | Width-CAM | Height-CAM | Thickness-CAM | GT
        p_orig = _label(img_bgr.copy(), "Original")
        p_hm   = _label(_overlay(img_bgr, _colorize(hm_up)), "Pred Heatmap")

        param_panels = []
        for ch, name in enumerate(PARAM_CH):
            cam_up = cv2.resize(cams[ch], (iw, ih), interpolation=cv2.INTER_LINEAR)
            panel  = _overlay(img_bgr, _colorize(cam_up))
            _draw_peak(panel, peak_y, peak_x, hm_h, hm_w)
            _label(panel, f"CAM-{name}")
            param_panels.append(panel)

        p_gt = _label(_gt_overlay(img_bgr, gt_mask), "GT")

        vis  = np.hstack([p_orig, p_hm] + param_panels + [p_gt])
        stem = os.path.splitext(meta["image_name"])[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_gradcam_param.png"), vis)
        print(f"  [{idx + 1:>3}/{len(dataset)}] {stem}  peak=({peak_y},{peak_x})")

    gcam.remove()
    print(f"\nSaved to: {out_dir}")


# =========================================================
# Entry point
# =========================================================

DEFAULT_CKPT = r"C:\Users\79152\Desktop\github\shuangquxian\ml\0409\0607-1_0608_0015\fold01\checkpoints\best_model.pth"


def main():
    ap = argparse.ArgumentParser(
        description="Grad-CAM (param head) visualization for HyperbolaNet"
    )
    ap.add_argument("--ckpt",    default=DEFAULT_CKPT,
                    help="path/to/best_model.pth")
    ap.add_argument("--out_dir", default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "gradcam_param_out"),
                    help="output directory (default: ./gradcam_param_out)")
    args = ap.parse_args()

    model = HyperbolaNet(in_ch=1, base_ch=32).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"Loaded checkpoint: {args.ckpt}")

    dataset = HyperbolaDataset(
        image_dir=data_sources[0]["image_dir"],
        annotation_json=data_sources[0]["annotation_json"],
        input_size=input_size,
        hm_stride=HM_STRIDE,
        sigma=HM_SIGMA,
    )
    print(f"Dataset: {len(dataset)} images  →  {args.out_dir}")

    run(model, dataset, args.out_dir)


if __name__ == "__main__":
    main()
