import os
import sys
import argparse
import importlib.util
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# =========================================================
# 加载 0607-2.py 里的模型和工具函数（文件名含连字符，用 importlib）
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
_heatmap_nms            = m._heatmap_nms

input_size   = m.input_size
HM_STRIDE    = m.HM_STRIDE
HM_SIGMA     = m.HM_SIGMA
HM_THRESH    = m.HM_THRESH
nms_kernel   = m.nms_kernel
device       = m.device
data_sources = m.data_sources


# =========================================================
# Grad-CAM
# =========================================================

class GradCAM:
    """
    bottleneck 输出做 Grad-CAM（HyperbolaNet 最深的特征层，256ch）。
    目标信号：预测热力图中最大值（最置信的顶点）。
    """

    def __init__(self, model: HyperbolaNet):
        self.model = model
        self._feats = None
        self._grads = None
        self._h_fwd = model.bottleneck.register_forward_hook(self._fwd_hook)
        self._h_bwd = model.bottleneck.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _, __, output):
        self._feats = output                  # (1, 128, hm_h, hm_w)

    def _bwd_hook(self, _, __, grad_output):
        self._grads = grad_output[0]          # (1, 128, hm_h, hm_w)

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()

    def __call__(self, x: torch.Tensor):
        """
        x: (1, 1, H, W) on device
        Returns
            cam   : (hm_h, hm_w) float32, 归一化到 0-1
            hm_np : (hm_h, hm_w) float32, sigmoid 热力图
        """
        self.model.zero_grad()
        hm_logit, _, _ = self.model(x)
        hm = torch.sigmoid(hm_logit)          # (1, 1, hm_h, hm_w)

        # 对热力图最大值求梯度
        hm.max().backward()

        # 每个通道的梯度做全局平均池化，得到权重
        weights = self._grads.mean(dim=(2, 3), keepdim=True)   # (1, 128, 1, 1)
        cam = (weights * self._feats).sum(dim=1, keepdim=True)  # (1, 1, hm_h, hm_w)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()

        mn, mx = cam.min(), cam.max()
        cam = (cam - mn) / (mx - mn + 1e-8)

        return cam, hm.squeeze().detach().cpu().numpy()


# =========================================================
# 可视化工具
# =========================================================

def _colorize(arr: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    """(H,W) float 0-1  →  (H,W,3) BGR uint8"""
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


# =========================================================
# 主循环
# =========================================================

def run(model, dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gcam = GradCAM(model)
    model.eval()

    ih, iw = input_size

    for idx in range(len(dataset)):
        img_t, _, _, _, _, meta = dataset[idx]
        x = img_t.unsqueeze(0).to(device)   # (1, 1, H, W)

        cam, hm_np = gcam(x)

        # Grad-CAM と heatmap を入力サイズにアップサンプル
        cam_up = cv2.resize(cam,   (iw, ih), interpolation=cv2.INTER_LINEAR)
        hm_up  = cv2.resize(hm_np, (iw, ih), interpolation=cv2.INTER_LINEAR)

        img_u8  = (img_t.numpy()[0] * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        gt_mask = build_gt_mask_from_meta(meta, input_size)

        # 4 张面板
        p_orig = _label(img_bgr.copy(),                                "Original")
        p_cam  = _label(_overlay(img_bgr, _colorize(cam_up)),          "Grad-CAM")
        p_hm   = _label(_overlay(img_bgr, _colorize(hm_up)),           "Pred Heatmap")
        p_gt   = _label(_gt_overlay(img_bgr, gt_mask),                 "GT")

        vis  = np.hstack([p_orig, p_cam, p_hm, p_gt])
        stem = os.path.splitext(meta["image_name"])[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_gradcam.png"), vis)
        print(f"  [{idx + 1:>3}/{len(dataset)}] {stem}")

    gcam.remove()
    print(f"\nSaved to: {out_dir}")


# =========================================================
# Entry point
# =========================================================

DEFAULT_CKPT = r"C:\Users\79152\Desktop\github\shuangquxian\ml\0409\0607-1_0608_0015\fold01\checkpoints\best_model.pth"


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM visualization for HyperbolaNet")
    ap.add_argument("--ckpt",    default=DEFAULT_CKPT,
                    help="path/to/best_model.pth")
    ap.add_argument("--out_dir", default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "gradcam_out"),
                    help="output directory (default: ./gradcam_out)")
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
