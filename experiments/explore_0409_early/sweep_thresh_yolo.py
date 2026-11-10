"""
YOLO 版 objectness 阈值扫描：加载已训练好的 yolo .pth（不重训），
对一组 HM_THRESH(objectness 阈值) 评估 P/R/F1。阈值是解码参数，改它无需重新训练。

用法：
  python sweep_thresh_yolo.py                                  # 自动找最新 yolo 目录，abs lam0.3
  python sweep_thresh_yolo.py --mode abs --lam 0.3
  python sweep_thresh_yolo.py --mode none
  python sweep_thresh_yolo.py --thresholds 0.3 0.5 0.7 0.9
"""
import os
import glob
import argparse

import numpy as np
import torch

import attn_cnn_yolo_final as M


def find_work(mode, lam, seeds):
    cands = sorted(glob.glob("attn_cnn_yolo_final_*"), reverse=True)
    for d in cands:
        if all(os.path.exists(os.path.join(d, f"seed{s}_{mode}_lam{lam}_final.pth")) for s in seeds):
            return d
    return None


def detect_fuse(state):
    return "concat" if any(k.startswith("fuse_conv") for k in state) else "gate"


def load_model(pth, use_attn):
    state = torch.load(pth, map_location=M.device)
    fuse = detect_fuse(state)
    model = M.AttnBBoxNet(in_ch=1, base_ch=M.BASE_CH, use_attn=use_attn, fuse=fuse).to(M.device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="abs", choices=["none", "abs", "soft"])
    ap.add_argument("--lam", default="0.3", help="模型文件名里的 lam 标记，如 0.3 / 0.5 / 1 / 3 / 5")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--work", default=None)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    args = ap.parse_args()

    work = args.work or find_work(args.mode, args.lam, args.seeds)
    if work is None or not os.path.isdir(work):
        raise SystemExit(f"找不到含 {args.mode} lam{args.lam} 全部 seed 的 yolo 目录，请用 --work 指定")

    use_attn = (args.mode != "none")
    full = M.AttnDataset(input_size=M.input_size, hm_stride=M.HM_STRIDE, sigma=M.HM_SIGMA)
    n = len(full)
    print(f"[yolo-sweep] work={work}  mode={args.mode}  lam={args.lam}  seeds={args.seeds}  n_total={n}")
    print(f"             max_det={M.max_det}  nms_iou_thr={M.NMS_IOU_THR}  bbox_iou_thr={M.BBOX_IOU_THR}  (70/0/30)")

    loaded = []
    for s in args.seeds:
        pth = os.path.join(work, f"seed{s}_{args.mode}_lam{args.lam}_final.pth")
        model = load_model(pth, use_attn)
        _, _, te = M.make_split(n, s, train_frac=0.70, val_frac=0.0)
        loaded.append((s, model, te))

    saved_thr = M.HM_THRESH
    print(f"\n{'objthr':>8}{'P':>16}{'R':>16}{'F1':>16}")
    print("-" * 56)
    rows = []
    for thr in args.thresholds:
        M.HM_THRESH = thr                       # yolo predict 读这个全局当 objectness 阈值
        Ps, Rs, F1s = [], [], []
        for s, model, te in loaded:
            m = M.evaluate(model, full, te)
            Ps.append(m["bbox_P"]); Rs.append(m["bbox_R"]); F1s.append(m["bbox_F1"])
        rows.append((thr, np.mean(Ps), np.std(Ps), np.mean(Rs), np.std(Rs), np.mean(F1s), np.std(F1s)))
        print(f"{thr:>8.2f}{np.mean(Ps):>8.4f}±{np.std(Ps):<7.4f}"
              f"{np.mean(Rs):>8.4f}±{np.std(Rs):<7.4f}"
              f"{np.mean(F1s):>8.4f}±{np.std(F1s):<7.4f}", flush=True)
    M.HM_THRESH = saved_thr

    best = max(rows, key=lambda r: r[5])
    print("-" * 56)
    print(f"最佳 F1：objthr={best[0]:.2f}  F1={best[5]:.4f}±{best[6]:.4f}  (P={best[1]:.4f}, R={best[3]:.4f})")
    print("注意：在 test 上挑阈值有泄漏，这里只看趋势。")


if __name__ == "__main__":
    main()
