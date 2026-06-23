"""
阈值扫描：加载已训练好的 .pth（不重训），对一组 HM_THRESH 评估 P/R/F1。
因为 HM_THRESH 只是解码阈值，改它无需重新训练，只要重跑 evaluate。

用法：
  python sweep_thresh.py                                  # 自动找最新含 abs 五个 seed 的目录
  python sweep_thresh.py --mode abs --seeds 0 1 2 3 4
  python sweep_thresh.py --work attn_cnn_merged_final_0623_1142
  python sweep_thresh.py --thresholds 0.1 0.15 0.2 0.25 0.3 0.4
"""
import os
import glob
import argparse

import numpy as np
import torch

import attn_cnn_merged_final as M


def find_work(mode, seeds):
    """挑选最新的、对该 mode 含齐所有 seed 的 work 目录。"""
    cands = sorted(glob.glob("attn_cnn_merged_final_*"), reverse=True)
    for d in cands:
        if all(os.path.exists(os.path.join(d, f"seed{s}_{mode}_final.pth")) for s in seeds):
            return d
    return None


def detect_fuse(state):
    """从 state_dict 的 key 判断融合方式（concat 会有 fuse_conv.*）。"""
    return "concat" if any(k.startswith("fuse_conv") for k in state) else "gate"


def load_model(pth, use_attn):
    state = torch.load(pth, map_location=M.device)
    fuse = detect_fuse(state)
    model = M.AttnBBoxNet(in_ch=1, base_ch=M.BASE_CH, use_attn=use_attn, fuse=fuse).to(M.device)
    model.load_state_dict(state)
    model.eval()
    return model, fuse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="abs", choices=["none", "abs", "soft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--work", default=None, help="指定 work 目录；不填则自动找最新完整的")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    args = ap.parse_args()

    work = args.work or find_work(args.mode, args.seeds)
    if work is None or not os.path.isdir(work):
        raise SystemExit(f"找不到含 {args.mode} 全部 seed 的 work 目录，请用 --work 指定")

    use_attn = (args.mode != "none")
    full = M.AttnDataset(input_size=M.input_size, hm_stride=M.HM_STRIDE, sigma=M.HM_SIGMA)
    n = len(full)
    print(f"[sweep] work={work}  mode={args.mode}  seeds={args.seeds}  n_total={n}")
    print(f"        max_det={M.max_det}  nms_kernel={M.nms_kernel}  bbox_iou_thr={M.BBOX_IOU_THR}")

    # 预加载每个 seed 的模型 + 其 test 划分（与训练时同 seed/同 70/15/15）
    loaded = []
    fuse_seen = set()
    for s in args.seeds:
        pth = os.path.join(work, f"seed{s}_{args.mode}_final.pth")
        model, fuse = load_model(pth, use_attn)
        fuse_seen.add(fuse)
        _, _, te = M.make_split(n, s, train_frac=0.70, val_frac=0.15)
        loaded.append((s, model, te))
    print(f"        fuse={'/'.join(sorted(fuse_seen))}\n")

    saved_thr = M.HM_THRESH
    print(f"{'thresh':>8}{'P':>16}{'R':>16}{'F1':>16}")
    print("-" * 56)
    rows = []
    for thr in args.thresholds:
        M.HM_THRESH = thr                       # predict 内读这个全局，改它即可换阈值
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
    print(f"最佳 F1：thresh={best[0]:.2f}  F1={best[5]:.4f}±{best[6]:.4f}  "
          f"(P={best[1]:.4f}, R={best[3]:.4f})")
    print("注意：在 test 上挑阈值会有泄漏；正式选阈值应在 val 上做，这里只是看趋势。")


if __name__ == "__main__":
    main()
