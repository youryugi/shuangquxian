"""band 在 224 / 56 / 28 分辨率下还剩多少，以及原 attention head（56×56）与 U-shaped decoder（224×224）的预测对比。

    python vis_band_resolution.py [--unet-run DIR] [--conv-run DIR] [--zoom K] [--out DIR]

1. 统计（不需要模型）：全部 5 折测试集的真实 band mask，
     keep56  = 原 head 的监督目标（INTER_AREA 缩到 56×56 后按 0.5 二值化，同 train_basic.PreparedDataset）放大回 224 后保留的 band 像素比例；
     keep28  = 同样平均到 28×28（检测头那一层）后 > 0.5 的格子保留的比例；
     mixed56 = 56×56 上 band 格子中"一半 band 一半背景"（0 < 覆盖率 < 1）的比例。
   → <out>/resolution_stats.json
2. 图（需要两个 final.pth，默认 fold 0 seed 0）：fold 0 测试集挑 4 张（缩小损失最大的单目标、中位数单目标、损失最大的多目标、中位数多目标），
   每张画 输入 / 真实 band / 56×56 监督目标（红 = 丢掉的）/ 28×28 平均 / 原 head 预测 / U-shaped decoder 预测。
   → <out>/band_resolution.png；第 K 张图翼尖附近 48×48 像素放大（叠 56×56 格子线）→ <out>/band_resolution_zoom.png

final.pth 不入库，只在跑那个实验的机器上有；别的机器上可以用 train_basic.py 按 run_info.json 的配置补训一个 run，再用 --unet-run 指过去。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))
import train_basic as tb  # noqa: E402

DS = HERE.parent / "shared" / "datasets"
RUNS = HERE / "runs_cv" / "ds2"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def gt_mask(fold, phase, stem):
    m = cv2.imread(str(DS / f"ds2_f{fold}_s100_hbatbox" / "masks_band" / phase / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST) > 127


def area(b, n):
    return cv2.resize(b.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)


def target56(b):
    """原 head 的监督目标：面积插值到 56×56，再按 0.5 二值化。"""
    return area(b, 56) > 0.5


def up(a, n=224):
    return cv2.resize(a.astype(np.float32), (n, n), interpolation=cv2.INTER_NEAREST)


def iou(p, g):
    return float((p & g).sum() / max((p | g).sum(), 1))


def resolution_stats():
    s = {"keep56": [], "keep28": [], "mixed56": []}
    for f in range(5):
        for p in sorted((DS / f"ds2_f{f}_s100_hbatbox" / "masks_band" / "test").glob("*.png")):
            b = gt_mask(f, "test", p.stem)
            if b.sum() == 0:
                continue
            s["keep56"].append((up(target56(b)) > 0.5)[b].mean())
            s["keep28"].append((up(area(b, 28) > 0.5) > 0.5)[b].mean())
            nz = area(b, 56); nz = nz[nz > 0]
            s["mixed56"].append((nz < 1).mean())
    out = {k: float(np.mean(v)) for k, v in s.items()}
    out["n_images"] = len(s["keep56"])
    return out


def load(run_dir, arch, stride):
    tb.configure_attn_stride(stride)
    m = tb.AttnBBoxNet(in_ch=1, base_ch=tb.BASE_CH, use_attn=True, fuse="concat", oracle=False, det_stride=8, attn_arch=arch).to(tb.device)
    m.load_state_dict(torch.load(Path(run_dir) / "final.pth", map_location=tb.device))
    m.eval()
    return m


@torch.no_grad()
def predict_fold0(conv_run, unet_run):
    tb.configure_det_stride(8)
    models = {"conv": (load(conv_run, "conv", 4), 4), "unet": (load(unet_run, "unet", 1), 1)}
    tb.configure_attn_stride(4)
    ds = tb.PreparedDataset(DS / "ds2_f0_s100_hbatbox", "test")
    rows = []
    for i in range(len(ds)):
        img, *_, meta = ds[i]
        b = gt_mask(0, "test", meta["stem"])
        A = {}
        for k, (m, s) in models.items():
            tb.configure_attn_stride(s)                     # forward 里的 pooling 按全局 ATTN_STRIDE 算
            A[k] = tb.predict(m, img.unsqueeze(0), meta, 0.001, 50)[1]
        rows.append(dict(stem=meta["stem"], img=img[0].numpy(), b=b, A=A, boxes=meta["boxes"], scale=meta["scale"], n=len(meta["boxes"]),
                         keep56=float((up(target56(b)) > 0.5)[b].mean()),
                         iou_c=iou(cv2.resize(A["conv"], (224, 224), interpolation=cv2.INTER_LINEAR) > 0.5, b),
                         iou_u=iou(A["unet"] > 0.5, b)))
    return rows


def draw_main(pick, path):
    cols = ["输入图像", "真实 band\n224×224", "原 head 的监督目标\n56×56（红 = 缩小后丢掉的）",
            "真实 band 平均到 28×28\n（检测头那一层）", "原 head 预测\n56×56", "U-shaped decoder 预测\n224×224"]
    fig, axes = plt.subplots(len(pick), 6, figsize=(17, 3.0 * len(pick) + 0.6))
    for r_i, r in enumerate(pick):
        b = r["b"]; t = up(target56(b)) > 0.5
        lost = np.zeros((224, 224, 3)); lost[t] = 1; lost[b & ~t] = (0.9, 0.15, 0.15)
        panels = [(r["img"], "gray"), (b, "gray"), (lost, None), (up(area(b, 28)), "gray"), (up(r["A"]["conv"]), "magma"), (r["A"]["unet"], "magma")]
        for c_i, (im, cm) in enumerate(panels):
            ax = axes[r_i, c_i]
            ax.imshow(im, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            if c_i in (4, 5):
                ax.contour(b, levels=[0.5], colors="#3ddc84", linewidths=0.6)
            if c_i == 0:
                sx, sy = r["scale"]
                for x1, y1, x2, y2 in r["boxes"]:
                    ax.add_patch(Rectangle((x1 * sx, y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy, fill=False, ec="#3ddc84", lw=0.8))
            ax.set_xticks([]); ax.set_yticks([])
            if r_i == 0:
                ax.set_title(cols[c_i], fontsize=10)
        axes[r_i, 0].set_ylabel(f"{r['stem']}（{r['n']} 个目标）", fontsize=10)
        axes[r_i, 2].set_xlabel(f"保留 {r['keep56']:.0%} 的 band 像素", fontsize=9)
        axes[r_i, 4].set_xlabel(f"band IoU {r['iou_c']:.2f}", fontsize=9)
        axes[r_i, 5].set_xlabel(f"band IoU {r['iou_u']:.2f}", fontsize=9)
    fig.suptitle("fold 0 测试图；绿线 = 真实 band 轮廓 / 真实框；预测图越亮 = attention 越高；band IoU 按阈值 0.5 在 224×224 上算", fontsize=10, y=0.995)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def draw_zoom(r, path):
    """翼尖（离 band 水平中心最远的像素）附近 48×48 像素，叠 56×56 的格子线（每 4 px 一格）。"""
    b = r["b"]; ys, xs = np.nonzero(b)
    k = np.argmax(np.abs(xs - xs.mean()))
    x0 = int(np.clip(xs[k] - 24, 0, 224 - 48)) // 4 * 4
    y0 = int(np.clip(ys[k] - 24, 0, 224 - 48)) // 4 * 4
    sl = (slice(y0, y0 + 48), slice(x0, x0 + 48))
    panels = [("输入图像", r["img"], "gray", False), ("真实 band 224×224", b.astype(float), "gray", False),
              ("原 head 监督目标 56×56", (up(target56(b)) > 0.5).astype(float), "gray", False),
              ("原 head 预测 56×56", up(r["A"]["conv"]), "magma", True), ("U-shaped decoder 预测 224×224", r["A"]["unet"], "magma", True)]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    for ax, (title, im, cm, contour) in zip(axes, panels):
        ax.imshow(im[sl], cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        for g in range(0, 49, 4):
            ax.axhline(g - 0.5, color="#4aa3ff", lw=0.3, alpha=0.6); ax.axvline(g - 0.5, color="#4aa3ff", lw=0.3, alpha=0.6)
        if contour:
            ax.contour(b[sl], levels=[0.5], colors="#3ddc84", linewidths=0.8)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{r['stem']} 翼尖附近放大（原图 x {x0}–{x0 + 48}, y {y0}–{y0 + 48}）；蓝色网格 = 56×56 的一个格子（4×4 像素）", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conv-run", default=str(RUNS / "hbatbox_aug_abs" / "f0_seed0"))
    ap.add_argument("--unet-run", default=str(RUNS / "hbatbox_aug_abs_unet" / "f0_seed0"))
    ap.add_argument("--zoom", type=int, default=1, help="放大第几张挑出来的图（0–3）；默认 1 = 中位数单目标，两个 head 都有预测")
    ap.add_argument("--out", default=str(HERE / "runs_cv" / "band_resolution"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    stats = resolution_stats()
    rows = predict_fold0(args.conv_run, args.unet_run)
    stats["fold0_seed0_test_band_iou"] = {"conv": float(np.mean([r["iou_c"] for r in rows])), "unet": float(np.mean([r["iou_u"] for r in rows]))}
    def rel(p):
        try:
            return Path(p).resolve().relative_to(HERE).as_posix()
        except ValueError:
            return "runs_cv 之外（按 run_info.json 配置补训的 run）"
    stats["conv_run"], stats["unet_run"] = rel(args.conv_run), rel(args.unet_run)
    json.dump(stats, open(out / "resolution_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))

    single = sorted([r for r in rows if r["n"] == 1], key=lambda r: r["keep56"])
    multi = sorted([r for r in rows if r["n"] >= 2], key=lambda r: r["keep56"])
    pick = [single[0], single[len(single) // 2], multi[0], multi[len(multi) // 2]]
    draw_main(pick, out / "band_resolution.png")
    draw_zoom(pick[args.zoom], out / "band_resolution_zoom.png")
    print("saved", out)


if __name__ == "__main__":
    main()
