"""第 16 轮：预测的 band 画得怎么样（都在输入分辨率 224×224 上量，不同 attention 头可以直接比）。

    python diag_band_quality.py --exps hbatbox_aug_abs hbatbox_aug_abs_unet [--t 0.5]

读 runs_cv/bandbox_cache/<exp>/f<fold>_seed<k>/attn_test.npz（infer_bandbox.py infer 的缓存），放大到 224×224 后按 t 二值化，和真值比：
  * band IoU / precision / recall：5 折测试集（131 张图）像素合计，每个 seed 一个数；
  * 每个目标的 band 覆盖率（该目标自己的 band 像素里被预测到的比例），按 GT 框宽度三分位、是否成簇（official box 套着 ≥2 条）、
    图里目标数分层——回答"两翼有没有画到头""窄目标、成簇目标的 band 是不是最差"。
输出 runs_cv/band_quality/summary.md、summary.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHARED = HERE.parent / "shared"
DATASETS = SHARED / "datasets"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(REPO / "tools" / "annotation"))
import hbat  # noqa: E402
from diag_strata import HBAT_JSON, label_gt_boxes  # noqa: E402

CACHE = HERE / "runs_cv" / "bandbox_cache"
FOLDS, SEEDS = list(range(5)), list(range(10))
STRATA = ("width", "grouped", "n_boxes")


def instance_bands(gt_test_all):
    """{(stem, j): 该 GT 框对应目标的 band（224×224 bool）或 None}，匹配方式与 diag_strata.label_gt_boxes 相同。"""
    ann = json.load(open(HBAT_JSON, encoding="utf-8"))
    out = {}
    for stem, item in gt_test_all.items():
        w, h = item.get("width", 224), item.get("height", 224)
        cand = []
        for o in ann.get(item.get("band_key") or f"{stem}.jpg", []):
            b = hbat.hyperbola_to_bbox(o, (w, h))
            if b is not None:
                cand.append(([b["x1"], b["y1"], round(b["x1"] + b["width"], 2), round(b["y1"] + b["height"], 2)], o))
        for j, g in enumerate(item["boxes"]):
            best, bo = 1e9, None
            for box, o in cand:
                d = max(abs(a - b) for a, b in zip(box, g))
                if d < best:
                    best, bo = d, o
            out[(stem, j)] = (np.asarray(hbat.band_mask(bo, (w, h))) > 0) if (bo is not None and best <= 0.06) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["hbatbox_aug_abs", "hbatbox_aug_abs_unet"])
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--out", default=str(HERE / "runs_cv" / "band_quality"))
    args = ap.parse_args()
    gt, fold_of = {}, {}
    for f in FOLDS:
        g = json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"]
        gt.update(g)
        fold_of.update({s: f for s in g})
    labels, _ = label_gt_boxes(gt)
    inst = instance_bands(gt)
    gt_band = {s: cv2.imread(str(DATASETS / f"ds2_f{fold_of[s]}_s100_hbatbox" / "masks_band" / "test" / f"{s}.png"), cv2.IMREAD_GRAYSCALE) > 127 for s in gt}
    res = {}
    for exp in args.exps:
        per_seed = []
        for seed in SEEDS:
            inter = union = pred_n = gt_n = 0
            cov = defaultdict(list)
            for f in FOLDS:
                z = np.load(CACHE / exp / f"f{f}_seed{seed}" / "attn_test.npz")
                for stem in z.files:
                    A = z[stem].astype(np.float32)
                    p = cv2.resize(A, (224, 224), interpolation=cv2.INTER_LINEAR) > args.t
                    b = gt_band[stem]
                    inter += int((p & b).sum()); union += int((p | b).sum()); pred_n += int(p.sum()); gt_n += int(b.sum())
                    for j in range(len(gt[stem]["boxes"])):
                        m = inst[(stem, j)]
                        if m is None or m.sum() == 0:
                            continue
                        c = float((p & m).sum() / m.sum())
                        cov[("all", "all")].append(c)
                        for k in STRATA:
                            cov[(k, labels[(stem, j)][k])].append(c)
            per_seed.append({"iou": inter / max(union, 1), "precision": inter / max(pred_n, 1), "recall": inter / max(gt_n, 1),
                             "coverage": {f"{k}|{v}": float(np.mean(c)) for (k, v), c in cov.items()},
                             "n": {f"{k}|{v}": len(c) for (k, v), c in cov.items()}})
        res[exp] = per_seed
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lines = [f"# 预测 band 的质量（测试集，224×224，阈值 {args.t}；10 个 seed 的均值 ± std）", "",
             "| exp | band IoU | precision | recall |", "|---|---|---|---|"]
    for exp in args.exps:
        g = lambda k: np.array([d[k] for d in res[exp]])  # noqa: E731
        lines.append(f"| {exp} | {g('iou').mean():.3f} ± {g('iou').std(ddof=1):.3f} | {g('precision').mean():.3f} | {g('recall').mean():.3f} |")
    lines += ["", "## 每个目标的 band 覆盖率（该目标的 band 像素里被预测到的比例）", "", "| 分层 | 取值 | n | " + " | ".join(args.exps) + " |", "|---|---|---|" + "---|" * len(args.exps)]
    keys = sorted(res[args.exps[0]][0]["coverage"], key=lambda s: (s.split("|")[0] != "all", s))
    for key in keys:
        k, v = key.split("|", 1)
        row = f"| {k} | {v} | {res[args.exps[0]][0]['n'][key]} |"
        for exp in args.exps:
            row += f" {np.mean([d['coverage'][key] for d in res[exp]]):.3f} |"
        lines.append(row)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
