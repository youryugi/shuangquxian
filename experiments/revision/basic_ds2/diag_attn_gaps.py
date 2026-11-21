"""attention 预测不到的部分在哪里（只读缓存，不训练）。

    python diag_attn_gaps.py --exps hbatbox_aug_abs_unet hbatbox_aug_abs

读 runs_cv/bandbox_cache/<exp>/f<fold>_seed<k>/attn_test.npz，放大到 224×224，对每个有 band 的 GT 目标：
  1. 沿曲线的覆盖率：band 像素按 |x - x_vertex| / (span/2) 分 5 段（0 = 顶点，1 = 翼尖），每段被预测到的比例，
     以及该段的平均预测概率（看"弱信号"还是"没信号"）；
  2. 按 HBAT 标签（visibility / completeness / distortion）和 grouped 分层的覆盖率；
  3. 阈值 0.5 / 0.3 / 0.2 下的覆盖率和背景误报率（band 外像素被判成 band 的比例）；
  4. 粘连：同一张图里框重叠的两目标，各取自己 band 的顶点段，预测 mask 里两者是否落在同一个连通块；
     和真值 band 的并集做同样判断（真值本身交叉就算"真值也连"）。
输出 runs_cv/attn_gaps/summary.md、summary.json。
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
DETCMP = HERE.parent / "detector_comparison"
DATASETS = DETCMP / "datasets"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DETCMP))
sys.path.insert(0, str(REPO / "tools" / "annotation"))
import hbat  # noqa: E402
from diag_strata import HBAT_JSON, label_gt_boxes  # noqa: E402

CACHE = HERE / "runs_cv" / "bandbox_cache"
FOLDS, SEEDS = list(range(5)), list(range(10))
NBIN = 5
THRS = (0.5, 0.3, 0.2)


def instances(gt):
    """{(stem, j): (obj, band 224×224 bool, bin map int8（-1 = 非 band）)}；匹配方式同 diag_band_quality.instance_bands。"""
    ann = json.load(open(HBAT_JSON, encoding="utf-8"))
    out = {}
    xx = np.arange(224)[None, :].repeat(224, 0)
    for stem, item in gt.items():
        w, h = item.get("width", 224), item.get("height", 224)
        assert (w, h) == (224, 224), (stem, w, h)
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
            if bo is None or best > 0.06:
                continue
            m = np.asarray(hbat.band_mask(bo, (w, h))) > 0
            if m.sum() == 0:
                continue
            o = hbat.normalize_obj(bo)
            r = np.abs(xx - o["x_vertex"]) / max(1.0, o["span"] / 2.0)
            bins = np.minimum((r * NBIN).astype(int), NBIN - 1).astype(np.int8)
            bins[~m] = -1
            out[(stem, j)] = (g, m, bins)
    return out


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def same_component(mask, ma, mb):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    la, lb = set(np.unique(lab[ma & mask])) - {0}, set(np.unique(lab[mb & mask])) - {0}
    if not la or not lb:
        return None  # 有一个完全没画到，不算粘连
    return bool(la & lb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["hbatbox_aug_abs_unet", "hbatbox_aug_abs"])
    ap.add_argument("--out", default=str(HERE / "runs_cv" / "attn_gaps"))
    args = ap.parse_args()
    gt, fold_of = {}, {}
    for f in FOLDS:
        g = json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"]
        gt.update(g)
        fold_of.update({s: f for s in g})
    labels, _ = label_gt_boxes(gt)
    inst = instances(gt)
    by_stem = defaultdict(list)
    for (s, j) in inst:
        by_stem[s].append(j)
    pairs = [(s, a, b) for s, js in by_stem.items() for i, a in enumerate(js) for b in js[i + 1:] if overlap(inst[(s, a)][0], inst[(s, b)][0])]
    gt_band = {s: cv2.imread(str(DATASETS / f"ds2_f{fold_of[s]}_s100_hbatbox" / "masks_band" / "test" / f"{s}.png"), cv2.IMREAD_GRAYSCALE) > 127 for s in gt}
    apex = {k: v[2] == 0 for k, v in inst.items()}
    gt_conn = {(s, a, b): same_component(inst[(s, a)][1] | inst[(s, b)][1], apex[(s, a)], apex[(s, b)]) for s, a, b in pairs}

    res = {}
    for exp in args.exps:
        per_seed = []
        for seed in SEEDS:
            acc = defaultdict(list)
            bg = {t: [0, 0] for t in THRS}
            merge = {"pred_merged": 0, "pred_merged_gt_separate": 0, "gt_separate": 0, "n": 0, "missing": 0}
            for f in FOLDS:
                z = np.load(CACHE / exp / f"f{f}_seed{seed}" / "attn_test.npz")
                for stem in z.files:
                    P = cv2.resize(z[stem].astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
                    for t in THRS:
                        nb = ~gt_band[stem]
                        bg[t][0] += int(((P > t) & nb).sum()); bg[t][1] += int(nb.sum())
                    for j in by_stem.get(stem, []):
                        _, m, bins = inst[(stem, j)]
                        lab = labels[(stem, j)]
                        keys = [("all", "all")] + [(k, lab[k]) for k in ("visibility", "completeness", "distortion", "grouped", "width")]
                        for t in THRS:
                            c = float((P[m] > t).mean())
                            for k in keys:
                                acc[(f"cov@{t}",) + k].append(c)
                        for bi in range(NBIN):
                            sel = bins == bi
                            if sel.sum() == 0:
                                continue
                            acc[("bin_cov", str(bi), "all")].append(float((P[sel] > 0.5).mean()))
                            acc[("bin_prob", str(bi), "all")].append(float(P[sel].mean()))
                            acc[("bin_cov", str(bi), lab["grouped"])].append(float((P[sel] > 0.5).mean()))
                    for s, a, b in pairs:
                        if s != stem:
                            continue
                        merge["n"] += 1
                        r = same_component(P > 0.5, apex[(s, a)], apex[(s, b)])
                        if r is None:
                            merge["missing"] += 1
                            continue
                        sep = gt_conn[(s, a, b)] is False
                        merge["gt_separate"] += sep
                        merge["pred_merged"] += r
                        merge["pred_merged_gt_separate"] += (r and sep)
            per_seed.append({"acc": {"|".join(k): float(np.mean(v)) for k, v in acc.items()},
                             "n": {"|".join(k): len(v) for k, v in acc.items()},
                             "bg_fp": {str(t): bg[t][0] / bg[t][1] for t in THRS}, "merge": merge})
        res[exp] = per_seed

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    E = args.exps
    mean = lambda exp, key: np.mean([d["acc"].get(key, np.nan) for d in res[exp]])  # noqa: E731
    n0 = res[E[0]][0]["n"]
    L = [f"# attention 预测不到的部分（测试集 5 折合并，10 seeds 均值；{len(inst)} 个有 band 的目标，{len(pairs)} 对框重叠的目标）", ""]
    L += ["## 1. 沿曲线：顶点 → 翼尖（阈值 0.5 覆盖率 / 平均预测概率）", "", "| 段（|dx|/半宽） | n | " + " | ".join(f"{e} 覆盖 | 概率" for e in E) + " |", "|---|---|" + "---|---|" * len(E)]
    for bi in range(NBIN):
        row = f"| {bi / NBIN:.1f}–{(bi + 1) / NBIN:.1f} | {n0.get(f'bin_cov|{bi}|all', 0)} |"
        for e in E:
            row += f" {mean(e, f'bin_cov|{bi}|all'):.3f} | {mean(e, f'bin_prob|{bi}|all'):.3f} |"
        L.append(row)
    L += ["", "grouped / alone 分开（阈值 0.5 覆盖率，" + E[0] + "）：", "", "| 段 | grouped | alone |", "|---|---|---|"]
    gk = sorted({k.split("|")[2] for k in n0 if k.startswith("bin_cov|0|") and not k.endswith("|all")})
    for bi in range(NBIN):
        L.append(f"| {bi / NBIN:.1f}–{(bi + 1) / NBIN:.1f} | " + " | ".join(f"{mean(E[0], f'bin_cov|{bi}|{g}'):.3f}" for g in gk) + " |")
    L += ["", f"（列顺序：{gk}）", "", "## 2. 分层覆盖率（阈值 0.5 / 0.3 / 0.2）", "", "| 分层 | 取值 | n | " + " | ".join(f"{e} @0.5 / 0.3 / 0.2" for e in E) + " |", "|---|---|---|" + "---|" * len(E)]
    keys = sorted({k.split("|", 1)[1] for k in n0 if k.startswith("cov@0.5|")}, key=lambda s: (s != "all|all", s))
    for k in keys:
        a, v = k.split("|", 1)
        row = f"| {a} | {v} | {n0[f'cov@0.5|{k}']} |"
        for e in E:
            row += " " + " / ".join(f"{mean(e, f'cov@{t}|{k}'):.3f}" for t in THRS) + " |"
        L.append(row)
    L += ["", "背景误报率（band 外像素 > t 的比例）：", "", "| exp | " + " | ".join(f"t={t}" for t in THRS) + " |", "|---|" + "---|" * len(THRS)]
    for e in E:
        L.append(f"| {e} | " + " | ".join(f"{np.mean([d['bg_fp'][str(t)] for d in res[e]]):.4f}" for t in THRS) + " |")
    L += ["", "## 3. 相邻目标粘连（框重叠的目标对，阈值 0.5）", "", "| exp | 对数 | 有一条没画到 | 预测连成一片 | 其中真值本来分开 | 真值分开的对数 |", "|---|---|---|---|---|---|"]
    for e in E:
        g = lambda k: np.mean([d["merge"][k] for d in res[e]])  # noqa: E731
        L.append(f"| {e} | {g('n'):.0f} | {g('missing'):.1f} | {g('pred_merged'):.1f} | {g('pred_merged_gt_separate'):.1f} | {g('gt_separate'):.1f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
