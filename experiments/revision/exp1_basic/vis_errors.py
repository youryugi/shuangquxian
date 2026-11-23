"""把每张测试图上"预测 vs 真值"的差距画出来：哪些 GT 漏检（FN）、哪些预测是误检（FP）、哪些命中（TP）。

    python vis_errors.py                                   # 默认 seed 0，baseline vs 方法（hbatbox_aug_none / hbatbox_aug_abs_bandbox）
    python vis_errors.py --exps hbatbox_aug_none hbatbox_aug_abs hbatbox_aug_abs_bandbox --seed 3
    python vis_errors.py --only-errors                     # 只画至少有一个 FN / FP 的图

协议和 aggregate_cv.py / diag_strata.py 相同：5 折测试集合并（全部 131 张图、222 个 HBAT 框），阈值用该 seed 的
5 折合并验证集选出的阈值，IoU ≥ 0.5 贪心匹配。

输出 runs_cv/vis_errors/seed<k>/：
    <stem>.png     每个 exp 一个面板并排（放大 2 倍）。绿 = 命中的 GT；橙 = 漏检的 GT（FN）；蓝 = 命中的预测（带分数）；
                   红 = 误检的预测（FP，带分数和它到最近 GT 的 IoU）；青色底 = 预测的 band（有缓存的 exp 才画）。
    index.md       每张图的 TP / FP / FN 计数（按错误数排序，链接到 png）和总计
    errors.csv     每个 GT 一行（命中 / 漏检、最佳 IoU、匹配分数、HBAT 标签），每个 FP 一行
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "shared"))
from diag_strata import DATASETS, FOLDS, label_gt_boxes, load_seed, match_detail, pooled_val_conf  # noqa: E402
from refine_with_band import attn_to_band  # noqa: E402
from evaluate import box_iou  # noqa: E402

REPO = HERE.parents[2]
CACHE = HERE / "runs_cv" / "bandbox_cache"
SCALE = 2
C_GT_TP, C_GT_FN, C_PR_TP, C_PR_FP, C_BAND = (0, 200, 0), (0, 140, 255), (255, 140, 0), (0, 0, 255), (200, 200, 0)


def find_image(stem, item):
    p = REPO / item["file"]
    if p.exists():
        return p
    for fold in FOLDS:
        for suf in ("_hbatbox", ""):
            for ext in (".jpg", ".png", ".jpeg"):
                q = DATASETS / f"ds2_f{fold}_s100{suf}" / "images" / "test" / f"{stem}{ext}"
                if q.exists():
                    return q
    raise FileNotFoundError(stem)


def band_cache(exp, seed, runs):
    """返回 (fold -> (attn dict, meta dict), t)；没有缓存返回 None。bandbox 变体用其源实验的缓存。"""
    ri = json.load(open(runs / "ds2" / exp / f"f0_seed{seed}" / "run_info.json", encoding="utf-8"))
    src = ri.get("source_exp", exp)
    t = (ri.get("bandbox") or {}).get("config", [0.6])[0] if exp.endswith("_bandbox") else 0.6
    out = {}
    for fold in FOLDS:
        cd = CACHE / src / f"f{fold}_seed{seed}"
        if not (cd / "attn_test.npz").exists():
            return None
        z = np.load(cd / "attn_test.npz")
        out[fold] = ({k: z[k].astype(np.float32) for k in z.files}, json.load(open(cd / "meta_test.json")))
    return out, float(t)


def rect(img, box, color, thick):
    x1, y1, x2, y2 = [int(round(v * SCALE)) for v in box[:4]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    return x1, y1, x2, y2


def text(img, s, org, color, scale=0.45):
    x, y = org
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_panel(img_gray, gts, preds, conf, band, title):
    """img_gray: 原图灰度；返回 (面板 BGR, matched, best_any, fp_list)。"""
    h, w = img_gray.shape
    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if band is not None:
        b = cv2.resize(band.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        tint = base.copy(); tint[b] = (0.55 * base[b] + 0.45 * np.array(C_BAND)).astype(np.uint8)
        base = tint
    panel = cv2.resize(base, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    matched, best_any, _, _ = match_detail(preds, gts, conf)
    kept = sorted([p for p in preds if p[4] >= conf], key=lambda p: -p[4])
    tp_preds = {id(m[0]) for m in matched if m is not None}
    fps = []
    for j, g in enumerate(gts):
        if matched[j] is None:
            x1, y1, _, _ = rect(panel, g, C_GT_FN, 2)
            text(panel, f"FN .{int(round(best_any[j] * 100)):02d}", (x1 + 2, max(12, y1 - 4)), C_GT_FN, 0.4)
        else:
            rect(panel, g, C_GT_TP, 2)
    for p in kept:
        if id(p) in tp_preds:
            x1, y1, x2, y2 = rect(panel, p, C_PR_TP, 1)
            text(panel, f"{p[4]:.2f}", (x1 + 2, min(y2 - 4, h * SCALE - 4)), C_PR_TP, 0.4)
        else:
            bi = max([box_iou(p, g) for g in gts], default=0.0)
            x1, y1, x2, y2 = rect(panel, p, C_PR_FP, 1)
            text(panel, f"FP {p[4]:.2f}/.{int(round(bi * 100)):02d}", (x1 + 2, min(y2 - 4, h * SCALE - 4)), C_PR_FP, 0.4)
            fps.append((p, bi))
    n_tp = sum(m is not None for m in matched); n_fn = len(gts) - n_tp
    head = np.full((22, panel.shape[1], 3), 30, np.uint8)
    text(head, f"{title}  conf>={conf:.2f}  TP {n_tp}  FP {len(fps)}  FN {n_fn}", (4, 16), (255, 255, 255), 0.45)
    return np.concatenate([head, panel], axis=0), matched, best_any, fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="+", default=["hbatbox_aug_none", "hbatbox_aug_abs_bandbox"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    ap.add_argument("--out", default=None, help="默认 runs_cv/vis_errors/seed<k>")
    ap.add_argument("--only-errors", action="store_true", help="只画至少一个 exp 上有 FN 或 FP 的图")
    ap.add_argument("--no-band", action="store_true", help="不叠加预测的 band")
    a = ap.parse_args()
    runs = Path(a.runs); out = Path(a.out) if a.out else runs / "vis_errors" / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    data, confs, bands, gt_all = {}, {}, {}, None
    for exp in a.exps:
        r = load_seed(exp, a.seed, runs)
        if r is None:
            sys.exit(f"{exp} seed {a.seed}: 5 折不全")
        val_folds, pt, gt = r
        data[exp] = pt; confs[exp] = pooled_val_conf(val_folds); gt_all = gt
        bands[exp] = None if a.no_band else band_cache(exp, a.seed, runs)
    labels, _ = label_gt_boxes(gt_all)
    stem_fold = {}
    for fold in FOLDS:
        for suf in ("_hbatbox", ""):
            gp = DATASETS / f"ds2_f{fold}_s100{suf}" / "gt_boxes.json"
            if gp.exists():
                for s in json.load(open(gp, encoding="utf-8"))["test"]:
                    stem_fold.setdefault(s, fold)
                break

    rows, index = [], []
    totals = {e: [0, 0, 0] for e in a.exps}
    for stem in sorted(gt_all):
        item = gt_all[stem]; gts = item["boxes"]
        img = cv2.imread(str(find_image(stem, item)), cv2.IMREAD_GRAYSCALE)
        panels, counts = [], {}
        for exp in a.exps:
            band = None
            if bands[exp] is not None:
                cache, t = bands[exp]; attn, meta = cache[stem_fold[stem]]
                if stem in attn:
                    band = attn_to_band(attn[stem], t)
            panel, matched, best_any, fps = draw_panel(img, gts, data[exp].get(stem, []), confs[exp], band, exp)
            panels.append(panel)
            n_tp = sum(m is not None for m in matched); n_fn = len(gts) - n_tp
            counts[exp] = (n_tp, len(fps), n_fn)
            for k, v in zip(range(3), (n_tp, len(fps), n_fn)):
                totals[exp][k] += v
            for j, g in enumerate(gts):
                lab = labels.get((stem, j), {})
                m = matched[j]
                rows.append({"stem": stem, "exp": exp, "kind": "GT", "idx": j, "status": "TP" if m else "FN",
                             "box": " ".join(f"{v:.1f}" for v in g), "score": f"{m[0][4]:.3f}" if m else "",
                             "iou": f"{(m[1] if m else best_any[j]):.3f}", **{k: lab.get(k, "") for k in
                             ("visibility", "completeness", "distortion", "param_status", "grouped", "n_boxes", "width")}})
            for p, bi in fps:
                rows.append({"stem": stem, "exp": exp, "kind": "PRED", "idx": "", "status": "FP",
                             "box": " ".join(f"{v:.1f}" for v in p[:4]), "score": f"{p[4]:.3f}", "iou": f"{bi:.3f}"})
        n_err = sum(c[1] + c[2] for c in counts.values())
        if a.only_errors and n_err == 0:
            continue
        gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
        canvas = panels[0]
        for p in panels[1:]:
            canvas = np.concatenate([canvas, gap, p], axis=1)
        cv2.imwrite(str(out / f"{stem}.png"), canvas)
        index.append((n_err, stem, len(gts), counts))

    fields = ["stem", "exp", "kind", "idx", "status", "box", "score", "iou", "visibility", "completeness", "distortion",
              "param_status", "grouped", "n_boxes", "width"]
    with open(out / "errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    index.sort(key=lambda r: (-r[0], r[1]))
    L = [f"# 预测 vs 真值（seed {a.seed}，5 折测试集合并，IoU ≥ 0.5，阈值 = 该 seed 合并验证集选出）", "",
         "图例：绿 = 命中的 GT；橙 = 漏检 GT（FN .xx = 任意预测到它的最佳 IoU）；蓝 = 命中的预测（分数）；红 = 误检预测（FP 分数/.xx = 到最近 GT 的 IoU）；青色底 = 预测的 band。", "",
         "| exp | 阈值 | TP | FP | FN | precision | recall | F1 |", "|---|---|---|---|---|---|---|---|"]
    for e in a.exps:
        tp, fp, fn = totals[e]; P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
        L.append(f"| {e} | {confs[e]:.2f} | {tp} | {fp} | {fn} | {P:.3f} | {R:.3f} | {2 * P * R / max(1e-9, P + R):.3f} |")
    L += ["", f"按错误数（全部 exp 的 FP + FN）排序，共 {len(index)} 张图：", "",
          "| 图 | GT 数 | " + " | ".join(f"{e}<br>TP / FP / FN" for e in a.exps) + " |", "|---|---|" + "---|" * len(a.exps)]
    for n_err, stem, n_gt, counts in index:
        L.append(f"| [{stem}]({stem}.png) | {n_gt} | " + " | ".join(" / ".join(map(str, counts[e])) for e in a.exps) + " |")
    (out / "index.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {out}  ({len(index)} 张图, errors.csv {len(rows)} 行)")
    for e in a.exps:
        print(f"   {e:30s} conf {confs[e]:.2f}  TP {totals[e][0]}  FP {totals[e][1]}  FN {totals[e][2]}")


if __name__ == "__main__":
    main()
