"""第 11 轮：分层分析——attention 到底帮了哪一类目标？只读 runs_cv 里已有的预测，不训练。

    python diag_strata.py                                    # 默认 none / abs / oracle / abs_extent，配对 none→abs、none→oracle、none→abs_extent
    python diag_strata.py --exps hbatbox_aug_none hbatbox_aug_abs --pairs hbatbox_aug_none:hbatbox_aug_abs

协议与 aggregate_cv.py 相同：每个 seed 把 5 折的验证集预测合起来选一个 score threshold，再把 5 折的测试集预测合起来
（= 全部 131 张图、222 个 HBAT box）做 IoU 0.5 贪心匹配。然后按下面几种分层算 recall，并按 seed 做配对差值：

  目标级（HBAT 标签，来自 data/additional_utilities/hbat/annotations.json）
    visibility   clear / medium / weak            回波清晰度
    completeness both / single / apex             两翼都在 / 只有一翼 / 只剩顶点
    distortion   fit / deviates                   双曲线拟合是否贴合
    param_status ok / rough / unparam             参数化质量
    width        窄 / 中 / 宽（GT 框宽度三分位）    目标尺寸
  图像级
    n_boxes      图里 1 个目标 / ≥2 个目标
    grouped      该目标的 official box 是否还套着别的双曲线（scene-level 框）
    stratum      used / unused（旧论文 247 张增强集是否用过这张原图）

  定位质量（不分层）
    F1@IoU 0.3 / 0.5 / 0.7（同一个 threshold）；每个 GT 的"最佳 IoU"分布：命中(≥0.5) / 差一点(0.3–0.5) / 没检到(<0.3)；
    命中框的 宽度比 / 高度比（预测 / GT 的中位数）：attention 是否把框画大了。
    FP：按图像 n_boxes 分层的每图 FP 数。

输出 runs_cv/strata/summary.md、summary.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DETCMP = HERE.parent / "detector_comparison"
DATASETS = DETCMP / "datasets"
sys.path.insert(0, str(DETCMP))
sys.path.insert(0, str(REPO / "tools" / "annotation"))
from evaluate import box_iou, clip_preds, collect, select_conf  # noqa: E402
import hbat  # noqa: E402

HBAT_JSON = REPO / "data" / "additional_utilities" / "hbat" / "annotations.json"
FOLDS = [0, 1, 2, 3, 4]
IOU_LEVELS = (0.3, 0.5, 0.7)


# ───────────────────────── GT 框 ↔ HBAT 目标标签 ─────────────────────────
def label_gt_boxes(gt_test_all):
    """gt_test_all: {stem: item}（5 折测试集合并 = 131 张）。给每个 GT 框找到生成它的 HBAT 目标，返回 {(stem, j): labels}。"""
    ann = json.load(open(HBAT_JSON, encoding="utf-8"))
    labels = {}
    n_unknown = 0
    for stem, item in gt_test_all.items():
        objs = ann.get(item.get("band_key") or f"{stem}.jpg", [])
        w, h = item.get("width", 224), item.get("height", 224)
        off = item.get("boxes_official", [])
        gid_count = defaultdict(int)
        for o in objs:
            if o.get("gt_box_id") is not None:
                gid_count[o["gt_box_id"]] += 1
        cand = []
        for o in objs:
            b = hbat.hyperbola_to_bbox(o, (w, h))
            if b is not None:
                box = [b["x1"], b["y1"], round(b["x1"] + b["width"], 2), round(b["y1"] + b["height"], 2)]
            else:
                gid = o.get("gt_box_id")
                box = list(off[gid]) if isinstance(gid, int) and 0 <= gid < len(off) else None
            if box is not None:
                cand.append((box, o))
        for j, g in enumerate(item["boxes"]):
            best, bo = 1e9, None
            for box, o in cand:
                d = max(abs(a - b) for a, b in zip(box, g))
                if d < best:
                    best, bo = d, o
            if bo is None or best > 0.06:
                n_unknown += 1
                lab = {k: "unknown" for k in ("visibility", "completeness", "distortion", "param_status")}
                lab["grouped"] = "unknown"
            else:
                lab = {k: str(bo.get(k, "unknown")) for k in ("visibility", "completeness", "distortion", "param_status")}
                gid = bo.get("gt_box_id")
                lab["grouped"] = "grouped(official box has >=2 bands)" if gid_count.get(gid, 0) >= 2 else "alone"
            lab["n_boxes"] = "1 target in image" if len(item["boxes"]) == 1 else ">=2 targets in image"
            lab["stratum"] = str(item.get("stratum"))
            lab["width_px"] = g[2] - g[0]
            labels[(stem, j)] = lab
    # 宽度三分位（全部 GT 框）
    ws = np.array([v["width_px"] for v in labels.values()])
    q1, q2 = np.percentile(ws, [100 / 3, 200 / 3])
    for v in labels.values():
        v["width"] = f"narrow (<{q1:.0f}px)" if v["width_px"] < q1 else (f"wide (>={q2:.0f}px)" if v["width_px"] >= q2 else f"medium ({q1:.0f}-{q2:.0f}px)")
    return labels, n_unknown


# ───────────────────────── 匹配（带细节） ─────────────────────────
def match_detail(preds, gts, conf, iou_thr=0.5):
    """只用 score >= conf 的预测，按 score 降序贪心匹配。返回每个 GT 的 (匹配到的预测 or None, 匹配 IoU, 任意预测的最佳 IoU) 和 FP 数。"""
    kept = sorted([p for p in preds if p[4] >= conf], key=lambda p: -p[4])
    matched = [None] * len(gts)
    best_any = [max([box_iou(p, g) for p in kept], default=0.0) for g in gts]
    fp = 0
    for p in kept:
        bi, bj = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j] is not None:
                continue
            iou = box_iou(p, g)
            if iou > bi:
                bi, bj = iou, j
        if bj >= 0 and bi >= iou_thr:
            matched[bj] = (p, bi)
        else:
            fp += 1
    return matched, best_any, fp, len(kept)


def load_seed(exp, seed, runs):
    """一个 exp 的一个 seed：5 折合并的 val/test 预测与 GT。缺折返回 None。
    验证集按折分别保存（同一张图可能出现在多折的验证集里，要和 aggregate_cv.py 一样逐折拼接，不能去重）。"""
    val_folds, pt, gt = [], {}, {}
    for fold in FOLDS:
        rd = runs / "ds2" / exp / f"f{fold}_seed{seed}"
        if not (rd / "metrics.json").exists():
            return None
        ri = json.load(open(rd / "run_info.json", encoding="utf-8"))
        suffix = "_hbatbox" if ri.get("box_source") == "hbat" else ""
        g = json.load(open(DATASETS / f"ds2_f{fold}_s100{suffix}" / "gt_boxes.json", encoding="utf-8"))
        # 和 evaluate.collect 一样，预测框先裁到图内（只比较图片范围内的重叠）
        val_folds.append((clip_preds(json.load(open(rd / "preds_val.json", encoding="utf-8")), g["val"]), g["val"]))
        pt.update(clip_preds(json.load(open(rd / "preds_test.json", encoding="utf-8")), g["test"]))
        gt.update(g["test"])
    return val_folds, pt, gt


def pooled_val_conf(val_folds):
    """与 aggregate_cv.pooled_metrics 相同：各折 collect 结果拼接后选阈值。"""
    sv, tv, nv = [], [], 0
    for pv, gv in val_folds:
        s, t, n = collect(pv, gv); sv.append(s); tv.append(t); nv += n
    conf, _ = select_conf(np.concatenate(sv), np.concatenate(tv), nv)
    return conf


def analyse_seed(val_folds, pt, gt, labels, strata_keys):
    conf = pooled_val_conf(val_folds)
    out = {"conf": conf, "recall": {}, "f1_at_iou": {}, "best_iou_bins": {}, "size_ratio": {}, "fp_per_image": {}}
    hit = defaultdict(lambda: [0, 0])          # stratum value -> [matched, total]
    bins = {"hit(>=0.5)": 0, "near(0.3-0.5)": 0, "miss(<0.3)": 0}
    wr, hr = [], []
    fp_by = defaultdict(lambda: [0, 0])        # n_boxes stratum -> [fp, images]
    for stem, item in gt.items():
        matched, best_any, fp, _ = match_detail(pt.get(stem, []), item["boxes"], conf)
        key = "1 target in image" if len(item["boxes"]) == 1 else (">=2 targets in image" if item["boxes"] else "no target")
        fp_by[key][0] += fp; fp_by[key][1] += 1
        for j, g in enumerate(item["boxes"]):
            lab = labels[(stem, j)]
            ok = matched[j] is not None
            for k in strata_keys:
                hit[(k, lab[k])][0] += int(ok); hit[(k, lab[k])][1] += 1
            hit[("all", "all")][0] += int(ok); hit[("all", "all")][1] += 1
            b = best_any[j]
            bins["hit(>=0.5)" if b >= 0.5 else ("near(0.3-0.5)" if b >= 0.3 else "miss(<0.3)")] += 1
            if ok:
                p = matched[j][0]
                wr.append((p[2] - p[0]) / max(g[2] - g[0], 1e-6)); hr.append((p[3] - p[1]) / max(g[3] - g[1], 1e-6))
    out["recall"] = {f"{k}|{v}": [m, tot] for (k, v), (m, tot) in hit.items()}
    n_gt = hit[("all", "all")][1]
    out["best_iou_bins"] = {k: v / n_gt for k, v in bins.items()}
    out["size_ratio"] = {"width_median": float(np.median(wr)) if wr else None, "height_median": float(np.median(hr)) if hr else None,
                         "width_mean": float(np.mean(wr)) if wr else None, "height_mean": float(np.mean(hr)) if hr else None}
    out["fp_per_image"] = {k: v[0] / max(v[1], 1) for k, v in fp_by.items()}
    # F1 at several IoU levels with the same conf
    for thr in IOU_LEVELS:
        tp = fp = 0
        for stem, item in gt.items():
            matched, _, f, _ = match_detail(pt.get(stem, []), item["boxes"], conf, iou_thr=thr)
            tp += sum(m is not None for m in matched); fp += f
        fn = n_gt - tp
        p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
        out["f1_at_iou"][str(thr)] = {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-9)}
    return out


def paired(a, b, n_boot=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    return {"mean_A": float(a.mean()), "mean_B": float(b.mean()), "delta": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}


def fmt_pair(p, scale=1.0, unit=""):
    return (f"{p['mean_A'] * scale:.3f}{unit} → {p['mean_B'] * scale:.3f}{unit} | {p['delta'] * scale:+.3f} "
            f"[{p['ci95'][0] * scale:+.3f}, {p['ci95'][1] * scale:+.3f}] | {p['wins']}/{p['ties']}/{p['losses']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    ap.add_argument("--exps", nargs="+", default=["hbatbox_aug_none", "hbatbox_aug_abs", "hbatbox_aug_oracle", "hbatbox_aug_abs_extent"])
    ap.add_argument("--pairs", nargs="+", default=["hbatbox_aug_none:hbatbox_aug_abs", "hbatbox_aug_none:hbatbox_aug_oracle",
                                                   "hbatbox_aug_none:hbatbox_aug_abs_extent"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = Path(args.runs)
    out_dir = Path(args.out) if args.out else runs / "strata"
    out_dir.mkdir(parents=True, exist_ok=True)
    strata_keys = ["visibility", "completeness", "distortion", "param_status", "width", "n_boxes", "grouped", "stratum"]

    # 载入所有 exp × seed
    data = {}
    gt_all = None
    for exp in args.exps:
        for seed in args.seeds:
            r = load_seed(exp, seed, runs)
            if r is None:
                print(f"[skip] {exp} seed{seed}: 折不全"); continue
            data[(exp, seed)] = r
            gt_all = gt_all or r[2]
    labels, n_unknown = label_gt_boxes(gt_all)
    n_gt = len(labels)
    print(f"GT boxes {n_gt}（{n_unknown} 个没对上 HBAT 目标）")

    res = {}
    for (exp, seed), (val_folds, pt, gt) in data.items():
        res[(exp, seed)] = analyse_seed(val_folds, pt, gt, labels, strata_keys)

    lines = [f"# 第 11 轮：分层分析（5 折合并 = 131 张图、{n_gt} 个 HBAT box；每个 exp 10 个 seed，按 seed 配对）", ""]
    lines.append("threshold 在合并验证集上选，与 aggregate_cv.py 相同。recall = 该层里被 IoU≥0.5 匹配到的 GT 比例；Δ 为 B − A 的 seed 均值与 bootstrap 95% CI；胜/平/负按 seed。")
    lines.append("")
    summary = {"n_gt": n_gt, "pairs": {}, "per_exp": {}}

    # 每个 exp 的整体指标
    lines.append("## 整体（各 exp 的 10 个 seed 均值）"); lines.append("")
    lines.append("| exp | recall@0.5 | F1@IoU0.3 | F1@IoU0.5 | F1@IoU0.7 | GT 最佳 IoU：命中 / 差一点 / 没检到 | 命中框 宽度比 / 高度比（中位数） | FP/图：单目标图 / 多目标图 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for exp in args.exps:
        rs = [res[(exp, s)] for s in args.seeds if (exp, s) in res]
        if not rs:
            continue
        rec = np.mean([r["recall"]["all|all"][0] / r["recall"]["all|all"][1] for r in rs])
        f1s = [np.mean([r["f1_at_iou"][str(t)]["F1"] for r in rs]) for t in IOU_LEVELS]
        bins = [np.mean([r["best_iou_bins"][k] for r in rs]) for k in ("hit(>=0.5)", "near(0.3-0.5)", "miss(<0.3)")]
        wr = np.mean([r["size_ratio"]["width_median"] for r in rs]); hr = np.mean([r["size_ratio"]["height_median"] for r in rs])
        fp1 = np.mean([r["fp_per_image"].get("1 target in image", 0) for r in rs]); fp2 = np.mean([r["fp_per_image"].get(">=2 targets in image", 0) for r in rs])
        lines.append(f"| {exp} | {rec:.3f} | {f1s[0]:.3f} | {f1s[1]:.3f} | {f1s[2]:.3f} | {bins[0]:.2f} / {bins[1]:.2f} / {bins[2]:.2f} | {wr:.2f} / {hr:.2f} | {fp1:.2f} / {fp2:.2f} |")
        summary["per_exp"][exp] = {"recall": rec, "F1_at_iou": dict(zip(map(str, IOU_LEVELS), f1s)), "best_iou_bins": bins,
                                   "width_ratio": wr, "height_ratio": hr, "fp_per_image": [fp1, fp2]}
    lines.append("")

    # 配对：定位质量
    for pr in args.pairs:
        a_exp, b_exp = pr.split(":")
        seeds = [s for s in args.seeds if (a_exp, s) in res and (b_exp, s) in res]
        if not seeds:
            continue
        P = {}
        lines.append(f"## {a_exp} → {b_exp}（n = {len(seeds)} seeds）"); lines.append("")
        lines.append("### 定位质量"); lines.append("")
        lines.append("| 指标 | A → B | Δ [95% CI] | 胜/平/负 |"); lines.append("|---|---|---|---|")
        for thr in IOU_LEVELS:
            p = paired([res[(a_exp, s)]["f1_at_iou"][str(thr)]["F1"] for s in seeds], [res[(b_exp, s)]["f1_at_iou"][str(thr)]["F1"] for s in seeds])
            lines.append(f"| F1@IoU{thr} | {fmt_pair(p)} |"); P[f"F1@{thr}"] = p
        for k in ("hit(>=0.5)", "near(0.3-0.5)", "miss(<0.3)"):
            p = paired([res[(a_exp, s)]["best_iou_bins"][k] for s in seeds], [res[(b_exp, s)]["best_iou_bins"][k] for s in seeds])
            lines.append(f"| GT 最佳 IoU {k} 的比例 | {fmt_pair(p)} |"); P[f"bin {k}"] = p
        for k, name in (("width_median", "命中框宽度比（预测/GT）中位数"), ("height_median", "命中框高度比中位数")):
            p = paired([res[(a_exp, s)]["size_ratio"][k] for s in seeds], [res[(b_exp, s)]["size_ratio"][k] for s in seeds])
            lines.append(f"| {name} | {fmt_pair(p)} |"); P[name] = p
        for k in ("1 target in image", ">=2 targets in image"):
            p = paired([res[(a_exp, s)]["fp_per_image"].get(k, 0) for s in seeds], [res[(b_exp, s)]["fp_per_image"].get(k, 0) for s in seeds])
            lines.append(f"| FP/图，{k} | {fmt_pair(p)} |"); P[f"FP {k}"] = p
        lines.append("")
        # 分层 recall
        lines.append("### 分层 recall（IoU 0.5）"); lines.append("")
        lines.append("| 分层 | 取值 | n GT | recall A → B | Δ [95% CI] | 胜/平/负 | Δ 折成目标数 |"); lines.append("|---|---|---|---|---|---|---|")
        keys = [("all", "all")] + [(k, v) for k in strata_keys for v in sorted({kk.split("|")[1] for kk in res[(a_exp, seeds[0])]["recall"] if kk.split("|")[0] == k})]
        for k, v in keys:
            key = f"{k}|{v}"
            n = res[(a_exp, seeds[0])]["recall"][key][1]
            ra = [res[(a_exp, s)]["recall"][key][0] / n for s in seeds]; rb = [res[(b_exp, s)]["recall"][key][0] / n for s in seeds]
            p = paired(ra, rb)
            lines.append(f"| {k} | {v} | {n} | {p['mean_A']:.3f} → {p['mean_B']:.3f} | {p['delta']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['wins']}/{p['ties']}/{p['losses']} | {p['delta'] * n:+.1f} / {n} |")
            P[f"recall {key}"] = {**p, "n_gt": n}
        lines.append("")
        summary["pairs"][pr] = P

    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(txt, encoding="utf-8")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(txt)
    print(f"-> {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
