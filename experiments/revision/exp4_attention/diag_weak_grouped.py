"""为什么 band + fit 在 weak 和 grouped 目标上弱（只读已有结果，不训练）。

    python diag_weak_grouped.py

A. 漏检拆解：每个 GT 在验证集选的阈值下归为一类（10 seeds、5 折合并）
   hit        被匹配（IoU ≥ 0.5）
   taken      有阈值以上的框和它 IoU ≥ 0.5，但贪心匹配给了别的 GT（相邻目标抢走）
   near       阈值以上的框最好 IoU 在 0.3–0.5（找到了，框没画准）
   low_score  阈值以上没有 IoU ≥ 0.3 的框，但全部输出（score ≥ 0.001）里有 IoU ≥ 0.5 的（分数不够）
   none       全部输出里都没有 IoU ≥ 0.5 的框（检测器没有给出候选）
B. band arm 的预测 band（attention 图 > 0.5）在每个目标自己的 band 上的覆盖率，按 visibility × grouped 分层；
   覆盖率为 0 = 这条 band 整条没画出来。
C. 框重叠的目标对里"整条没画出来"的那一条，visibility 分布（和全部重叠目标的分布比）。
D. band + fit 的 recall，按"这条 band 有没有画出来"分开。
输出 results/weak_grouped.md、weak_grouped.json。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "exp1_basic"
sys.path.insert(0, str(EXP1))
from diag_attn_gaps import instances, overlap  # noqa: E402
from diag_strata import box_iou, label_gt_boxes, load_seed, match_detail, pooled_val_conf  # noqa: E402

RUNS = EXP1 / "runs_cv"
CACHE = RUNS / "bandbox_cache"
DATASETS = HERE.parent / "shared" / "datasets"
OUT = HERE / "results"
FOLDS, SEEDS = range(5), range(10)
P = "hbatbox_flipfix_"
ARMS = [("none", "baseline"), ("cbam", "CBAM"), ("nonlocal", "Non-local"), ("unsup", "不监督"),
        ("abs", "band fusion"), ("abs_fitbox", "band + fit")]
CATS = ("hit", "taken", "near", "low_score", "none")
GROUPS = [("all", lambda l: True), ("weak", lambda l: l["visibility"] == "weak"),
          ("medium", lambda l: l["visibility"] == "medium"), ("clear", lambda l: l["visibility"] == "clear"),
          ("grouped", lambda l: l["grouped"].startswith("grouped")), ("alone", lambda l: l["grouped"] == "alone"),
          ("grouped & 非 weak", lambda l: l["grouped"].startswith("grouped") and l["visibility"] != "weak")]


def classify(preds, gts, conf):
    matched, best_kept, _, _ = match_detail(preds, gts, conf)
    out = []
    for j, g in enumerate(gts):
        if matched[j] is not None:
            out.append("hit")
        elif best_kept[j] >= 0.5:
            out.append("taken")
        elif best_kept[j] >= 0.3:
            out.append("near")
        elif max([box_iou(p, g) for p in preds], default=0.0) >= 0.5:
            out.append("low_score")
        else:
            out.append("none")
    return out


def main():
    OUT.mkdir(exist_ok=True)
    gt_all, fold_of = {}, {}
    for f in FOLDS:
        g = json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"]
        gt_all.update(g); fold_of.update({s: f for s in g})
    labels, _ = label_gt_boxes(gt_all)
    res = {"A": {}, "B": {}, "C": {}, "D": {}}

    # ── A. 漏检拆解
    cat_of = {}   # (arm, seed) -> {(stem, j): cat}
    for key, _ in ARMS:
        cnt = defaultdict(lambda: defaultdict(list))   # group -> cat -> [比例 per seed]
        for s in SEEDS:
            val_folds, pt, gt = load_seed(P + key, s, RUNS)
            conf = pooled_val_conf(val_folds)
            cats = {}
            for stem, item in gt.items():
                for j, c in enumerate(classify(pt.get(stem, []), item["boxes"], conf)):
                    cats[(stem, j)] = c
            cat_of[(key, s)] = cats
            for gname, fn in GROUPS:
                ks = [k for k in cats if fn(labels[k])]
                for c in CATS:
                    cnt[gname][c].append(sum(cats[k] == c for k in ks) / max(len(ks), 1))
        res["A"][key] = {g: {c: float(np.mean(v)) for c, v in d.items()} for g, d in cnt.items()}

    # ── B / C / D. band arm 的预测 band
    inst = instances(gt_all)
    by_stem = defaultdict(list)
    for (s, j) in inst:
        by_stem[s].append(j)
    pairs = [(s, a, b) for s, js in by_stem.items() for i, a in enumerate(js) for b in js[i + 1:]
             if overlap(inst[(s, a)][0], inst[(s, b)][0])]
    in_pair = {(s, a) for s, a, _ in pairs} | {(s, b) for s, _, b in pairs}
    covB = defaultdict(list); zeroB = defaultdict(list)
    miss_vis, pair_vis = defaultdict(float), defaultdict(float)
    D = defaultdict(lambda: [0, 0])
    for s in SEEDS:
        cov = {}
        for f in FOLDS:
            z = np.load(CACHE / f"{P}abs" / f"f{f}_seed{s}" / "attn_test.npz")
            for stem in z.files:
                Pm = cv2.resize(z[stem].astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR) > 0.5
                for j in by_stem.get(stem, []):
                    cov[(stem, j)] = float(Pm[inst[(stem, j)][1]].mean())
        for gname, fn in GROUPS:
            ks = [k for k in cov if fn(labels[k])]
            covB[gname].append(float(np.mean([cov[k] for k in ks])))
            zeroB[gname].append(float(np.mean([cov[k] == 0 for k in ks])))
        for k in in_pair:
            pair_vis[labels[k]["visibility"]] += 1 / len(SEEDS)
        for st, a, b in pairs:
            for k in ((st, a), (st, b)):
                if cov[k] == 0:
                    miss_vis[labels[k]["visibility"]] += 1 / len(SEEDS)
        cats = cat_of[("abs_fitbox", s)]
        for k, c in cov.items():
            g = "grouped" if labels[k]["grouped"].startswith("grouped") else "alone"
            for kk in (("drawn" if c > 0 else "not drawn", g), ("drawn" if c > 0 else "not drawn", "all")):
                D[kk][0] += cats[k] == "hit"; D[kk][1] += 1
    res["B"] = {g: {"coverage": float(np.mean(covB[g])), "not_drawn": float(np.mean(zeroB[g]))} for g, _ in GROUPS}
    res["C"] = {"missing_in_pairs_by_visibility": dict(miss_vis), "pair_members_by_visibility": dict(pair_vis),
                "n_pairs": len(pairs)}
    res["D"] = {"|".join(k): {"recall": v[0] / v[1], "n_per_seed": v[1] / len(SEEDS)} for k, v in D.items()}

    # ── 输出
    L = ["# band + fit 在 weak / grouped 上为什么弱（数据集 2，5 折合并 × 10 seeds）", "",
         "## A. 漏检拆解（每类占该分层 GT 的比例，10 seeds 均值）", "",
         "hit = 检出；taken = 框给了相邻目标；near = 框没画准（IoU 0.3–0.5）；low_score = 有对的框但分数低于阈值；none = 检测器没有给出候选", ""]
    n_of = {g: sum(fn(labels[k]) for k in labels) for g, fn in GROUPS}
    for gname, _ in GROUPS:
        L += [f"**{gname}**（n = {n_of[gname]}）", "", "| arm | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
        for key, name in ARMS:
            L.append(f"| {name} | " + " | ".join(f"{res['A'][key][gname][c]:.3f}" for c in CATS) + " |")
        L.append("")
    L += ["## B. band arm 预测 band 的覆盖率（attention > 0.5，224×224）", "", "| 分层 | 平均覆盖率 | 整条没画出来的比例 |", "|---|---|---|"]
    for gname, _ in GROUPS:
        L.append(f"| {gname} | {res['B'][gname]['coverage']:.3f} | {res['B'][gname]['not_drawn']:.3f} |")
    L += ["", f"## C. 框重叠的目标对（{len(pairs)} 对）：整条没画出来的那一条是什么 visibility（每 seed 平均条数）", "",
          "| visibility | 整条没画出来 | 重叠对里的全部目标 | 没画出来的比例 |", "|---|---|---|---|"]
    for v in ("clear", "medium", "weak"):
        m, t = miss_vis.get(v, 0.0), pair_vis.get(v, 0.0)
        L.append(f"| {v} | {m:.1f} | {t:.0f} | {m / max(t, 1e-9):.3f} |")
    L += ["", "## D. band + fit 的 recall，按这条 band 有没有画出来", "", "| | 全部 | grouped | alone |", "|---|---|---|---|"]
    for d in ("drawn", "not drawn"):
        L.append(f"| {d} | " + " | ".join(f"{res['D'][f'{d}|{g}']['recall']:.3f}（{res['D'][f'{d}|{g}']['n_per_seed']:.0f}）"
                                          for g in ("all", "grouped", "alone")) + " |")
    (OUT / "weak_grouped.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(res, open(OUT / "weak_grouped.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
