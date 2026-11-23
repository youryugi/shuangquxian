"""第 24 轮：attention 损失按 visibility 加权（weak ×4、medium ×2）之后，weak / grouped 有没有变好、alone 有没有掉。只读结果。

    python aggregate_cv.py && python r24_compare.py

对比（都是 5 折合并 × 10 seeds，按 seed 配对）：
  band：hbatbox_flipfix_abs{,_fitbox,_bandbox_nms} → hbatbox_flipfix_abs_vis{,_fitbox,_bandbox_nms}
  box-mask 对照：hbatbox_flipfix_boxmask* → hbatbox_flipfix_boxmask_vis*
表 1：F1 和配对 Δ；表 2：分层 recall（all / weak / medium / clear / grouped / alone，验证集选的阈值）及配对 Δ；
表 3：预测 band 的覆盖率（t = 0.5）按 visibility，整条没画出来的比例，框重叠的目标对里整条没画出来的条数。
输出 runs_cv/r24/summary.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aggregate_cv import paired  # noqa: E402
from diag_attn_gaps import instances, overlap  # noqa: E402
from diag_strata import DATASETS, analyse_seed, label_gt_boxes, load_seed  # noqa: E402

RUNS = HERE / "runs_cv"
CACHE = RUNS / "bandbox_cache"
OUT = RUNS / "r24"
SEEDS, FOLDS = range(10), range(5)
P = "hbatbox_flipfix_"
STRATA = [("all", "all", "all"), ("visibility", "weak", "weak"), ("visibility", "medium", "medium"), ("visibility", "clear", "clear"),
          ("grouped", "grouped(official box has >=2 bands)", "grouped"), ("grouped", "alone", "alone")]


def pfmt(p):
    sig = "" if p["ci95"][0] > 0 or p["ci95"][1] < 0 else "（n.s.）"
    return f"{p['delta_mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]，{p['wins']}/{p['ties']}/{p['losses']}{sig}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv = json.load(open(RUNS / "cv_summary.json", encoding="utf-8"))["per_exp"]
    f1 = lambda e: [cv[e][str(s)]["primary"]["test"]["F1"] for s in SEEDS]  # noqa: E731
    L = ["# 第 24 轮：attention 损失按 visibility 加权（weak ×4、medium ×2，box-mask 对照同样加权）", "",
         "## 表 1：F1（5 折合并，10 seeds）", "", "| 臂 | 后处理 | 原 | visibility 加权 | Δ |", "|---|---|---|---|---|"]
    for arm in ("abs", "boxmask"):
        for post, name in (("", "fusion"), ("_bandbox_nms", "Voronoi + NMS"), ("_fitbox", "fit")):
            a, b = f"{P}{arm}{post}", f"{P}{arm}_vis{post}"
            if a in cv and b in cv:
                L.append(f"| {arm} | {name} | {np.mean(f1(a)):.3f} | {np.mean(f1(b)):.3f} | {pfmt(paired(f1(a), f1(b)))} |")
    a, b = f"{P}boxmask_vis_bandbox_nms", f"{P}abs_vis_fitbox"
    if a in cv and b in cv:
        L += ["", f"加权后 band + fit 对 box-mask（加权）最好的后处理（Voronoi + NMS）：{pfmt(paired(f1(a), f1(b)))}；"
                  f"对 baseline：{pfmt(paired(f1(P + 'none'), f1(b)))}"]

    gt_all = {}
    for f in FOLDS:
        gt_all.update(json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"])
    labels, _ = label_gt_boxes(gt_all)
    keys = ["visibility", "grouped"]

    def strata(exp):
        out = defaultdict(list)
        for s in SEEDS:
            r = analyse_seed(*load_seed(exp, s, RUNS), labels, keys)["recall"]
            for k, v, lab in STRATA:
                m, n = r.get(f"{k}|{v}", [0, 0])
                out[lab].append(m / n if n else np.nan)
        return out

    L += ["", "## 表 2：分层 recall（IoU 0.5，验证集选的阈值）", "",
          "| 系统 | " + " | ".join(lab for _, _, lab in STRATA) + " |", "|---|" + "---|" * len(STRATA)]
    S = {}
    for exp in (f"{P}none", f"{P}abs", f"{P}abs_vis", f"{P}abs_fitbox", f"{P}abs_vis_fitbox", f"{P}boxmask_bandbox_nms", f"{P}boxmask_vis_bandbox_nms"):
        if exp in cv:
            S[exp] = strata(exp)
            L.append(f"| {exp.replace(P, '')} | " + " | ".join(f"{np.mean(S[exp][lab]):.3f}" for _, _, lab in STRATA) + " |")
    for a, b in ((f"{P}abs_fitbox", f"{P}abs_vis_fitbox"), (f"{P}abs", f"{P}abs_vis")):
        if a in S and b in S:
            L.append(f"| Δ {b.replace(P, '')} − {a.replace(P, '')} | " + " | ".join(
                pfmt(paired(S[a][lab], S[b][lab])) for _, _, lab in STRATA) + " |")

    # 表 3：band 覆盖率
    inst = instances(gt_all)
    by_stem = defaultdict(list)
    for (s, j) in inst:
        by_stem[s].append(j)
    pairs = [(s, a, b) for s, js in by_stem.items() for i, a in enumerate(js) for b in js[i + 1:] if overlap(inst[(s, a)][0], inst[(s, b)][0])]
    L += ["", "## 表 3：预测 band（attention > 0.5）对每个目标自己 band 的覆盖率；括号内 = 整条没画出来的比例", "",
          "| 臂 | all | weak | medium | clear | grouped | 框重叠的 70 对里整条没画出来（条 / seed） |", "|---|---|---|---|---|---|---|"]
    for exp in (f"{P}abs", f"{P}abs_vis"):
        cov = defaultdict(list); zero = defaultdict(list); miss = []
        for s in SEEDS:
            c = {}
            for f in FOLDS:
                z = np.load(CACHE / exp / f"f{f}_seed{s}" / "attn_test.npz")
                for stem in z.files:
                    Pm = cv2.resize(z[stem].astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR) > 0.5
                    for j in by_stem.get(stem, []):
                        c[(stem, j)] = float(Pm[inst[(stem, j)][1]].mean())
            for g, fn in (("all", lambda l: True), ("weak", lambda l: l["visibility"] == "weak"), ("medium", lambda l: l["visibility"] == "medium"),
                          ("clear", lambda l: l["visibility"] == "clear"), ("grouped", lambda l: l["grouped"].startswith("grouped"))):
                ks = [k for k in c if fn(labels[k])]
                cov[g].append(np.mean([c[k] for k in ks])); zero[g].append(np.mean([c[k] == 0 for k in ks]))
            miss.append(sum(c[(st, a)] == 0 or c[(st, b)] == 0 for st, a, b in pairs))
        L.append(f"| {exp.replace(P, '')} | " + " | ".join(f"{np.mean(cov[g]):.3f}（{np.mean(zero[g]):.2f}）" for g in ("all", "weak", "medium", "clear", "grouped"))
                 + f" | {np.mean(miss):.1f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
