"""实验 4：注意力机制对比汇总（替换投稿版 Table III）。只读已有结果，不训练。

    cd ../exp1_basic && python aggregate_cv.py && cd ../exp4_attention && python attn_compare.py

读 ../exp1_basic/runs_cv/cv_summary.json（5 折合并、每个 seed 一个数字）和各 run 的 run_info.json。
arm（全部 5 折 × 10 seeds，--workers 1，和实验 1 Round 18 配对初始化：backbone 和检测头逐参数相同，数据顺序和翻转相同）：
  baseline        hbatbox_flipfix_none
  SE / CBAM / Non-local / CA   hbatbox_flipfix_{se,cbam,nonlocal,coord}：无监督通用模块，插在 band 融合的同一位置
  unsup           hbatbox_flipfix_unsup：结构和 band 臂完全相同（attention 头 + concat 融合），attention 损失权重 0
  box-mask        hbatbox_flipfix_boxmask：attention 用填实的 HBAT 框监督
  band fusion     hbatbox_flipfix_abs：attention 用 HBAT band 监督（只看检测头输出）
  后处理（infer_bandbox.py build，参数在合并验证集上全局选）：*_scale（框绕中心等比放大 k），*_fitbox（沿 attention 图拟合双曲线定框），
  *_bandbox_nms（Voronoi + NMS）
表 1：每个 arm 的参数量、P / R / F1（mean ± std）、AP50
表 2：配对 ΔF1（95% CI，胜/平/负）
表 3：分层 recall（验证集选的阈值下）：weak、单翼、畸变、grouped
表 4：attention 图和 band 的像素 IoU（测试集，t = 0.5，56×56）
输出 results/summary.md、summary.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "exp1_basic"     # 训练、后处理、配对聚合都在实验 1 的管线里
sys.path.insert(0, str(EXP1))
from aggregate_cv import paired  # noqa: E402
from diag_strata import analyse_seed, label_gt_boxes, load_seed  # noqa: E402

RUNS = EXP1 / "runs_cv"
DATASETS = HERE.parent / "shared" / "datasets"
OUT = HERE / "results"
SEEDS, FOLDS = range(10), range(5)
P = "hbatbox_flipfix_"
GENERIC = (("se", "SE"), ("cbam", "CBAM"), ("nonlocal", "Non-local"), ("coord", "CA"))

# 表 1 的行：(exp, 名称, 说明)
ROWS = [(f"{P}none", "baseline", "无注意力"),
        *[(f"{P}{k}", n, "通用模块，无监督") for k, n in GENERIC],
        (f"{P}unsup", "attention 分支，不监督", "结构同 band 臂，λ = 0"),
        (f"{P}boxmask", "box-mask 监督", "attention 用填实的框监督"),
        (f"{P}abs", "**band 监督（本文，fusion）**", "attention 用 band 监督"),
        None,
        (f"{P}none_scale", "baseline + scale", "框等比放大 k"),
        *[(f"{P}{k}_scale", f"{n} + scale", "框等比放大 k") for k, n in GENERIC],
        (f"{P}unsup_bandbox_nms", "不监督 + Voronoi + NMS", "无监督 attention 图当 band"),
        (f"{P}unsup_fitbox", "不监督 + fit", "无监督 attention 图当 band"),
        (f"{P}boxmask_bandbox_nms", "box-mask + Voronoi + NMS", "box-mask 最好的后处理"),
        (f"{P}boxmask_fitbox", "box-mask + fit", ""),
        (f"{P}abs_fitbox", "**band + fit（本文）**", "实验 1 的主结果")]

STRATA = [("visibility", "weak", "weak"), ("completeness", "single", "单翼"), ("distortion", "deviates", "畸变"),
          ("grouped", "grouped(official box has >=2 bands)", "grouped")]
STRATA_EXPS = [f"{P}none", *[f"{P}{k}" for k, _ in GENERIC], f"{P}unsup", f"{P}abs", f"{P}abs_fitbox"]


def main():
    OUT.mkdir(exist_ok=True)
    cv = json.load(open(RUNS / "cv_summary.json", encoding="utf-8"))["per_exp"]

    def vals(exp, key):
        if exp not in cv or len(cv[exp]) < len(SEEDS):
            return None
        g = {"F1": lambda r: r["primary"]["test"]["F1"], "P": lambda r: r["primary"]["test"]["P"],
             "R": lambda r: r["primary"]["test"]["R"], "AP50": lambda r: r["primary"]["AP50"]}[key]
        return [g(cv[exp][str(s)]) for s in SEEDS]

    def params(exp):
        src = exp.split("_scale")[0].split("_fitbox")[0].split("_bandbox_nms")[0]
        p = RUNS / "ds2" / src / "f0_seed0" / "run_info.json"
        return json.load(open(p, encoding="utf-8"))["params_M"] if p.exists() else None

    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"

    def pfmt(p):
        sig = "" if p["ci95"][0] > 0 or p["ci95"][1] < 0 else "（n.s.）"
        return f"{p['delta_mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]，{p['wins']}/{p['ties']}/{p['losses']}{sig}"

    res = {"arms": {}, "pairs": {}, "strata": {}, "attn_iou": {}}
    L = ["# 实验 4：注意力机制对比（数据集 2，5 折合并 × 10 seeds）", "",
         "## 表 1：各 arm 的检测结果（mean ± std over 10 seeds；阈值在合并验证集上选）", "",
         "| 方法 | 说明 | 参数量 (M) | Precision | Recall | F1 | AP50 |", "|---|---|---|---|---|---|---|"]
    for row in ROWS:
        if row is None:
            L.append("| *加推理后处理* | | | | | | |"); continue
        exp, name, note = row
        f1 = vals(exp, "F1")
        if f1 is None:
            L.append(f"| {name} | {note} | — | — | — | 结果不全 | — |"); continue
        r = {k: vals(exp, k) for k in ("P", "R", "F1", "AP50")}
        res["arms"][exp] = {**r, "params_M": params(exp)}
        L.append(f"| {name} | {note} | {params(exp)} | {ms(r['P'])} | {ms(r['R'])} | {ms(r['F1'])} | {ms(r['AP50'])} |")

    def pair(a, b, label):
        fa, fb = vals(a, "F1"), vals(b, "F1")
        if fa is None or fb is None:
            return f"| {label} | 结果不全 | |"
        p, q = paired(fa, fb), paired(vals(a, "AP50"), vals(b, "AP50"))
        res["pairs"][f"{a}:{b}"] = {"F1": p, "AP50": q}
        return f"| {label} | {pfmt(p)} | {pfmt(q)} |"

    L += ["", "## 表 2：配对差（B − A，同一 seed 配对；95% bootstrap CI，胜/平/负）", "",
          "| A → B | ΔF1 | ΔAP50 |", "|---|---|---|", "| ***注意力模块本身（都不加后处理）*** | | |"]
    for k, n in GENERIC:
        L.append(pair(f"{P}none", f"{P}{k}", f"baseline → {n}"))
    L.append(pair(f"{P}none", f"{P}unsup", "baseline → attention 不监督"))
    L.append(pair(f"{P}none", f"{P}abs", "baseline → band 监督（fusion）"))
    for k, n in GENERIC:
        L.append(pair(f"{P}{k}", f"{P}abs", f"{n} → band 监督（fusion）"))
    L.append(pair(f"{P}unsup", f"{P}abs", "attention 不监督 → band 监督（同一结构，只差监督）"))
    L.append(pair(f"{P}boxmask", f"{P}abs", "box-mask → band（fusion）"))
    L.append("| ***完整方法（band + fit）对各模块*** | | |")
    for k, n in GENERIC:
        L.append(pair(f"{P}{k}", f"{P}abs_fitbox", f"{n} → band + fit"))
        L.append(pair(f"{P}{k}_scale", f"{P}abs_fitbox", f"{n} + scale → band + fit"))
    L.append(pair(f"{P}unsup_fitbox", f"{P}abs_fitbox", "不监督 + fit → band + fit（fit 是否需要 band 监督）"))
    L.append(pair(f"{P}unsup", f"{P}unsup_fitbox", "不监督 → 不监督 + fit"))

    # 表 3：分层 recall
    gt_all = {}
    for f in FOLDS:
        gt_all.update(json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["test"])
    labels, _ = label_gt_boxes(gt_all)
    keys = sorted({k for k, _, _ in STRATA})
    L += ["", "## 表 3：分层 recall（IoU 0.5，验证集选的阈值；10 seeds 均值）", "",
          "| 方法 | 全部 | " + " | ".join(f"{lab}（n）" for _, _, lab in STRATA) + " |", "|---" * (2 + len(STRATA)) + "|"]
    name_of = {r[0]: r[1] for r in ROWS if r}
    for exp in STRATA_EXPS:
        per = []
        for s in SEEDS:
            r = load_seed(exp, s, RUNS)
            if r is None:
                per = None; break
            per.append(analyse_seed(*r, labels, keys)["recall"])
        if per is None:
            L.append(f"| {name_of[exp]} | 结果不全 |"); continue
        cells, out = [], {}
        for k, v, lab in [("all", "all", "全部")] + STRATA:
            rec = [o.get(f"{k}|{v}", [0, 0]) for o in per]
            n = rec[0][1]
            m = float(np.mean([a / b for a, b in rec])) if n else float("nan")
            out[lab] = {"recall": m, "n": n}
            cells.append(f"{m:.3f}" + ("" if k == "all" else f"（{n}）"))
        res["strata"][exp] = out
        L.append(f"| {name_of[exp]} | " + " | ".join(cells) + " |")

    # 表 4：attention 图与 band 的重合（train_basic.py 训练结束时算的，测试集 56×56、阈值 0.5）
    L += ["", "## 表 4：attention 图与真实 band 的像素 IoU（测试集，56×56，t = 0.5；50 个 run 均值）", "",
          "| arm | band IoU | band 内外均值差 |", "|---|---|---|"]
    for exp, name in ((f"{P}unsup", "attention 不监督"), (f"{P}boxmask", "box-mask 监督"), (f"{P}abs", "band 监督")):
        iou, gap = [], []
        for f in FOLDS:
            for s in SEEDS:
                p = RUNS / "ds2" / exp / f"f{f}_seed{s}" / "run_info.json"
                if p.exists():
                    a = json.load(open(p, encoding="utf-8"))["attn_test"]
                    iou.append(a["attn_band_iou"] or 0.0); gap.append(a["attn_gap"] or 0.0)
        if iou:
            res["attn_iou"][exp] = {"band_iou": float(np.mean(iou)), "gap": float(np.mean(gap)), "n_runs": len(iou)}
            L.append(f"| {name} | {np.mean(iou):.3f} | {np.mean(gap):+.3f} |")

    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(res, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
