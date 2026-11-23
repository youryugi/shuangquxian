"""第 19 轮：固定测试集 + 嵌套训练子集（回应 R2-6）。只读 runs_cv/cv_summary.json（先跑 aggregate_cv.py），不训练。

    python data_efficiency.py

每折的测试集、验证集在各比例下不变，只把训练集取 25 / 50 / 75 / 100%（25 ⊂ 50 ⊂ 75 ⊂ 100，各 arm、各 seed 同一批图）。
100% = 第 18 轮的 hbatbox_flipfix_*。表 1：各方法在各比例下的 F1（5 折合并，10 seed 均值 ± std）；
表 2：按 seed 配对的 Δ（95% CI，胜/平/负）。输出 runs_cv/data_efficiency/summary.md、summary.json。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_cv"
FRACS = (25, 50, 75, 100)
ROWS = [  # (显示名, arm 后缀)：arm 名 = hbatbox_flipfix_<arm>[_frac<p>]<后缀>
    ("baseline", "none", ""),
    ("scale control", "none", "_scale"),
    ("+attention（band fusion）", "abs", ""),
    ("box-mask fusion", "boxmask", ""),
    ("box-mask + Voronoi", "boxmask", "_bandbox"),
    ("box-mask + Voronoi + NMS", "boxmask", "_bandbox_nms"),
    ("box-mask + fit", "boxmask", "_fitbox"),
    ("band + Voronoi", "abs", "_bandbox"),
    ("**band + fit**", "abs", "_fitbox"),
    ("上限：真实 band + fit", "abs", "_gtfitbox"),
]
PAIRS = [  # (显示名, A, B)，用 ROWS 的 (arm, 后缀)
    ("baseline → band + fit", ("none", ""), ("abs", "_fitbox")),
    ("scale control → band + fit", ("none", "_scale"), ("abs", "_fitbox")),
    ("**box-mask + fit → band + fit**", ("boxmask", "_fitbox"), ("abs", "_fitbox")),
    ("**box-mask + Voronoi + NMS → band + fit**", ("boxmask", "_bandbox_nms"), ("abs", "_fitbox")),
    ("baseline → band fusion", ("none", ""), ("abs", "")),
    ("box-mask fusion → band fusion", ("boxmask", ""), ("abs", "")),
    ("band + fit → 真实 band + fit", ("abs", "_fitbox"), ("abs", "_gtfitbox")),
]


def name(arm, suffix, p):
    return f"hbatbox_flipfix_{arm}" + ("" if p == 100 else f"_frac{p}") + suffix


def main():
    cv = json.load(open(RUNS / "cv_summary.json", encoding="utf-8"))
    per_exp, pairs = cv["per_exp"], cv["pairs"]

    def f1s(exp):
        if exp not in per_exp:
            return None
        return [per_exp[exp][str(s)]["primary"]["test"]["F1"] for s in range(10) if str(s) in per_exp[exp]]

    out = {"table": {}, "pairs": {}}
    L = ["# 第 19 轮：固定测试集 + 嵌套训练子集（数据集 2，5 折合并，10 seeds）", "",
         "每折测试集 / 验证集在各比例下不变；训练集 25 / 50 / 75 / 100% ≈ 23 / 46 / 69 / 92 张（嵌套，各 arm、各 seed 同一批图）。"
         "后处理参数在每个比例的合并验证集上各自选。", "",
         "## F1（均值 ± std）", "", "| 方法 | " + " | ".join(f"{p}%" for p in FRACS) + " |", "|---|" + "---|" * len(FRACS)]
    for disp, arm, suf in ROWS:
        cells = []
        for p in FRACS:
            v = f1s(name(arm, suf, p))
            out["table"].setdefault(disp, {})[p] = v
            cells.append("—" if not v else f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}")
        L.append(f"| {disp} | " + " | ".join(cells) + " |")
    L += ["", "## 配对 ΔF1（B − A；95% CI；胜/平/负）", "", "| A → B | " + " | ".join(f"{p}%" for p in FRACS) + " |", "|---|" + "---|" * len(FRACS)]
    for disp, a, b in PAIRS:
        cells = []
        for p in FRACS:
            key = f"{name(*a, p)}:{name(*b, p)}"
            r = pairs.get(key, {}).get("F1")
            out["pairs"].setdefault(disp, {})[p] = r
            cells.append("—" if not r else f"{r['delta_mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]，{r['wins']}/{r['ties']}/{r['losses']}")
        L.append(f"| {disp} | " + " | ".join(cells) + " |")
    d = RUNS / "data_efficiency"; d.mkdir(exist_ok=True)
    json.dump(out, open(d / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (d / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
