"""从零训练 YOLOv8s（只翻转）用多少 epoch：只看 5 折合并的验证集，不读测试集。

    python pick_scr_epochs.py [--exps y8s_scr_flip_none_e100 y8s_scr_flip_none_e200 y8s_scr_flip_none_e300] [--seeds 0]

对每个候选：5 折验证集预测合并 → 验证集上 F1 最高的阈值和那个 F1、验证集 AP50。按验证集 F1 选，打平时取轮数少的。
输出 runs_cv/scr_epochs.md（选完之后把 run_cv.py 的 SCR_EPOCHS 改成选中的值）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
from evaluate import ap50, collect, select_conf  # noqa: E402

DATASETS = HERE.parents[1] / "shared" / "datasets"


def pooled_val(exp, seed, runs):
    import numpy as np
    S, T, N = [], [], 0
    for fold in range(5):
        rd = runs / "ds2" / exp / f"f{fold}_seed{seed}"
        gt = json.load(open(DATASETS / f"ds2_f{fold}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))["val"]
        pv = json.load(open(rd / "preds_val.json", encoding="utf-8"))
        s, t, n = collect(pv, gt)   # collect 自己会把预测框裁到图内
        S.append(s); T.append(t); N += n
    s, t = np.concatenate(S), np.concatenate(T)
    conf, f1 = select_conf(s, t, N)
    return dict(val_F1=f1, val_conf=conf, val_AP50=ap50(s, t, N), n_gt=N)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="+", default=[f"y8s_scr_flip_none_e{e}" for e in (100, 200, 300)])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    args = ap.parse_args()
    runs = Path(args.runs)
    rows = []
    for exp in args.exps:
        ep = int(exp.rsplit("_e", 1)[1])
        per = [pooled_val(exp, s, runs) for s in args.seeds]
        rows.append(dict(exp=exp, epochs=ep, val_F1=sum(r["val_F1"] for r in per) / len(per),
                         val_AP50=sum(r["val_AP50"] for r in per) / len(per),
                         val_conf=[r["val_conf"] for r in per], n_gt=per[0]["n_gt"]))
    best = max(rows, key=lambda r: (round(r["val_F1"], 4), -r["epochs"]))
    lines = ["# 从零训练 YOLOv8s（只翻转）的轮数：5 折合并验证集", "",
             f"seeds {args.seeds}；验证集目标 {rows[0]['n_gt']} 个；不读测试集。", "",
             "| exp | epoch | 验证集 F1（最优阈值） | 验证集 AP50 | 阈值 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['exp']} | {r['epochs']} | {r['val_F1']:.3f} | {r['val_AP50']:.3f} | "
                     f"{', '.join(f'{c:.2f}' for c in r['val_conf'])} |")
    lines += ["", f"**选中：{best['epochs']} epoch**（验证集 F1 最高；打平时取轮数少的）。"]
    out = runs / "scr_epochs.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
