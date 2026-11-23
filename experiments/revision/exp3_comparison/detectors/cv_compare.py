"""检测器对比表（数据集 2，5 折合并 × 10 seeds）：本文方法、YOLOv8s 各行、Faster R-CNN 放进同一个口径。

    python cv_compare.py            # -> cv_compare.md / cv_compare.json（本目录）

口径和 exp1_basic/aggregate_cv.py 一致：每个 (exp, seed) 把 5 折的验证集预测合并，选 F1 最高的阈值，
再在合并的测试集上算 P / R / F1；另报 AP50 和同一阈值下 IoU 0.7 的 F1。真值是 HBAT 框（gt_boxes.json）。
不同 run 目录（exp1_basic/runs_cv、exp3_comparison/yolo_band/runs_cv）直接读 preds，不依赖各自的 cv_summary.json。

配对统计只用在"同一套代码、同一个 seed 的初始化可以对上"的两行之间（band 效果、增广效果、预训练效果）；
本文方法和 YOLO 之间没有配对关系，给的是两组 seed 的独立 bootstrap 差。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
from evaluate import ap50, collect, prf_at, select_conf  # noqa: E402

REV = HERE.parents[1]          # experiments/revision
DATASETS = REV / "shared" / "datasets"
SEEDS = list(range(10))
FOLDS = [0, 1, 2, 3, 4]

# (key, 表里的名字, run 根目录, exp, 预训练, 增广)
ROWS = [
    ("ours_none", "本文 baseline（轻量 CNN，无 attention）", REV / "exp1_basic/runs_cv", "hbatbox_flipfix_none", "无", "左右翻转"),
    ("ours_band", "本文 + band attention", REV / "exp1_basic/runs_cv", "hbatbox_flipfix_abs", "无", "左右翻转"),
    ("y8s_scr_flip", "YOLOv8s 从零", REV / "exp3_comparison/yolo_band/runs_cv", "y8s_scr_flip_none", "无", "左右翻转"),
    ("y8s_scr_flip_band", "YOLOv8s 从零 + band 辅助监督", REV / "exp3_comparison/yolo_band/runs_cv", "y8s_scr_flip_band", "无", "左右翻转"),
    ("y8s_pre_flip", "YOLOv8s 预训练", REV / "exp3_comparison/yolo_band/runs_cv", "y8s_pre_flip_none", "COCO", "左右翻转"),
    ("frcnn_pre", "Faster R-CNN R50-FPN v2 预训练", REV / "exp3_comparison/yolo_band/runs_cv", "frcnn_pre", "COCO", "左右翻转"),
    ("y8s_pre_full", "参考：YOLOv8s 预训练、全套增广", REV / "exp3_comparison/yolo_band/runs_cv", "y8s_pre_none", "COCO", "全套"),
]
# 配对比较（A -> B，同一 seed）
PAIRED = [
    ("ours_none", "ours_band", "本文：band attention 的作用"),
    ("y8s_scr_flip", "y8s_scr_flip_band", "YOLOv8s 从零：band 辅助监督的作用"),
    ("y8s_scr_flip", "y8s_pre_flip", "YOLOv8s：预训练的作用（都只翻转）"),
    ("y8s_pre_flip", "y8s_pre_full", "YOLOv8s 预训练：全套增广的作用"),
]
# 独立比较（两组 seed 没有配对关系）
UNPAIRED = [
    ("ours_band", "y8s_scr_flip", "本文 + band vs YOLOv8s 从零（条件对齐）"),
    ("ours_none", "y8s_scr_flip", "本文 baseline vs YOLOv8s 从零（条件对齐）"),
]


def run_dirs(root, exp, seed):
    out = {}
    for f in FOLDS:
        rd = root / "ds2" / exp / f"f{f}_seed{seed}"
        if not (rd / "preds_test.json").exists():
            return None
        out[f] = rd
    return out


def pooled(rds):
    sv, tv, nv, st, tt, nt, st7, tt7 = [], [], 0, [], [], 0, [], []
    for f, rd in sorted(rds.items()):
        gt = json.load(open(DATASETS / f"ds2_f{f}_s100_hbatbox" / "gt_boxes.json", encoding="utf-8"))
        pv = json.load(open(rd / "preds_val.json", encoding="utf-8"))
        pt = json.load(open(rd / "preds_test.json", encoding="utf-8"))
        s, t, n = collect(pv, gt["val"]); sv.append(s); tv.append(t); nv += n
        s, t, n = collect(pt, gt["test"]); st.append(s); tt.append(t); nt += n
        s, t, _ = collect(pt, gt["test"], iou_thr=0.7); st7.append(s); tt7.append(t)
    sv, tv, st, tt = map(np.concatenate, (sv, tv, st, tt))
    st7, tt7 = np.concatenate(st7), np.concatenate(tt7)
    conf, _ = select_conf(sv, tv, nv)
    m = prf_at(st, tt, nt, conf)
    p, r, f1 = m["P"], m["R"], m["F1"]
    f1_07 = prf_at(st7, tt7, nt, conf)["F1"]
    return dict(conf=conf, P=p, R=r, F1=f1, F1_07=f1_07, AP50=ap50(st, tt, nt), n_test=nt)


# 轻量模型的 run_info 没有 GFLOPs：用 exp1_basic/train_basic.py 的 build_model 在 1×1×224×224 上
# 以 torch.utils.flop_counter 数出来（2026-11-21，CPU）。和 ultralytics / train_frcnn 一样按 2×MAC 计。
LIGHT_GFLOPS = {"hbatbox_flipfix_none": 6.272, "hbatbox_flipfix_abs": 6.838}


def model_size(root, exp):
    ri = json.load(open(root / "ds2" / exp / "f0_seed0" / "run_info.json", encoding="utf-8"))
    params = ri.get("params_M_inference", ri.get("params_M"))
    return params, ri.get("GFLOPs", LIGHT_GFLOPS.get(exp)), ri.get("imgsz")


def boot_paired(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_unpaired(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n_boot)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    res, meta = {}, {}
    for key, name, root, exp, pre, aug in ROWS:
        per = {}
        for s in SEEDS:
            rds = run_dirs(root, exp, s)
            if rds is not None:
                per[s] = pooled(rds)
        res[key] = per
        meta[key] = dict(name=name, exp=exp, root=str(root.relative_to(REV)), pretrained=pre, aug=aug,
                         size=model_size(root, exp) if per else (None, None, None))
        print(f"{key}: {len(per)} seeds", flush=True)

    def col(key, m):
        return np.array([res[key][s][m] for s in sorted(res[key])])

    L = ["# 检测器对比（数据集 2，5 折合并 × 10 seeds，HBAT 框）", "",
         "每个 seed：5 折验证集合并选阈值 → 合并测试集（222 个目标）。均值 ± 标准差（seed 间）。",
         "F1@0.7 用同一个阈值、IoU 0.7 判定。参数量是推理时的（band 辅助头不算）；GFLOPs 按各自的输入尺寸。", "",
         "| 方法 | 预训练 | 增广 | seeds | F1 | P | R | AP50 | F1@0.7 | 参数量 (M) | GFLOPs | 输入 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, *_ in ROWS:
        m = meta[key]
        n = len(res[key])
        if n == 0:
            L.append(f"| {m['name']} | {m['pretrained']} | {m['aug']} | 0 | 未跑完 | | | | | | | |")
            continue
        f = lambda k: f"{col(key, k).mean():.3f} ± {col(key, k).std(ddof=1) if n > 1 else 0:.3f}"
        g = lambda k: f"{col(key, k).mean():.3f}"
        params, gf, sz = m["size"]
        L.append(f"| {m['name']} | {m['pretrained']} | {m['aug']} | {n} | {f('F1')} | {g('P')} | {g('R')} | "
                 f"{f('AP50')} | {g('F1_07')} | {params if params is not None else '–'} | {gf if gf is not None else '–'} | {sz} |")
    L += ["", "## 配对比较（同一 seed，B − A）", "", "| 比较 | n | ΔF1 [95% CI] | 胜/平/负 | ΔAP50 [95% CI] |", "|---|---|---|---|---|"]
    out_pairs = {}
    for a, b, name in PAIRED:
        seeds = sorted(set(res[a]) & set(res[b]))
        if len(seeds) < 2:
            L.append(f"| {name} | {len(seeds)} | 未跑完 | | |"); continue
        d = np.array([res[b][s]["F1"] - res[a][s]["F1"] for s in seeds])
        dap = np.array([res[b][s]["AP50"] - res[a][s]["AP50"] for s in seeds])
        lo, hi = boot_paired(d); lo2, hi2 = boot_paired(dap)
        w, t, l = int((d > 1e-9).sum()), int((abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())
        L.append(f"| {name} | {len(seeds)} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {w}/{t}/{l} | {dap.mean():+.3f} [{lo2:+.3f}, {hi2:+.3f}] |")
        out_pairs[f"{a}:{b}"] = dict(n=len(seeds), dF1=float(d.mean()), ci=[lo, hi], wtl=[w, t, l], dAP50=float(dap.mean()), ci_AP50=[lo2, hi2])
    L += ["", "## 独立比较（两组 seed 没有配对关系，bootstrap 两组均值之差，B − A）", "",
          "| 比较 | ΔF1 [95% CI] | ΔAP50 [95% CI] |", "|---|---|---|"]
    for a, b, name in UNPAIRED:
        if len(res[a]) < 2 or len(res[b]) < 2:
            L.append(f"| {name} | 未跑完 | |"); continue
        fa, fb, aa, ab = col(a, "F1"), col(b, "F1"), col(a, "AP50"), col(b, "AP50")
        lo, hi = boot_unpaired(fa, fb); lo2, hi2 = boot_unpaired(aa, ab)
        L.append(f"| {name} | {fb.mean() - fa.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {ab.mean() - aa.mean():+.3f} [{lo2:+.3f}, {hi2:+.3f}] |")
        out_pairs[f"{a}~{b}"] = dict(dF1=float(fb.mean() - fa.mean()), ci=[lo, hi], dAP50=float(ab.mean() - aa.mean()), ci_AP50=[lo2, hi2])
    (HERE / "cv_compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(dict(rows=meta, per_seed={k: {str(s): v for s, v in r.items()} for k, r in res.items()}, comparisons=out_pairs),
              open(HERE / "cv_compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
