"""Round 3：basic_ds2 Round 15 的 hyperbola fit 定框用在 YOLO 的检测上（不重训，只用 CPU）。

问题：basic_ds2 里 band 形状唯一不可替代的证据是"真实 band + hyperbola fit"（grouped recall 比框回归高 0.10）。
换成强得多的 YOLO 检测以后，这条路还有没有空间？用预测的 band 能拿到多少？

    python diag_fit.py --jobs 8
    python ../basic_ds2/aggregate_cv.py --runs runs_cv --pairs y8s_pre_none:y8s_pre_none_gtfit ...

变体（检测来源 + band 来源，规则 = ../basic_ds2/fit_hyperbola.py，参数网格同 basic_ds2 infer_bandbox.py）：
    y8s_pre_none_nms      对照：baseline 检测只多一道 NMS（fit 变体都带 NMS 选项，排除"NMS 本身有用"）
    y8s_pre_none_gtfit    上限：baseline 检测 + 真实 band（masks_band）+ fit
    y8s_pre_band_fit      方法：band arm 的检测 + 它自己预测的 band（> t）+ fit
    y8s_pre_boxmask_fit   对照：boxmask arm 的检测 + 它预测的框掩码（> t）+ fit
参数在 10 个 seed 的合并验证 F1 均值上全局选一组；写成 runs_cv/ds2/<变体>/f<fold>_seed<k>/，aggregate_cv / diag_strata 直接可读。
网格敏感性写在 runs_cv/fit/sweep.md，选中的参数在 runs_cv/fit/chosen.json。
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DETCMP = HERE.parent / "detector_comparison"
sys.path.insert(0, str(DETCMP))
sys.path.insert(0, str(HERE.parent / "basic_ds2"))
from evaluate import box_iou, collect, evaluate_run, prf_at, select_conf  # noqa: E402
from fit_hyperbola import apply_fits, fit_image  # noqa: E402

DATASETS = DETCMP / "datasets"
RUNS = HERE / "runs_cv"
FOLDS = [0, 1, 2, 3, 4]
SEEDS = list(range(10))
T_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)     # 预测 band 图的二值化阈值（同 basic_ds2）
SEED_GRID = (0.1, 0.2, 0.3, 0.5)       # 只修正 score >= s_seed 的检测
MODES = ("replace", "union", "blend")
NMS_GRID = (None, 0.5, 0.7)
SCALE = (1.0, 1.0)                     # YOLO 的预测已经是 224×224 原图坐标

# 变体: (检测来源 exp, band 来源, 参数网格)；band 来源 "gt" = 真实 band，"pred" = 该 run 导出的 band_{phase}/*.png
VARIANTS = {
    "y8s_pre_none_nms":    ("y8s_pre_none",    None,   [(n,) for n in NMS_GRID]),
    "y8s_pre_none_gtfit":  ("y8s_pre_none",    "gt",   list(itertools.product(SEED_GRID, MODES, NMS_GRID))),
    "y8s_pre_band_fit":    ("y8s_pre_band",    "pred", list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID))),
    "y8s_pre_boxmask_fit": ("y8s_pre_boxmask", "pred", list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID))),
}


def run_dir(exp, fold, seed):
    return RUNS / "ds2" / exp / f"f{fold}_seed{seed}"


def data_dir(fold):
    return DATASETS / f"ds2_f{fold}_s100_hbatbox"


@lru_cache(maxsize=None)
def load_preds(exp, fold, seed, phase):
    """只读；调用方不能改返回的 dict / list。"""
    return json.load(open(run_dir(exp, fold, seed) / f"preds_{phase}.json", encoding="utf-8"))


def band_mask(src_exp, kind, fold, seed, phase, stem, t):
    if kind == "gt":
        m = cv2.imread(str(data_dir(fold) / "masks_band" / phase / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        return m > 127
    m = cv2.imread(str(run_dir(src_exp, fold, seed) / f"band_{phase}" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return m.astype(np.float32) / 255.0 > t


def _fit_job(job):
    """一个 (来源, band, fold, seed, phase, t) 组合：每张图里 score >= min(SEED_GRID) 的检测各拟合一次。"""
    src, kind, fold, seed, phase, t = job
    preds = load_preds(src, fold, seed, phase)
    gt = json.load(open(data_dir(fold) / "gt_boxes.json", encoding="utf-8"))[phase]
    out = {}
    for stem in gt:
        dets = preds.get(stem, [])
        out[stem] = fit_image(dets, band_mask(src, kind, fold, seed, phase, stem, t), SCALE, min(SEED_GRID))
    return job, out


def nms(dets, thr):
    if thr is None:
        return dets
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(box_iou(d, k) <= thr for k in keep):
            keep.append(d)
    return keep


def transform(variant, cfg, fits, fold, seed, phase):
    src, kind, _ = VARIANTS[variant]
    preds = load_preds(src, fold, seed, phase)
    *cfg, nms_thr = cfg
    out = {}
    for stem, dets in preds.items():
        if kind is None:
            d = dets
        else:
            t, s_seed, mode = (None, *cfg) if kind == "gt" else cfg
            d = apply_fits(dets, fits[(src, kind, fold, seed, phase, t)].get(stem, {}), SCALE, s_seed, mode)
        out[stem] = nms(d, nms_thr)
    return out


def pooled(preds_by_fold, gt_by_fold):
    sc, tp, n = [], [], 0
    for f in FOLDS:
        s, t, k = collect(preds_by_fold[f], gt_by_fold[f]); sc.append(s); tp.append(t); n += k
    return np.concatenate(sc), np.concatenate(tp), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    jobs = set()
    for v in args.variants:
        src, kind, _ = VARIANTS[v]
        if kind is None:
            continue
        for f, s, ph, t in itertools.product(FOLDS, SEEDS, ("val", "test"), (None,) if kind == "gt" else T_GRID):
            jobs.add((src, kind, f, s, ph, t))
    fits = {}
    if jobs:
        print(f"fitting: {len(jobs)} (source, band, fold, seed, phase, t) combinations, {args.jobs} processes", flush=True)
        with Pool(args.jobs) as pool:
            for i, (job, out) in enumerate(pool.imap_unordered(_fit_job, sorted(jobs, key=str)), 1):
                fits[job] = out
                if i % 100 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)

    gt_val = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["val"] for f in FOLDS}
    gt_test = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["test"] for f in FOLDS}
    out_dir = RUNS / "fit"
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen_path = out_dir / "chosen.json"
    chosen = json.load(open(chosen_path, encoding="utf-8")) if chosen_path.exists() else {}
    sweep = []
    for v in args.variants:
        src, kind, grid = VARIANTS[v]
        print(f"== {v} (source {src}, band {kind}, {len(grid)} configs)", flush=True)
        val_f1 = {c: [] for c in grid}; test_f1 = {c: [] for c in grid}
        for seed in SEEDS:
            for c in grid:
                pv = {f: transform(v, c, fits, f, seed, "val") for f in FOLDS}
                pt = {f: transform(v, c, fits, f, seed, "test") for f in FOLDS}
                sv, tv, nv = pooled(pv, gt_val); conf, f1v = select_conf(sv, tv, nv)
                st, tt, nt = pooled(pt, gt_test)
                val_f1[c].append(f1v); test_f1[c].append(prf_at(st, tt, nt, conf)["F1"])
        best = max(grid, key=lambda c: np.mean(val_f1[c]))
        chosen[v] = {"source_exp": src, "band": kind, "config": list(best), "config_names": "(t,) s_seed, mode, nms" if kind else "nms",
                     "val_F1_mean": float(np.mean(val_f1[best])), "test_F1_mean_at_chosen": float(np.mean(test_f1[best]))}
        sweep += [f"## {v}（检测 {src}，band {kind}）", "", f"选中（10 seed 合并验证 F1 均值最高）：{best}", "",
                  "| 参数 | 合并验证 F1 均值 | 测试 F1 均值 ± std（只看敏感性） |", "|---|---|---|"]
        for c in sorted(grid, key=lambda c: -np.mean(val_f1[c]))[:15]:
            sweep.append(f"| {c} | {np.mean(val_f1[c]):.3f} | {np.mean(test_f1[c]):.3f} ± {np.std(test_f1[c], ddof=1):.3f}{' **←**' if c == best else ''} |")
        sweep += ["", f"（只列验证 F1 最高的 15 组，共 {len(grid)} 组）", ""]
        for seed in SEEDS:
            for f in FOLDS:
                rd = run_dir(v, f, seed); rd.mkdir(parents=True, exist_ok=True)
                json.dump(transform(v, best, fits, f, seed, "val"), open(rd / "preds_val.json", "w"))
                json.dump(transform(v, best, fits, f, seed, "test"), open(rd / "preds_test.json", "w"))
                info = json.load(open(run_dir(src, f, seed) / "run_info.json", encoding="utf-8"))
                info.update(exp=v, source_exp=src, fit={"band": kind, "config": list(best),
                                                        "selection": "global on pooled val F1 over 10 seeds", "script": "diag_fit.py"})
                json.dump(info, open(rd / "run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                evaluate_run(rd, data_dir(f))
        print(f"  chosen {best}: val F1 {chosen[v]['val_F1_mean']:.3f}, test F1 {chosen[v]['test_F1_mean_at_chosen']:.3f}", flush=True)
    json.dump(chosen, open(chosen_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (out_dir / "sweep.md").write_text("# fit 变体的参数网格（diag_fit.py）\n\n" + "\n".join(sweep) + "\n", encoding="utf-8")
    print(json.dumps(chosen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
