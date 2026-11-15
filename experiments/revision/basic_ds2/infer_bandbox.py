"""第 12 轮：把"band → 框尺寸"做进推理，作为方法正式的输出（不重训，用 runs_cv 里已有的 final.pth）。

    python infer_bandbox.py infer    # 重新推理 none / abs / oracle 三个实验的 150 个 run，缓存检测与注意力图（runs_cv/bandbox_cache/，不入库）
    python infer_bandbox.py build    # 在合并验证集上全局选参，生成下面几个"新实验"的 run 目录（preds_*.json + metrics*.json），之后 aggregate_cv.py 直接配对

生成的实验（都写到 runs_cv/ds2/<exp>/f<fold>_seed<k>/，格式与 train_basic.py 的输出相同）：
    hbatbox_aug_abs_bandbox        方法：+attention 的检测 + 用预测 band 修正框范围（Round 6 的规则，参数 (t, s_seed, mode) 在合并验证集上全局选）
    hbatbox_aug_abs_gtbandbox      上限：+attention 的检测 + 用真实 band 修正（参数 (s_seed, mode) 验证集全局选）
    hbatbox_aug_oracle_gtbandbox   上限：oracle 模型（真实 band 融合）的检测 + 真实 band 修正
    hbatbox_aug_none_scale         对照：baseline 的框绕中心等比放大 k 倍（k 验证集全局选）——"框普遍画小了，随便放大就行"这个解释
    hbatbox_aug_abs_scale          对照：+attention 的框等比放大 k 倍

修正规则（refine_with_band.refine_image）：注意力图放大到输入分辨率、按 t 二值化得到 band；分数 ≥ s_seed 的检测为种子，
band 像素按最近种子划分（切开相邻双曲线）；每个种子只取与原框相连的连通块，取外接框；mode = union（原框 ∪ 带框，只扩不缩）/ replace / blend。
参数选择：每组参数对 10 个 seed 分别算"5 折合并验证集 F1"，取 10 个 seed 均值最大的一组，全部 seed 共用。测试集不参与。

输出：runs_cv/bandbox/sweep.md（各参数组的验证 / 测试 F1，测试列只看敏感性）、chosen.json。
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "detector_comparison"))
from train_basic import AttnBBoxNet, BASE_CH, PreparedDataset, configure_det_stride, device, predict  # noqa: E402
from refine_with_band import MODES, SEED_GRID, T_GRID, attn_to_band, gt_band, refine_image  # noqa: E402
from evaluate import collect, evaluate_run, prf_at, select_conf  # noqa: E402

RUNS = HERE / "runs_cv" / "ds2"
CACHE = HERE / "runs_cv" / "bandbox_cache"
OUT = HERE / "runs_cv" / "bandbox"
DATASETS = HERE.parent / "detector_comparison" / "datasets"
FOLDS, SEEDS = list(range(5)), list(range(10))
SRC_EXPS = ("hbatbox_aug_none", "hbatbox_aug_abs", "hbatbox_aug_oracle")
K_GRID = (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6)


def data_dir(fold):
    return DATASETS / f"ds2_f{fold}_s100_hbatbox"


def run_dir(exp, fold, seed):
    return RUNS / exp / f"f{fold}_seed{seed}"


def cache_dir(exp, fold, seed):
    return CACHE / exp / f"f{fold}_seed{seed}"


# ───────────────────────── 阶段 1：推理 ─────────────────────────
@torch.no_grad()
def infer():
    max_diff = {e: 0.0 for e in SRC_EXPS}
    for exp in SRC_EXPS:
        for fold, seed in itertools.product(FOLDS, SEEDS):
            rd = run_dir(exp, fold, seed)
            info = json.load(open(rd / "run_info.json", encoding="utf-8"))
            mode = info["mode"]
            configure_det_stride(info.get("det_stride", 8))
            model = AttnBBoxNet(in_ch=1, base_ch=BASE_CH, use_attn=(mode == "abs"), fuse=info["fuse"],
                                oracle=(mode == "oracle"), det_stride=info.get("det_stride", 8)).to(device)
            model.load_state_dict(torch.load(rd / "final.pth", map_location=device))
            model.eval()
            cd = cache_dir(exp, fold, seed); cd.mkdir(parents=True, exist_ok=True)
            for phase in ("val", "test"):
                ds = PreparedDataset(data_dir(fold), phase)
                old = json.load(open(rd / f"preds_{phase}.json", encoding="utf-8"))
                dets_all, attn, metas = {}, {}, {}
                for i in range(len(ds)):
                    img, _, _, _, _, band, meta = ds[i]
                    dets, A = predict(model, img.unsqueeze(0), meta, info["args"]["pred_thresh"], info["args"]["pred_max_det"],
                                      band.unsqueeze(0) if model.oracle else None)
                    dets_all[meta["stem"]] = dets
                    if A is not None:
                        attn[meta["stem"]] = A.astype(np.float16)
                    metas[meta["stem"]] = {"orig_size": meta["orig_size"], "scale": meta["scale"]}
                    o = np.array(old.get(meta["stem"], []), float).reshape(-1, 5); n = np.array(dets, float).reshape(-1, 5)
                    if o.shape == n.shape and len(o):
                        max_diff[exp] = max(max_diff[exp], float(np.abs(o - n).max()))
                    elif o.shape != n.shape:
                        print(f"  [warn] {exp} {rd.name} {phase} {meta['stem']}: 重推理检测数 {len(n)} != 原 {len(o)}", flush=True)
                json.dump(dets_all, open(cd / f"dets_{phase}.json", "w"))
                json.dump(metas, open(cd / f"meta_{phase}.json", "w"))
                if attn:
                    np.savez_compressed(cd / f"attn_{phase}.npz", **attn)
            print(f"{exp} f{fold} seed{seed} done", flush=True)
    print("重推理与原 preds 的最大坐标/分数差（应接近 0）:", {k: f"{v:.2e}" for k, v in max_diff.items()})


# ───────────────────────── 阶段 2：修正 + 选参 + 写 run 目录 ─────────────────────────
def load_cache(exp, fold, seed, phase):
    cd = cache_dir(exp, fold, seed)
    dets = json.load(open(cd / f"dets_{phase}.json"))
    metas = json.load(open(cd / f"meta_{phase}.json"))
    attn = None
    if (cd / f"attn_{phase}.npz").exists():
        z = np.load(cd / f"attn_{phase}.npz"); attn = {k: z[k].astype(np.float32) for k in z.files}
    return dets, metas, attn


def scale_dets(dets, k, orig_size):
    oh, ow = orig_size
    out = []
    for x1, y1, x2, y2, s in dets:
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * k, (y2 - y1) * k
        out.append([max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(ow), cx + w / 2), min(float(oh), cy + h / 2), s])
    return out


def transform_phase(variant, cfg, src, fold, seed, phase):
    """对一折一个 phase 的全部检测做变换，返回 {stem: dets}。"""
    dets, metas, attn = load_cache(src, fold, seed, phase)
    out = {}
    for stem, d in dets.items():
        if variant.endswith("_scale"):
            out[stem] = scale_dets(d, cfg[0], metas[stem]["orig_size"])
        elif variant.endswith("_gtbandbox"):
            s_seed, mode = cfg
            out[stem] = refine_image(d, gt_band(fold, phase, stem), metas[stem]["scale"], s_seed, mode)
        else:  # _bandbox：预测 band
            t, s_seed, mode = cfg
            out[stem] = refine_image(d, attn_to_band(attn[stem], t), metas[stem]["scale"], s_seed, mode)
    return out


def pooled_f1(preds_by_fold, gt_by_fold):
    sc, tp, n = [], [], 0
    for f in FOLDS:
        s, t, k = collect(preds_by_fold[f], gt_by_fold[f]); sc.append(s); tp.append(t); n += k
    return np.concatenate(sc), np.concatenate(tp), n


VARIANTS = {
    # 新实验名: (来源实验, 参数网格)
    "hbatbox_aug_abs_bandbox":      ("hbatbox_aug_abs",    list(itertools.product(T_GRID, SEED_GRID, MODES))),
    "hbatbox_aug_abs_gtbandbox":    ("hbatbox_aug_abs",    list(itertools.product(SEED_GRID, MODES))),
    "hbatbox_aug_oracle_gtbandbox": ("hbatbox_aug_oracle", list(itertools.product(SEED_GRID, MODES))),
    "hbatbox_aug_none_scale":       ("hbatbox_aug_none",   [(k,) for k in K_GRID]),
    "hbatbox_aug_abs_scale":        ("hbatbox_aug_abs",    [(k,) for k in K_GRID]),
}


def build(variants):
    gt_val = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["val"] for f in FOLDS}
    gt_test = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["test"] for f in FOLDS}
    OUT.mkdir(parents=True, exist_ok=True)
    chosen, sweep_lines = {}, []
    for variant in variants:
        src, grid = VARIANTS[variant]
        print(f"== {variant}  (source {src}, {len(grid)} configs)", flush=True)
        # 每组参数 × 每个 seed：合并验证 F1（选参用）、合并测试 F1（只看敏感性）
        val_f1 = {c: [] for c in grid}; test_f1 = {c: [] for c in grid}
        cache_preds = {}
        for seed in SEEDS:
            for c in grid:
                pv = {f: transform_phase(variant, c, src, f, seed, "val") for f in FOLDS}
                pt = {f: transform_phase(variant, c, src, f, seed, "test") for f in FOLDS}
                cache_preds[(seed, c)] = (pv, pt)
                sv, tv, nv = pooled_f1(pv, gt_val); conf, f1v = select_conf(sv, tv, nv)
                st, tt, nt = pooled_f1(pt, gt_test)
                val_f1[c].append(f1v); test_f1[c].append(prf_at(st, tt, nt, conf)["F1"])
            print(f"  seed {seed} done", flush=True)
        best = max(grid, key=lambda c: np.mean(val_f1[c]))
        chosen[variant] = {"source_exp": src, "config": list(best), "val_F1_mean": float(np.mean(val_f1[best])),
                           "test_F1_mean_at_chosen": float(np.mean(test_f1[best]))}
        sweep_lines += [f"## {variant}（来源 {src}）", "", f"选中（按 10 seed 合并验证 F1 均值）：{best}", "",
                        "| 参数 | 合并验证 F1 均值 | 测试 F1 均值 ± std（只看敏感性） |", "|---|---|---|"]
        for c in grid:
            mark = " **←**" if c == best else ""
            sweep_lines.append(f"| {c} | {np.mean(val_f1[c]):.3f} | {np.mean(test_f1[c]):.3f} ± {np.std(test_f1[c], ddof=1):.3f}{mark} |")
        sweep_lines.append("")
        # 写 run 目录（选中的参数）
        for seed in SEEDS:
            pv, pt = cache_preds[(seed, best)]
            for f in FOLDS:
                rd = run_dir(variant, f, seed); rd.mkdir(parents=True, exist_ok=True)
                json.dump(pv[f], open(rd / "preds_val.json", "w")); json.dump(pt[f], open(rd / "preds_test.json", "w"))
                info = json.load(open(run_dir(src, f, seed) / "run_info.json", encoding="utf-8"))
                info.update(exp=variant, source_exp=src, source_run=str(run_dir(src, f, seed).relative_to(HERE)),
                            bandbox={"variant": variant, "config": list(best), "selection": "global on pooled val F1 over 10 seeds"})
                json.dump(info, open(rd / "run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                shutil.copy(run_dir(src, f, seed) / "train_log.csv", rd / "train_log.csv")
                evaluate_run(rd, data_dir(f))
                for alt in ("official", "hbat"):
                    if (data_dir(f) / f"gt_boxes_{alt}.json").exists():
                        evaluate_run(rd, data_dir(f), gt_name=f"gt_boxes_{alt}.json", out_name=f"metrics_{alt}.json")
        print(f"  chosen {best}: val F1 {chosen[variant]['val_F1_mean']:.3f}, test F1 {chosen[variant]['test_F1_mean_at_chosen']:.3f}", flush=True)
    json.dump(chosen, open(OUT / "chosen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (OUT / "sweep.md").write_text("# 第 12 轮参数敏感性（选参只用验证列；测试列仅供查看）\n\n" + "\n".join(sweep_lines), encoding="utf-8")
    print(json.dumps(chosen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["infer", "build"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args()
    infer() if args.stage == "infer" else build(args.variants)
