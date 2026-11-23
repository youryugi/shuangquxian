"""rounds.md Round 6：用预测的带（注意力图）修正检测框的范围 —— 纯后处理，不重训。

    python refine_with_band.py infer     # 用 runs_cv/ds2/hbatbox_aug_abs 的 50 个 final.pth 重新推理，保存注意力图
    python refine_with_band.py refine    # 在注意力图上做框修正、在合并验证集上选参数（逐 seed / 全局两种）、按 README 附录 A 协议评测试集

做法（每张图）：
  1. 注意力图 56×56 → 双线性放大到输入分辨率，按阈值 t 二值化得到带区域 B；
  2. 以分数 ≥ s_seed 的检测为种子，把 B 的像素按"离哪个检测框中心最近"划分（Voronoi），切开相邻双曲线；
  3. 每个种子检测只保留分给它、且与原框相连的连通块，取外接框作为"带框"；带像素太少则保留原框；
  4. 新框 = replace（带框）/ union（原框 ∪ 带框）/ blend（两者坐标平均）；分数不变；分数 < s_seed 的检测不改。
  参数 (t, s_seed, mode) 只在 5 折合并验证集上按 F1 选；测试集不参与。
对照：oracle = 用真值带掩码（masks_band/<phase>/<stem>.png）走同一套流程，给出"带完全正确时"的上限。

输出：runs_cv/refine/hbatbox_aug_abs/f<fold>_seed<k>/attn_{val,test}.npz（注意力图，float16）
      runs_cv/refine/summary.json / summary.md（含固定参数敏感性表）、各 run 的 preds_test_refined.json / preds_test_oracle.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))
from train_basic import ATTN_STRIDE, INPUT_SIZE, AttnBBoxNet, BASE_CH, PreparedDataset, device, predict  # noqa: E402
from evaluate import ap50, collect, prf_at, select_conf  # noqa: E402
from aggregate_cv import paired  # noqa: E402

RUNS = HERE / "runs_cv" / "ds2"
OUT = HERE / "runs_cv" / "refine"
DATASETS = HERE.parent / "shared" / "datasets"
EXP_ABS, EXP_NONE = "hbatbox_aug_abs", "hbatbox_aug_none"
FOLDS, SEEDS = range(5), range(10)

T_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
SEED_GRID = (0.1, 0.2, 0.3, 0.5)
MODES = ("replace", "union", "blend")
MIN_PX = 30          # 带像素少于此数（输入分辨率）→ 保留原框


def data_dir(fold):
    return DATASETS / f"ds2_f{fold}_s100_hbatbox"


def run_dir(exp, fold, seed):
    return RUNS / exp / f"f{fold}_seed{seed}"


def out_dir(fold, seed):
    return OUT / EXP_ABS / f"f{fold}_seed{seed}"


# ───────────────────────── 推理：保存注意力图 ─────────────────────────
@torch.no_grad()
def infer():
    max_abs_diff = 0.0
    for fold, seed in itertools.product(FOLDS, SEEDS):
        rd = run_dir(EXP_ABS, fold, seed)
        info = json.load(open(rd / "run_info.json", encoding="utf-8"))
        model = AttnBBoxNet(in_ch=1, base_ch=BASE_CH, use_attn=True, fuse=info["fuse"]).to(device)
        model.load_state_dict(torch.load(rd / "final.pth", map_location=device))
        model.eval()
        od = out_dir(fold, seed); od.mkdir(parents=True, exist_ok=True)
        for phase in ("val", "test"):
            ds = PreparedDataset(data_dir(fold), phase)
            old = json.load(open(rd / f"preds_{phase}.json", encoding="utf-8"))
            attn, metas = {}, {}
            for i in range(len(ds)):
                img, _, _, _, _, _, meta = ds[i]
                dets, A = predict(model, img.unsqueeze(0), meta, info["args"]["pred_thresh"], info["args"]["pred_max_det"])
                attn[meta["stem"]] = A.astype(np.float16)
                metas[meta["stem"]] = {"orig_size": meta["orig_size"], "scale": meta["scale"]}
                o = np.array(old.get(meta["stem"], []), float).reshape(-1, 5); n = np.array(dets, float).reshape(-1, 5)
                if o.shape == n.shape and len(o):
                    max_abs_diff = max(max_abs_diff, float(np.abs(o - n).max()))
                elif o.shape != n.shape:
                    print(f"  [warn] {rd.name} {phase} {meta['stem']}: 重推理检测数 {len(n)} != 原 {len(o)}")
            np.savez_compressed(od / f"attn_{phase}.npz", **attn)
            json.dump(metas, open(od / f"meta_{phase}.json", "w"))
        print(f"f{fold} seed{seed} done", flush=True)
    print(f"重推理与原 preds 的最大坐标/分数差: {max_abs_diff:.2e}（应接近 0，说明模型加载正确）")


# ───────────────────────── 修正 ─────────────────────────
def refine_image(dets, band_bin, scale, s_seed, mode, min_px=MIN_PX):
    """dets: [[x1,y1,x2,y2,s],...] 原图坐标；band_bin: 输入分辨率 (H,W) bool；返回同格式的新 dets。"""
    if not dets or band_bin.sum() == 0:
        return [list(d) for d in dets]
    H, W = band_bin.shape
    sx, sy = scale
    boxes_in = np.array([[d[0] * sx, d[1] * sy, d[2] * sx, d[3] * sy] for d in dets], float)
    scores = np.array([d[4] for d in dets], float)
    seed_idx = np.where(scores >= s_seed)[0]
    if len(seed_idx) == 0:
        return [list(d) for d in dets]
    # Voronoi：带像素分给最近的种子中心
    cx = (boxes_in[seed_idx, 0] + boxes_in[seed_idx, 2]) / 2; cy = (boxes_in[seed_idx, 1] + boxes_in[seed_idx, 3]) / 2
    ys, xs = np.nonzero(band_bin)
    d2 = (xs[:, None] - cx[None, :]) ** 2 + (ys[:, None] - cy[None, :]) ** 2
    owner = np.argmin(d2, axis=1)
    out = [list(d) for d in dets]
    for k, di in enumerate(seed_idx):
        cell = np.zeros((H, W), np.uint8)
        sel = owner == k
        cell[ys[sel], xs[sel]] = 1
        if cell.sum() < min_px:
            continue
        n_lab, lab = cv2.connectedComponents(cell, connectivity=8)
        x1, y1, x2, y2 = boxes_in[di]
        xi1, yi1 = int(np.clip(np.floor(x1), 0, W - 1)), int(np.clip(np.floor(y1), 0, H - 1))
        xi2, yi2 = int(np.clip(np.ceil(x2), 0, W - 1)), int(np.clip(np.ceil(y2), 0, H - 1))
        touching = np.unique(lab[yi1:yi2 + 1, xi1:xi2 + 1]); touching = touching[touching > 0]
        if len(touching) == 0:
            continue
        region = np.isin(lab, touching)
        if region.sum() < min_px:
            continue
        ry, rx = np.nonzero(region)
        bx1, by1, bx2, by2 = rx.min(), ry.min(), rx.max() + 1, ry.max() + 1
        if mode == "replace":
            nb = (bx1, by1, bx2, by2)
        elif mode == "union":
            nb = (min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2))
        else:  # blend
            nb = ((x1 + bx1) / 2, (y1 + by1) / 2, (x2 + bx2) / 2, (y2 + by2) / 2)
        out[di] = [nb[0] / sx, nb[1] / sy, nb[2] / sx, nb[3] / sy, float(scores[di])]
    return out


def load_attn(fold, seed, phase):
    od = out_dir(fold, seed)
    z = np.load(od / f"attn_{phase}.npz")
    metas = json.load(open(od / f"meta_{phase}.json"))
    return {k: z[k].astype(np.float32) for k in z.files}, metas


def attn_to_band(A, t):
    H, W = INPUT_SIZE
    return cv2.resize(A, (W, H), interpolation=cv2.INTER_LINEAR) > t


def gt_band(fold, phase, stem):
    m = cv2.imread(str(data_dir(fold) / "masks_band" / phase / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    H, W = INPUT_SIZE
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m > 127


def refine_preds(preds, fold, phase, t, s_seed, mode, oracle=False, cache=None):
    """对一折一个 phase 的全部预测做修正。"""
    attn, metas = cache if cache is not None else load_attn(fold, seed=None, phase=phase)
    out = {}
    for stem, dets in preds.items():
        band = gt_band(fold, phase, stem) if oracle else attn_to_band(attn[stem], t)
        out[stem] = refine_image(dets, band, metas[stem]["scale"], s_seed, mode)
    return out


def pooled(pred_by_fold, phase_gt_by_fold):
    sc, tp, n = [], [], 0
    for f in FOLDS:
        s, t, k = collect(pred_by_fold[f], phase_gt_by_fold[f]); sc.append(s); tp.append(t); n += k
    return np.concatenate(sc), np.concatenate(tp), n


def evaluate_seed(preds_val_by_fold, preds_test_by_fold, gt_val, gt_test):
    sv, tv, nv = pooled(preds_val_by_fold, gt_val)
    st, tt, nt = pooled(preds_test_by_fold, gt_test)
    conf, f1v = select_conf(sv, tv, nv)
    m = prf_at(st, tt, nt, conf)
    return {"val_conf": float(conf), "val_F1": float(f1v), "F1": float(m["F1"]), "P": float(m["P"]), "R": float(m["R"]),
            "TP": int(m["TP"]), "FP": int(m["FP"]), "FN": int(m["FN"]), "AP50": float(ap50(st, tt, nt)),
            "oracle_F1": float(select_conf(st, tt, nt)[1])}


def _load_seed(seed):
    pv_none = {f: json.load(open(run_dir(EXP_NONE, f, seed) / "preds_val.json")) for f in FOLDS}
    pt_none = {f: json.load(open(run_dir(EXP_NONE, f, seed) / "preds_test.json")) for f in FOLDS}
    pv_abs = {f: json.load(open(run_dir(EXP_ABS, f, seed) / "preds_val.json")) for f in FOLDS}
    pt_abs = {f: json.load(open(run_dir(EXP_ABS, f, seed) / "preds_test.json")) for f in FOLDS}
    att_v = {f: load_attn(f, seed, "val") for f in FOLDS}
    att_t = {f: load_attn(f, seed, "test") for f in FOLDS}
    return pv_none, pt_none, pv_abs, pt_abs, att_v, att_t


def refine_all():
    gt_val = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["val"] for f in FOLDS}
    gt_test = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["test"] for f in FOLDS}
    CONFIGS = list(itertools.product(T_GRID, SEED_GRID, MODES))
    OCONFIGS = list(itertools.product(SEED_GRID, MODES))
    per_cfg = {c: [] for c in CONFIGS}        # 每个参数组 × seed 的 (val_F1, test metrics)
    per_ocfg = {c: [] for c in OCONFIGS}
    arms = {"none": [], "abs": []}
    cache = {}
    for seed in SEEDS:
        pv_none, pt_none, pv_abs, pt_abs, att_v, att_t = _load_seed(seed)
        cache[seed] = (pv_abs, pt_abs, att_v, att_t)
        arms["none"].append(evaluate_seed(pv_none, pt_none, gt_val, gt_test))
        arms["abs"].append(evaluate_seed(pv_abs, pt_abs, gt_val, gt_test))
        for t, s_seed, mode in CONFIGS:
            rv = {f: refine_preds(pv_abs[f], f, "val", t, s_seed, mode, cache=att_v[f]) for f in FOLDS}
            rt = {f: refine_preds(pt_abs[f], f, "test", t, s_seed, mode, cache=att_t[f]) for f in FOLDS}
            per_cfg[(t, s_seed, mode)].append(evaluate_seed(rv, rt, gt_val, gt_test))
        for s_seed, mode in OCONFIGS:
            ov = {f: refine_preds(pv_abs[f], f, "val", None, s_seed, mode, oracle=True, cache=att_v[f]) for f in FOLDS}
            ot = {f: refine_preds(pt_abs[f], f, "test", None, s_seed, mode, oracle=True, cache=att_t[f]) for f in FOLDS}
            per_ocfg[(s_seed, mode)].append(evaluate_seed(ov, ot, gt_val, gt_test))
        print(f"seed {seed} done", flush=True)

    # 臂 1：每个 seed 单独在合并验证集上选参
    arms["abs_refined_perseed"], chosen_perseed = [], []
    for i in SEEDS:
        c = max(CONFIGS, key=lambda c: per_cfg[c][i]["val_F1"])
        r = dict(per_cfg[c][i]); r.update(t=c[0], s_seed=c[1], mode=c[2]); arms["abs_refined_perseed"].append(r); chosen_perseed.append(c)
    # 臂 2：全部 seed 共用一组参数，按 10 seed 的合并验证 F1 均值选
    g = max(CONFIGS, key=lambda c: np.mean([x["val_F1"] for x in per_cfg[c]]))
    arms["abs_refined_global"] = [dict(x, t=g[0], s_seed=g[1], mode=g[2]) for x in per_cfg[g]]
    # 臂 3：oracle（真值带），每个 seed 在验证集上选 (s_seed, mode)
    arms["abs_oracle_gtband"], chosen_oracle = [], []
    for i in SEEDS:
        c = max(OCONFIGS, key=lambda c: per_ocfg[c][i]["val_F1"])
        r = dict(per_ocfg[c][i]); r.update(s_seed=c[0], mode=c[1]); arms["abs_oracle_gtband"].append(r); chosen_oracle.append(c)
    # 保存全局参数下的修正预测，供可视化 / 漏检诊断
    for seed in SEEDS:
        pv_abs, pt_abs, att_v, att_t = cache[seed]
        for f in FOLDS:
            od = out_dir(f, seed)
            json.dump(refine_preds(pt_abs[f], f, "test", g[0], g[1], g[2], cache=att_t[f]), open(od / "preds_test_refined.json", "w"))
            json.dump(refine_preds(pt_abs[f], f, "test", None, chosen_oracle[seed][0], chosen_oracle[seed][1], oracle=True, cache=att_t[f]),
                      open(od / "preds_test_oracle.json", "w"))

    def col(arm, k):
        return [x[k] for x in arms[arm]]
    summary = {"arms": arms, "paired": {}, "global_config": {"t": g[0], "s_seed": g[1], "mode": g[2]},
               "per_config": {f"{c}": {"val_F1_mean": float(np.mean([x['val_F1'] for x in v])), "test_F1_mean": float(np.mean([x['F1'] for x in v])),
                                       "test_F1_std": float(np.std([x['F1'] for x in v], ddof=1)), "test_R_mean": float(np.mean([x['R'] for x in v])),
                                       "test_AP50_mean": float(np.mean([x['AP50'] for x in v]))} for c, v in per_cfg.items()}}
    for a, b in [("abs", "abs_refined_global"), ("none", "abs_refined_global"), ("abs", "abs_refined_perseed"), ("none", "abs_refined_perseed"),
                 ("abs", "abs_oracle_gtband"), ("abs_refined_global", "abs_oracle_gtband"), ("none", "abs")]:
        summary["paired"][f"{a}->{b}"] = {k: paired(col(a, k), col(b, k)) for k in ("F1", "AP50", "R", "P", "oracle_F1")}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    lines = ["# 带修正框范围（后处理）：5 折合并、10 seed、合并验证集选阈值", "",
             f"全局参数（按 10 seed 合并验证 F1 均值选）：t={g[0]}, s_seed={g[1]}, mode={g[2]}", "",
             "| 臂 | F1 | P | R | AP50 | 测试最优阈值 F1 |", "|---|---|---|---|---|---|"]
    for arm in arms:
        f = lambda k: f"{np.mean(col(arm, k)):.3f} ± {np.std(col(arm, k), ddof=1):.3f}"
        lines.append(f"| {arm} | {f('F1')} | {f('P')} | {f('R')} | {f('AP50')} | {f('oracle_F1')} |")
    lines += ["", "| 配对 | ΔF1 [95% CI] | p | 胜/平/负 | ΔR | ΔAP50 [CI] |", "|---|---|---|---|---|---|"]
    for k, v in summary["paired"].items():
        d = v["F1"]; r = v["R"]; a = v["AP50"]
        lines.append(f"| {k} | {d['delta_mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {d['wilcoxon_p']:.3f} | "
                     f"{d['wins']}/{d['ties']}/{d['losses']} | {r['delta_mean']:+.3f} | {a['delta_mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] |")
    lines += ["", "逐 seed 选到的参数 (t, s_seed, mode)：" + ", ".join(map(str, chosen_perseed)),
              "", "oracle 臂逐 seed 选到的 (s_seed, mode)：" + ", ".join(map(str, chosen_oracle)),
              "", "## 固定参数敏感性（每组参数用于全部 10 seed；测试列只看敏感性，不用于选参）", "",
              "| t | s_seed | mode | 合并验证 F1 均值 | 测试 F1 (均值 ± std) | 测试 R | 测试 AP50 |", "|---|---|---|---|---|---|---|",
              f"| - | - | unrefined | {np.mean(col('abs', 'val_F1')):.3f} | {np.mean(col('abs', 'F1')):.3f} ± {np.std(col('abs', 'F1'), ddof=1):.3f} | {np.mean(col('abs', 'R')):.3f} | {np.mean(col('abs', 'AP50')):.3f} |"]
    for c in CONFIGS:
        v = summary["per_config"][f"{c}"]
        lines.append(f"| {c[0]} | {c[1]} | {c[2]} | {v['val_F1_mean']:.3f} | {v['test_F1_mean']:.3f} ± {v['test_F1_std']:.3f} | {v['test_R_mean']:.3f} | {v['test_AP50_mean']:.3f} |")
    base = np.mean(col("abs", "F1"))
    for mode in MODES:
        vals = [summary["per_config"][f"{c}"]["test_F1_mean"] for c in CONFIGS if c[2] == mode]
        lines.append(f"\n{mode}: {len(vals)} 组中测试 F1 高于未修正的 {sum(v > base for v in vals)} 组，范围 {min(vals):.3f}–{max(vals):.3f}")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:24] + lines[-3:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["infer", "refine"])
    args = ap.parse_args()
    infer() if args.stage == "infer" else refine_all()
