"""第 12 轮：把"band → 框尺寸"做进推理，作为方法正式的输出（不重训，用 runs_cv 里已有的 final.pth）。

    python infer_bandbox.py infer    # 重新推理 SRC_EXPS 里各实验的 run，缓存检测与注意力图（runs_cv/bandbox_cache/，不入库）；已有缓存的跳过（--force 重做），--exps 只做指定实验
    python infer_bandbox.py build    # 在合并验证集上全局选参，生成下面几个"新实验"的 run 目录（preds_*.json + metrics*.json），之后 aggregate_cv.py 直接配对

生成的实验（都写到 runs_cv/ds2/<exp>/f<fold>_seed<k>/，格式与 train_basic.py 的输出相同）：
    hbatbox_aug_abs_bandbox        方法：+attention 的检测 + 用预测 band 修正框范围（Round 6 的规则，参数 (t, s_seed, mode) 在合并验证集上全局选）
    hbatbox_aug_abs_gtbandbox      上限：+attention 的检测 + 用真实 band 修正（参数 (s_seed, mode) 验证集全局选）
    hbatbox_aug_oracle_gtbandbox   上限：oracle 模型（真实 band 融合）的检测 + 真实 band 修正
    hbatbox_aug_none_scale         对照：baseline 的框绕中心等比放大 k 倍（k 验证集全局选）——"框普遍画小了，随便放大就行"这个解释
    hbatbox_aug_abs_scale          对照：+attention 的框等比放大 k 倍
    hbatbox_aug_boxmask_bandbox    对照（第 13 轮）：attention 头用填实的 HBAT 框当监督（masks_box，没有 band 形状），检测 + 用预测的框掩码做同样的 mask → 框步骤。
                                   回答"收益来自稠密头估尺寸，还是来自双曲线 band 的形状"
    第 15 轮（fit_hyperbola.py）：band 的归属不再按"离哪个框中心最近"切，而是给每个检测沿 band 拟合一条双曲线（顶点 + slope），取被 band 支持的那一段。
    hbatbox_aug_abs_fitbox         方法：+attention 的检测 + 预测 band + 双曲线拟合定框（参数网格与 *_bandbox 相同）
    hbatbox_aug_abs_gtfitbox       上限：+attention 的检测 + 真实 band + 双曲线拟合定框
    hbatbox_aug_oracle_gtfitbox    上限：oracle 模型的检测 + 真实 band + 双曲线拟合定框
    hbatbox_aug_boxmask_fitbox     对照：框掩码模型 + 预测的框掩码走同一条拟合规则
    *_bandbox_nms                  公平性对照：Voronoi 定框也加上和拟合定框相同的 NMS 选项
    第 16 轮：hbatbox_aug_{abs,boxmask}_unet_{bandbox,fitbox}（以及 abs_unet_gtfitbox）= 换成 U 形解码的 attention 头之后的同一套定框。
    拟合较慢（约 10 ms / 检测），先 `python infer_bandbox.py prefit` 并行算好缓存（runs_cv/bandbox_cache/fit/，不入库），再 build。

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
from train_basic import AttnBBoxNet, BASE_CH, PreparedDataset, configure_attn_stride, configure_det_stride, device, predict  # noqa: E402
from refine_with_band import MODES, SEED_GRID, T_GRID, attn_to_band, gt_band, refine_image  # noqa: E402
from fit_hyperbola import apply_fits, fit_image  # noqa: E402
from evaluate import box_iou, collect, evaluate_run, prf_at, select_conf  # noqa: E402

RUNS = HERE / "runs_cv" / "ds2"
CACHE = HERE / "runs_cv" / "bandbox_cache"
OUT = HERE / "runs_cv" / "bandbox"
DATASETS = HERE.parent / "detector_comparison" / "datasets"
FOLDS, SEEDS = list(range(5)), list(range(10))
SRC_EXPS = ("hbatbox_aug_none", "hbatbox_aug_abs", "hbatbox_aug_oracle", "hbatbox_aug_boxmask",
            "hbatbox_aug_abs_unet", "hbatbox_aug_boxmask_unet")       # 后两个：第 16 轮
K_GRID = (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6)


def data_dir(fold):
    return DATASETS / f"ds2_f{fold}_s100_hbatbox"


def run_dir(exp, fold, seed):
    return RUNS / exp / f"f{fold}_seed{seed}"


def cache_dir(exp, fold, seed):
    return CACHE / exp / f"f{fold}_seed{seed}"


# ───────────────────────── 阶段 1：推理 ─────────────────────────
@torch.no_grad()
def infer(exps=SRC_EXPS, force=False):
    max_diff = {e: 0.0 for e in exps}
    for exp in exps:
        for fold, seed in itertools.product(FOLDS, SEEDS):
            rd = run_dir(exp, fold, seed)
            if not force and all((cache_dir(exp, fold, seed) / f"{n}_{ph}.json").exists() for n in ("dets", "meta") for ph in ("val", "test")):
                continue
            info = json.load(open(rd / "run_info.json", encoding="utf-8"))
            mode = info["mode"]
            configure_det_stride(info.get("det_stride", 8))
            configure_attn_stride(info.get("attn_stride", 4))
            model = AttnBBoxNet(in_ch=1, base_ch=BASE_CH, use_attn=(mode == "abs"), fuse=info["fuse"],
                                oracle=(mode == "oracle"), det_stride=info.get("det_stride", 8), attn_arch=info.get("attn_head", "conv")).to(device)
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
_CACHE_MEM = {}


def load_cache(exp, fold, seed, phase):
    key = (exp, fold, seed, phase)
    if key not in _CACHE_MEM:
        _CACHE_MEM[key] = _load_cache(exp, fold, seed, phase)
    return _CACHE_MEM[key]


def _load_cache(exp, fold, seed, phase):
    cd = cache_dir(exp, fold, seed)
    dets = json.load(open(cd / f"dets_{phase}.json"))
    metas = json.load(open(cd / f"meta_{phase}.json"))
    attn = None
    if (cd / f"attn_{phase}.npz").exists():
        z = np.load(cd / f"attn_{phase}.npz"); attn = {k: z[k].astype(np.float32) for k in z.files}
    return dets, metas, attn


FIT_CACHE = CACHE / "fit"
_FITS = {}


def fits_for(src, fold, seed, phase, tkey):
    """一折一个 phase 里所有检测（分数 ≥ min(SEED_GRID)）的双曲线拟合结果，{stem: {检测序号: {...}}}。
    tkey = "gt"（真实 band）或 attention map 的二值化阈值 t。结果缓存在磁盘和内存里。"""
    key = (src, fold, seed, phase, str(tkey))
    if key in _FITS:
        return _FITS[key]
    p = FIT_CACHE / src / str(tkey) / f"f{fold}_seed{seed}_{phase}.json"
    if p.exists():
        out = json.load(open(p))
    else:
        dets, metas, attn = load_cache(src, fold, seed, phase)
        out = {}
        for stem, d in dets.items():
            band = gt_band(fold, phase, stem) if tkey == "gt" else attn_to_band(attn[stem], tkey)
            out[stem] = {str(i): v for i, v in fit_image(d, band, metas[stem]["scale"], min(SEED_GRID)).items()}
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(p, "w"))
    _FITS[key] = out
    return out


def _prefit_one(job):
    fits_for(*job)
    _FITS.clear()
    return job


def prefit(variants, jobs=24):
    from multiprocessing import Pool
    todo = set()
    for v in variants:
        if not v.endswith("fitbox"):
            continue
        src = VARIANTS[v][0]
        for tkey in (["gt"] if v.endswith("_gtfitbox") else list(T_GRID)):
            for f, s, ph in itertools.product(FOLDS, SEEDS, ("val", "test")):
                if not (FIT_CACHE / src / str(tkey) / f"f{f}_seed{s}_{ph}.json").exists():
                    todo.add((src, f, s, ph, tkey))
    print(f"prefit: {len(todo)} 个 (来源, fold, seed, phase, band) 组合，{jobs} 个进程", flush=True)
    with Pool(jobs) as pool:
        for i, _ in enumerate(pool.imap_unordered(_prefit_one, sorted(todo, key=str)), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("prefit done", flush=True)


def scale_dets(dets, k, orig_size):
    oh, ow = orig_size
    out = []
    for x1, y1, x2, y2, s in dets:
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * k, (y2 - y1) * k
        out.append([max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(ow), cx + w / 2), min(float(oh), cy + h / 2), s])
    return out


NMS_GRID = (None, 0.5, 0.7)


def nms(dets, thr):
    """按分数从高到低的贪心 NMS（thr = None 时不做）。拟合定框会让两个检测收敛到同一条双曲线上，需要去重。"""
    if thr is None:
        return dets
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(box_iou(d, k) <= thr for k in keep):
            keep.append(d)
    return keep


def transform_phase(variant, cfg, src, fold, seed, phase):
    """对一折一个 phase 的全部检测做变换，返回 {stem: dets}。
    *fitbox 和 *_nms 变体的参数最后一项是 NMS 的 IoU 阈值（None = 不做）。"""
    nms_thr = None
    if variant.endswith("fitbox") or variant.endswith("_nms"):
        cfg, nms_thr = cfg[:-1], cfg[-1]
        variant = variant[:-4] if variant.endswith("_nms") else variant
    out = _transform_phase(variant, cfg, src, fold, seed, phase)
    return out if nms_thr is None else {k: nms(v, nms_thr) for k, v in out.items()}


def _transform_phase(variant, cfg, src, fold, seed, phase):
    dets, metas, attn = load_cache(src, fold, seed, phase)
    out = {}
    for stem, d in dets.items():
        if variant.endswith("_scale"):
            out[stem] = scale_dets(d, cfg[0], metas[stem]["orig_size"])
        elif variant.endswith("_gtfitbox"):
            s_seed, mode = cfg
            out[stem] = apply_fits(d, fits_for(src, fold, seed, phase, "gt")[stem], metas[stem]["scale"], s_seed, mode)
        elif variant.endswith("_fitbox"):
            t, s_seed, mode = cfg
            out[stem] = apply_fits(d, fits_for(src, fold, seed, phase, t)[stem], metas[stem]["scale"], s_seed, mode)
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
    # 第 13 轮：box-mask 对照，参数网格与方法完全相同
    "hbatbox_aug_boxmask_bandbox":  ("hbatbox_aug_boxmask", list(itertools.product(T_GRID, SEED_GRID, MODES))),
    # 第 15 轮：双曲线拟合定框（fit_hyperbola.py），参数网格 = 对应的 *_bandbox / *_gtbandbox 的网格 × NMS（None / 0.5 / 0.7）
    "hbatbox_aug_abs_fitbox":       ("hbatbox_aug_abs",    list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID))),
    "hbatbox_aug_abs_gtfitbox":     ("hbatbox_aug_abs",    list(itertools.product(SEED_GRID, MODES, NMS_GRID))),
    "hbatbox_aug_oracle_gtfitbox":  ("hbatbox_aug_oracle", list(itertools.product(SEED_GRID, MODES, NMS_GRID))),
    "hbatbox_aug_boxmask_fitbox":   ("hbatbox_aug_boxmask", list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID))),
    # 公平性对照：Round 12 / 13 的 Voronoi 定框也加上同样的 NMS 选项
    "hbatbox_aug_abs_bandbox_nms":      ("hbatbox_aug_abs",     list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID))),
    "hbatbox_aug_abs_gtbandbox_nms":    ("hbatbox_aug_abs",     list(itertools.product(SEED_GRID, MODES, NMS_GRID))),
    "hbatbox_aug_boxmask_bandbox_nms":  ("hbatbox_aug_boxmask", list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID))),
    # 第 16 轮：U 形解码的 attention 头（224×224 上输出和监督）。同样的两种定框规则；框掩码对照用同一个头
    "hbatbox_aug_abs_unet_bandbox":     ("hbatbox_aug_abs_unet",     list(itertools.product(T_GRID, SEED_GRID, MODES))),
    "hbatbox_aug_abs_unet_fitbox":      ("hbatbox_aug_abs_unet",     list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID))),
    "hbatbox_aug_abs_unet_gtfitbox":    ("hbatbox_aug_abs_unet",     list(itertools.product(SEED_GRID, MODES, NMS_GRID))),
    "hbatbox_aug_boxmask_unet_bandbox": ("hbatbox_aug_boxmask_unet", list(itertools.product(T_GRID, SEED_GRID, MODES))),
    "hbatbox_aug_boxmask_unet_fitbox":  ("hbatbox_aug_boxmask_unet", list(itertools.product(T_GRID, SEED_GRID, MODES, NMS_GRID))),
}


def merge_sweep(new_sections):
    """sweep.md 按变体分节；重建的变体替换对应节，其余保留。"""
    old = {}
    if (OUT / "sweep.md").exists():
        cur = None
        for line in (OUT / "sweep.md").read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                cur = line[3:].split("（")[0].strip(); old[cur] = []
            if cur is not None:
                old[cur].append(line)
    old.update(new_sections)
    body = "\n".join(l for v in old.values() for l in v)
    (OUT / "sweep.md").write_text("# band → 框参数敏感性（第 12 / 13 / 15 轮；选参只用验证列，测试列仅供查看）\n\n" + body.rstrip("\n") + "\n", encoding="utf-8")


def build(variants):
    gt_val = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["val"] for f in FOLDS}
    gt_test = {f: json.load(open(data_dir(f) / "gt_boxes.json", encoding="utf-8"))["test"] for f in FOLDS}
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = json.load(open(OUT / "chosen.json", encoding="utf-8")) if (OUT / "chosen.json").exists() else {}
    sweep_sections = {}
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
        lines = [f"## {variant}（来源 {src}）", "", f"选中（按 10 seed 合并验证 F1 均值）：{best}", "",
                        "| 参数 | 合并验证 F1 均值 | 测试 F1 均值 ± std（只看敏感性） |", "|---|---|---|"]
        for c in grid:
            mark = " **←**" if c == best else ""
            lines.append(f"| {c} | {np.mean(val_f1[c]):.3f} | {np.mean(test_f1[c]):.3f} ± {np.std(test_f1[c], ddof=1):.3f}{mark} |")
        lines.append(""); sweep_sections[variant] = lines
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
    merge_sweep(sweep_sections)
    print(json.dumps(chosen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["infer", "prefit", "build"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS), help="build 阶段：要生成的变体")
    ap.add_argument("--exps", nargs="+", default=list(SRC_EXPS), choices=list(SRC_EXPS), help="infer 阶段：要重推理的源实验")
    ap.add_argument("--force", action="store_true", help="infer 阶段：已有缓存也重做")
    args = ap.parse_args()
    if args.stage == "infer":
        infer(args.exps, args.force)
    elif args.stage == "prefit":
        prefit(args.variants)
    else:
        build(args.variants)
