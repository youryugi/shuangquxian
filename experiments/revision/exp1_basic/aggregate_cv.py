"""把 5 折 × 多 seed 的 run 合并成"每个 seed 一个交叉验证结果"，再按 seed 做配对统计。

    python aggregate_cv.py [--runs runs_cv] [--pairs hbatbox_aug_none:hbatbox_aug_abs aug_none:aug_abs]

合并方式（每个 exp、每个 seed）：
  * 把 5 折的验证集预测合在一起选一个阈值（验证集合并后 ~70 个目标，比单折 14–19 个稳得多）；
  * 把 5 折的测试集预测合在一起（每张图恰好在一折的测试集里出现一次，= 全部 131 张），在这个阈值下算 F1；
  * 同时给合并测试集的 AP50（不依赖阈值）和"测试集最优阈值 F1"（只作参考）。
配对统计（none -> abs，按 seed 配对）：ΔF1 / ΔAP50 的均值、bootstrap 95% CI、Wilcoxon 符号秩 p、胜/平/负。
另对另一套框（gt_boxes_official.json / gt_boxes_hbat.json）给同样的合并数字，作附表。

输出：<runs>/cv_summary.json、<runs>/cv_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent / "shared" / "datasets"
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE / "dataset1"))
from evaluate import ap50, collect, prf_at, select_conf  # noqa: E402
from d1_exps import d1_pairs  # noqa: E402

DEFAULT_PAIRS = ["hbatbox_aug_none:hbatbox_aug_abs", "aug_none:aug_abs", "hbatbox_none:hbatbox_abs", "basic_none:basic_abs",
                 # 第 7 轮
                 "hbatbox_aug_none:hbatbox_aug_oracle", "hbatbox_aug_abs:hbatbox_aug_oracle",
                 # 第 8 轮
                 "hbatbox_aug_none:hbatbox_aug_abs_extent", "hbatbox_aug_abs:hbatbox_aug_abs_extent",
                 "hbatbox_aug_none:hbatbox_aug_oracle_extent",
                 "hbatbox_aug_none_s4:hbatbox_aug_abs_s4", "hbatbox_aug_none:hbatbox_aug_none_s4", "hbatbox_aug_abs:hbatbox_aug_abs_s4",
                 # 第 9 轮
                 "hbatbox_aug_none_frac50:hbatbox_aug_abs_frac50", "hbatbox_aug_none_frac25:hbatbox_aug_abs_frac25",
                 # 第 10 轮
                 "hbatbox_aug_none:hbatbox_aug_abs_lam0.1", "hbatbox_aug_none:hbatbox_aug_abs_lam0.3",
                 "hbatbox_aug_none:hbatbox_aug_abs_lam1.5", "hbatbox_aug_none:hbatbox_aug_abs_gate",
                 # 换机器（Windows / 5060 Ti → Linux / 5090，2026-11-15）：旧结果备份为 *_win5060ti。
                 # 第一对 = 第 4 轮原表；后两对 = 同一配置换卡前后的差别，量化硬件带来的噪声
                 "hbatbox_aug_none_win5060ti:hbatbox_aug_abs_win5060ti",
                 "hbatbox_aug_none_win5060ti:hbatbox_aug_none", "hbatbox_aug_abs_win5060ti:hbatbox_aug_abs",
                 # 第 12 轮：band → 框尺寸做进推理（infer_bandbox.py）。方法 vs baseline；推理步骤本身的增量；等比放大对照；真实 band 上限
                 "hbatbox_aug_none:hbatbox_aug_abs_bandbox", "hbatbox_aug_abs:hbatbox_aug_abs_bandbox",
                 "hbatbox_aug_none:hbatbox_aug_none_scale", "hbatbox_aug_abs:hbatbox_aug_abs_scale",
                 "hbatbox_aug_none_scale:hbatbox_aug_abs_bandbox", "hbatbox_aug_abs_scale:hbatbox_aug_abs_bandbox",
                 "hbatbox_aug_abs_bandbox:hbatbox_aug_abs_gtbandbox", "hbatbox_aug_none:hbatbox_aug_oracle_gtbandbox",
                 # 第 13 轮：box-mask 对照（attention 头用填实的 HBAT 框监督，再做同样的 mask → 框）
                 "hbatbox_aug_none:hbatbox_aug_boxmask", "hbatbox_aug_abs:hbatbox_aug_boxmask",
                 "hbatbox_aug_none:hbatbox_aug_boxmask_bandbox", "hbatbox_aug_boxmask:hbatbox_aug_boxmask_bandbox",
                 "hbatbox_aug_none_scale:hbatbox_aug_boxmask_bandbox", "hbatbox_aug_boxmask_bandbox:hbatbox_aug_abs_bandbox",
                 # 第 14 轮：按每条双曲线做 band attention（stage2_instance.py）。baseline 的候选框 + 第二阶段；
                 # arm：plain = 普通二阶段框回归，unsup = attention 分支不监督，boxmask = 用实例框监督，band = 用实例 band 监督（方法），oracle = 真实实例 band
                 "hbatbox_aug_none:hbatbox_aug_none_nms", "hbatbox_aug_none_nms:hbatbox_aug_none_s2plain",
                 "hbatbox_aug_none:hbatbox_aug_none_s2plain", "hbatbox_aug_none:hbatbox_aug_none_s2band",
                 "hbatbox_aug_none_s2plain:hbatbox_aug_none_s2band", "hbatbox_aug_none_s2unsup:hbatbox_aug_none_s2band",
                 "hbatbox_aug_none_s2boxmask:hbatbox_aug_none_s2band",
                 "hbatbox_aug_none_s2plain:hbatbox_aug_none_s2unsup", "hbatbox_aug_none_s2plain:hbatbox_aug_none_s2boxmask",
                 "hbatbox_aug_none_s2band:hbatbox_aug_none_s2oracle", "hbatbox_aug_none_s2plain:hbatbox_aug_none_s2oracle",
                 # 框 = attention map 的外接框（实例级 mask 定框）
                 "hbatbox_aug_none_s2band:hbatbox_aug_none_s2band_mb", "hbatbox_aug_none_s2boxmask_mb:hbatbox_aug_none_s2band_mb",
                 "hbatbox_aug_none_s2plain:hbatbox_aug_none_s2band_mb",
                 # 和 Round 12 / 13 的整图方案比
                 "hbatbox_aug_abs_bandbox:hbatbox_aug_none_s2plain", "hbatbox_aug_abs_bandbox:hbatbox_aug_none_s2band",
                 "hbatbox_aug_boxmask_bandbox:hbatbox_aug_none_s2plain",
                 # 第一阶段带 attention（hbatbox_aug_abs）时再接同样的第二阶段：第一阶段的 attention 在框修好之后还剩多少
                 "hbatbox_aug_abs:hbatbox_aug_abs_s2plain", "hbatbox_aug_none_s2plain:hbatbox_aug_abs_s2plain",
                 "hbatbox_aug_abs_s2plain:hbatbox_aug_abs_s2band", "hbatbox_aug_none_s2band:hbatbox_aug_abs_s2band",
                 "hbatbox_aug_none_s2plain:hbatbox_aug_boxmask_s2plain", "hbatbox_aug_boxmask_s2plain:hbatbox_aug_abs_s2plain",
                 # 第 15 轮：双曲线拟合定框（fit_hyperbola.py）代替按框中心的 Voronoi 切分
                 "hbatbox_aug_abs_gtbandbox:hbatbox_aug_abs_gtfitbox", "hbatbox_aug_oracle_gtbandbox:hbatbox_aug_oracle_gtfitbox",
                 "hbatbox_aug_abs_bandbox:hbatbox_aug_abs_fitbox", "hbatbox_aug_none:hbatbox_aug_abs_fitbox", "hbatbox_aug_abs:hbatbox_aug_abs_fitbox",
                 "hbatbox_aug_none_scale:hbatbox_aug_abs_fitbox", "hbatbox_aug_abs_fitbox:hbatbox_aug_abs_gtfitbox",
                 "hbatbox_aug_boxmask_bandbox:hbatbox_aug_boxmask_fitbox", "hbatbox_aug_boxmask_fitbox:hbatbox_aug_abs_fitbox",
                 "hbatbox_aug_abs_fitbox:hbatbox_aug_none_s2plain", "hbatbox_aug_none_s2plain:hbatbox_aug_abs_gtfitbox",
                 "hbatbox_aug_none_s2oracle:hbatbox_aug_abs_gtfitbox",
                 "hbatbox_aug_abs_s2plain:hbatbox_aug_abs_gtfitbox",      # 同一个第一阶段上：不用 band 的第二阶段 vs 真实 band + 拟合
                 "hbatbox_aug_abs_bandbox:hbatbox_aug_abs_bandbox_nms", "hbatbox_aug_abs_bandbox_nms:hbatbox_aug_abs_fitbox",
                 "hbatbox_aug_abs_gtbandbox_nms:hbatbox_aug_abs_gtfitbox", "hbatbox_aug_boxmask_bandbox:hbatbox_aug_boxmask_bandbox_nms",
                 "hbatbox_aug_boxmask_bandbox_nms:hbatbox_aug_abs_bandbox_nms",
                 # 第 16 轮：U 形解码的 attention 头（224×224 上输出和监督），train_basic.py --attn-head unet
                 "hbatbox_aug_none:hbatbox_aug_abs_unet", "hbatbox_aug_abs:hbatbox_aug_abs_unet",
                 "hbatbox_aug_none:hbatbox_aug_boxmask_unet", "hbatbox_aug_boxmask:hbatbox_aug_boxmask_unet", "hbatbox_aug_boxmask_unet:hbatbox_aug_abs_unet",
                 "hbatbox_aug_abs_bandbox:hbatbox_aug_abs_unet_bandbox", "hbatbox_aug_abs_fitbox:hbatbox_aug_abs_unet_fitbox",
                 "hbatbox_aug_abs_unet:hbatbox_aug_abs_unet_bandbox", "hbatbox_aug_abs_unet:hbatbox_aug_abs_unet_fitbox",
                 "hbatbox_aug_abs_unet_bandbox:hbatbox_aug_abs_unet_fitbox", "hbatbox_aug_none:hbatbox_aug_abs_unet_fitbox", "hbatbox_aug_none:hbatbox_aug_abs_unet_bandbox",
                 "hbatbox_aug_boxmask_unet_bandbox:hbatbox_aug_abs_unet_bandbox", "hbatbox_aug_boxmask_unet_fitbox:hbatbox_aug_abs_unet_fitbox",
                 "hbatbox_aug_boxmask_unet_bandbox:hbatbox_aug_abs_unet_fitbox",
                 "hbatbox_aug_abs_unet_fitbox:hbatbox_aug_abs_unet_gtfitbox", "hbatbox_aug_abs_gtfitbox:hbatbox_aug_abs_unet_gtfitbox",
                 "hbatbox_aug_abs_unet_fitbox:hbatbox_aug_none_s2plain", "hbatbox_aug_abs_unet_bandbox:hbatbox_aug_none_s2plain",
                 # 第 18 轮：修好翻转（train_basic.py 几何翻转，hbatbox_flipfix_*）后重做第 4–16 轮的主要配对；新口径（预测框裁到图内）
                 "hbatbox_flipfix_none:hbatbox_flipfix_abs", "hbatbox_flipfix_none:hbatbox_flipfix_oracle",
                 "hbatbox_flipfix_abs:hbatbox_flipfix_oracle", "hbatbox_flipfix_none:hbatbox_flipfix_abs_bandbox",
                 "hbatbox_flipfix_abs:hbatbox_flipfix_abs_bandbox", "hbatbox_flipfix_none:hbatbox_flipfix_none_scale",
                 "hbatbox_flipfix_abs:hbatbox_flipfix_abs_scale", "hbatbox_flipfix_none_scale:hbatbox_flipfix_abs_bandbox",
                 "hbatbox_flipfix_abs_scale:hbatbox_flipfix_abs_bandbox", "hbatbox_flipfix_abs_bandbox:hbatbox_flipfix_abs_gtbandbox",
                 "hbatbox_flipfix_none:hbatbox_flipfix_oracle_gtbandbox", "hbatbox_flipfix_none:hbatbox_flipfix_boxmask",
                 "hbatbox_flipfix_abs:hbatbox_flipfix_boxmask", "hbatbox_flipfix_none:hbatbox_flipfix_boxmask_bandbox",
                 "hbatbox_flipfix_boxmask:hbatbox_flipfix_boxmask_bandbox", "hbatbox_flipfix_none_scale:hbatbox_flipfix_boxmask_bandbox",
                 "hbatbox_flipfix_boxmask_bandbox:hbatbox_flipfix_abs_bandbox", "hbatbox_flipfix_abs_gtbandbox:hbatbox_flipfix_abs_gtfitbox",
                 "hbatbox_flipfix_oracle_gtbandbox:hbatbox_flipfix_oracle_gtfitbox", "hbatbox_flipfix_abs_bandbox:hbatbox_flipfix_abs_fitbox",
                 "hbatbox_flipfix_none:hbatbox_flipfix_abs_fitbox", "hbatbox_flipfix_abs:hbatbox_flipfix_abs_fitbox",
                 "hbatbox_flipfix_none_scale:hbatbox_flipfix_abs_fitbox", "hbatbox_flipfix_abs_fitbox:hbatbox_flipfix_abs_gtfitbox",
                 "hbatbox_flipfix_boxmask_bandbox:hbatbox_flipfix_boxmask_fitbox", "hbatbox_flipfix_boxmask_fitbox:hbatbox_flipfix_abs_fitbox",
                 "hbatbox_flipfix_abs_bandbox:hbatbox_flipfix_abs_bandbox_nms", "hbatbox_flipfix_abs_bandbox_nms:hbatbox_flipfix_abs_fitbox",
                 "hbatbox_flipfix_abs_gtbandbox_nms:hbatbox_flipfix_abs_gtfitbox",
                 "hbatbox_flipfix_boxmask_bandbox:hbatbox_flipfix_boxmask_bandbox_nms",
                 "hbatbox_flipfix_boxmask_bandbox_nms:hbatbox_flipfix_abs_bandbox_nms", "hbatbox_flipfix_none:hbatbox_flipfix_abs_unet",
                 "hbatbox_flipfix_abs:hbatbox_flipfix_abs_unet", "hbatbox_flipfix_none:hbatbox_flipfix_boxmask_unet",
                 "hbatbox_flipfix_boxmask:hbatbox_flipfix_boxmask_unet", "hbatbox_flipfix_boxmask_unet:hbatbox_flipfix_abs_unet",
                 "hbatbox_flipfix_abs_bandbox:hbatbox_flipfix_abs_unet_bandbox", "hbatbox_flipfix_abs_fitbox:hbatbox_flipfix_abs_unet_fitbox",
                 "hbatbox_flipfix_abs_unet:hbatbox_flipfix_abs_unet_bandbox", "hbatbox_flipfix_abs_unet:hbatbox_flipfix_abs_unet_fitbox",
                 "hbatbox_flipfix_abs_unet_bandbox:hbatbox_flipfix_abs_unet_fitbox", "hbatbox_flipfix_none:hbatbox_flipfix_abs_unet_fitbox",
                 "hbatbox_flipfix_none:hbatbox_flipfix_abs_unet_bandbox", "hbatbox_flipfix_boxmask_unet_bandbox:hbatbox_flipfix_abs_unet_bandbox",
                 "hbatbox_flipfix_boxmask_unet_fitbox:hbatbox_flipfix_abs_unet_fitbox",
                 "hbatbox_flipfix_boxmask_unet_bandbox:hbatbox_flipfix_abs_unet_fitbox",
                 "hbatbox_flipfix_abs_unet_fitbox:hbatbox_flipfix_abs_unet_gtfitbox",
                 "hbatbox_flipfix_abs_gtfitbox:hbatbox_flipfix_abs_unet_gtfitbox", "hbatbox_flipfix_boxmask_bandbox_nms:hbatbox_flipfix_abs_fitbox",
                 "hbatbox_flipfix_boxmask_bandbox_nms:hbatbox_flipfix_abs_unet_fitbox",
                 # 只修翻转的变化（同一配置，旧 run → 新 run）
                 "hbatbox_aug_none:hbatbox_flipfix_none", "hbatbox_aug_abs:hbatbox_flipfix_abs", "hbatbox_aug_boxmask:hbatbox_flipfix_boxmask",
                 "hbatbox_aug_oracle:hbatbox_flipfix_oracle", "hbatbox_aug_abs_unet:hbatbox_flipfix_abs_unet",
                 "hbatbox_aug_boxmask_unet:hbatbox_flipfix_boxmask_unet", "hbatbox_aug_abs_bandbox:hbatbox_flipfix_abs_bandbox",
                 "hbatbox_aug_abs_fitbox:hbatbox_flipfix_abs_fitbox", "hbatbox_aug_boxmask_bandbox:hbatbox_flipfix_boxmask_bandbox",
                 "hbatbox_aug_boxmask_fitbox:hbatbox_flipfix_boxmask_fitbox", "hbatbox_aug_abs_unet_fitbox:hbatbox_flipfix_abs_unet_fitbox",
                 "hbatbox_aug_abs_unet_bandbox:hbatbox_flipfix_abs_unet_bandbox",
                 "hbatbox_aug_boxmask_unet_fitbox:hbatbox_flipfix_boxmask_unet_fitbox",
                 "hbatbox_aug_boxmask_unet_bandbox:hbatbox_flipfix_boxmask_unet_bandbox", "hbatbox_aug_none_scale:hbatbox_flipfix_none_scale",
                 "hbatbox_aug_abs_gtfitbox:hbatbox_flipfix_abs_gtfitbox", "hbatbox_aug_abs_gtbandbox:hbatbox_flipfix_abs_gtbandbox",
                 # 第 17 轮：两种翻转实现的交叉检查、参数化头（train_param.py）
                 "hbatbox_flipfix_none:hbatbox_augfix_none", "hbatbox_flipfix_none:hbatbox_augfix_param",
                 "hbatbox_flipfix_none:hbatbox_augfix_pseudo"]
# 第 19 轮：固定测试集 + 嵌套训练子集（训练集 25 / 50 / 75%；100% 即上面第 18 轮的配对）
def _frac(name, p):   # "abs_fitbox" → "hbatbox_flipfix_abs_frac25_fitbox"（比例插在 arm 名后面，和 infer_bandbox.py 的变体名一致）
    arm, _, rest = name.partition("_")
    return f"hbatbox_flipfix_{arm}_frac{p}" + (f"_{rest}" if rest else "")


DEFAULT_PAIRS += [f"{_frac(a, p)}:{_frac(b, p)}" for p in (25, 50, 75) for a, b in (
    ("none", "abs"), ("none", "boxmask"), ("boxmask", "abs"),
    ("none", "none_scale"), ("none", "abs_fitbox"), ("none_scale", "abs_fitbox"),
    ("boxmask_fitbox", "abs_fitbox"), ("boxmask_bandbox_nms", "abs_fitbox"), ("boxmask_bandbox", "abs_bandbox"),
    ("abs_bandbox", "abs_fitbox"), ("abs_fitbox", "abs_gtfitbox"), ("none", "boxmask_bandbox_nms"))]
DEFAULT_PAIRS += ["hbatbox_flipfix_boxmask:hbatbox_flipfix_abs"]   # 100% 一列的 box-mask fusion → band fusion（上面只有反方向）


def data_dir_of(run_info, fold):
    suffix = "_hbatbox" if run_info.get("box_source") == "hbat" else ""
    return DATASETS / f"{run_info.get('dataset', 'ds2')}_f{fold}_s100{suffix}"


def pooled_metrics(run_dirs, gt_name="gt_boxes.json"):
    """run_dirs: {fold: run_dir}. 合并各折的 val / test 预测与 GT。"""
    sv, tv, nv = [], [], 0
    st, tt, nt = [], [], 0
    for fold, rd in sorted(run_dirs.items()):
        ri = json.load(open(rd / "run_info.json", encoding="utf-8"))
        gt = json.load(open(data_dir_of(ri, fold) / gt_name, encoding="utf-8"))
        pv = json.load(open(rd / "preds_val.json", encoding="utf-8"))
        pt = json.load(open(rd / "preds_test.json", encoding="utf-8"))
        s, t, n = collect(pv, gt["val"]); sv.append(s); tv.append(t); nv += n
        s, t, n = collect(pt, gt["test"]); st.append(s); tt.append(t); nt += n
    sv, tv = np.concatenate(sv), np.concatenate(tv)
    st, tt = np.concatenate(st), np.concatenate(tt)
    conf, f1v = select_conf(sv, tv, nv)
    oc, of = select_conf(st, tt, nt)
    out = {"val_conf": float(conf), "val_F1": float(f1v), "n_val_gt": int(nv), "n_test_gt": int(nt),
           "test": {k: float(v) for k, v in prf_at(st, tt, nt, conf).items()},
           "AP50": float(ap50(st, tt, nt)), "oracle_conf": float(oc), "oracle_F1": float(of),
           "F1_at_0.5": float(prf_at(st, tt, nt, 0.5)["F1"])}
    return out


def paired(a, b, n_boot=10000, seed=0):
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(b, a, zero_method="zsplit", alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
    except Exception:
        p = float("nan")
    return {"n": int(len(d)), "mean_A": float(a.mean()), "mean_B": float(b.mean()), "std_A": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "std_B": float(b.std(ddof=1)) if len(b) > 1 else 0.0, "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "wilcoxon_p": p,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "per_seed_delta": [round(float(x), 4) for x in d]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs_cv"))
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--dataset", default="ds2", choices=["ds1", "ds2"],
                    help="ds1：读 <runs>/ds1/，默认配对用 dataset1/d1_exps.py，输出 cv_summary_ds1.{json,md}（不覆盖 ds2 的）")
    args = ap.parse_args()
    if args.pairs is None:
        args.pairs = d1_pairs() if args.dataset == "ds1" else DEFAULT_PAIRS
    out_stem = "cv_summary" if args.dataset == "ds2" else f"cv_summary_{args.dataset}"
    runs = Path(args.runs) / args.dataset

    # 收集：exp -> seed -> {fold: run_dir}
    tree = defaultdict(lambda: defaultdict(dict))
    for m in runs.glob("*/f*_seed*/metrics.json"):
        rd = m.parent
        exp = rd.parent.name
        f, s = rd.name.split("_seed")
        tree[exp][int(s)][int(f[1:])] = rd

    results = {}
    lines = ["# 5 折合并结果（每个 seed 一个交叉验证数字）", ""]
    for exp in sorted(tree):
        results[exp] = {}
        for seed in sorted(tree[exp]):
            rds = {f: rd for f, rd in tree[exp][seed].items() if f in args.folds}
            if len(rds) != len(args.folds):
                continue   # 折没跑齐的 seed 不算
            ri = json.load(open(next(iter(rds.values())) / "run_info.json", encoding="utf-8"))
            r = {"primary": pooled_metrics(rds), "box_source": ri.get("box_source")}
            for alt in ("official", "hbat"):
                if (data_dir_of(ri, args.folds[0]) / f"gt_boxes_{alt}.json").exists():
                    r[f"vs_{alt}"] = pooled_metrics(rds, f"gt_boxes_{alt}.json")
            # 各折自己的（单折验证集选阈值）F1，作参考
            r["per_fold_F1"] = {f: json.load(open(rd / "metrics.json", encoding="utf-8"))["test"]["at_selected"]["F1"] for f, rd in sorted(rds.items())}
            results[exp][seed] = r
        if not results[exp]:
            continue
        f1 = np.array([r["primary"]["test"]["F1"] for r in results[exp].values()])
        apv = np.array([r["primary"]["AP50"] for r in results[exp].values()])
        orc = np.array([r["primary"]["oracle_F1"] for r in results[exp].values()])
        pf = np.array([[r["per_fold_F1"][f] for f in args.folds] for r in results[exp].values()])
        lines.append(f"## {exp}  (seeds {len(f1)}, 目标框 = {next(iter(results[exp].values()))['box_source']})")
        lines.append(f"- 合并 5 折：F1 {f1.mean():.3f} ± {f1.std(ddof=1) if len(f1) > 1 else 0:.3f}（中位数 {np.median(f1):.3f}），"
                     f"AP50 {apv.mean():.3f} ± {apv.std(ddof=1) if len(apv) > 1 else 0:.3f}，测试最优阈值 F1 {orc.mean():.3f}")
        lines.append(f"- 各折单独（单折验证集选阈值）F1 均值：" + "、".join(f"f{f} {pf[:, i].mean():.3f}" for i, f in enumerate(args.folds)))
        for alt in ("official", "hbat"):
            vals = [r.get(f"vs_{alt}") for r in results[exp].values()]
            if all(vals):
                a = np.array([v["test"]["F1"] for v in vals]); b = np.array([v["AP50"] for v in vals])
                lines.append(f"- 对 {alt} 框评：F1 {a.mean():.3f} ± {a.std(ddof=1) if len(a) > 1 else 0:.3f}，AP50 {b.mean():.3f}")
        lines.append("")

    lines.append("# 配对统计（按 seed 配对，B − A）"); lines.append("")
    lines.append("| A → B | n | F1 A | F1 B | ΔF1 均值 [95% CI] | Wilcoxon p | 胜/平/负 | ΔAP50 均值 [CI] | p | 胜 | Δ最优阈值F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    pairs_out = {}
    for pr in args.pairs:
        a_exp, b_exp = pr.split(":")
        seeds = sorted(set(results.get(a_exp, {})) & set(results.get(b_exp, {})))
        if not seeds:
            continue
        st = {}
        for key, get in (("F1", lambda r: r["primary"]["test"]["F1"]), ("AP50", lambda r: r["primary"]["AP50"]),
                         ("oracle_F1", lambda r: r["primary"]["oracle_F1"]), ("F1_at_0.5", lambda r: r["primary"]["F1_at_0.5"])):
            st[key] = paired([get(results[a_exp][s]) for s in seeds], [get(results[b_exp][s]) for s in seeds])
        for alt in ("official", "hbat"):
            if all(f"vs_{alt}" in results[a_exp][s] and f"vs_{alt}" in results[b_exp][s] for s in seeds):
                st[f"F1_vs_{alt}"] = paired([results[a_exp][s][f"vs_{alt}"]["test"]["F1"] for s in seeds],
                                            [results[b_exp][s][f"vs_{alt}"]["test"]["F1"] for s in seeds])
        pairs_out[pr] = {"seeds": seeds, **st}
        f, a_, o = st["F1"], st["AP50"], st["oracle_F1"]
        lines.append(f"| {a_exp} → {b_exp} | {f['n']} | {f['mean_A']:.3f} | {f['mean_B']:.3f} | {f['delta_mean']:+.3f} [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}] | "
                     f"{f['wilcoxon_p']:.3f} | {f['wins']}/{f['ties']}/{f['losses']} | {a_['delta_mean']:+.3f} [{a_['ci95'][0]:+.3f}, {a_['ci95'][1]:+.3f}] | "
                     f"{a_['wilcoxon_p']:.3f} | {a_['wins']}/{a_['n']} | {o['delta_mean']:+.3f} |")
    out = Path(args.runs)
    with open(out / f"{out_stem}.json", "w", encoding="utf-8") as f:
        json.dump({"per_exp": {e: {str(s): r for s, r in d.items()} for e, d in results.items()}, "pairs": pairs_out}, f, ensure_ascii=False, indent=1)
    (out / f"{out_stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out / (out_stem + '.md')}")


if __name__ == "__main__":
    main()
