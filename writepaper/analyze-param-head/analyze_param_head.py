# -*- coding: utf-8 -*-
"""
Head 2: Size / 参数回归头评价分析
读取 test_metrics_summary.csv，针对不同 lambda_size 在多个随机种子上的结果取均值与标准差，
并对适合评估参数回归头的指标进行可视化对比。

适合评估 size/参数头的指标:
  (A) 检测层面（基于参数生成mask后与GT匹配的结果）-- 越大越好
      - detection_precision  匹配数 / 预测双曲线数
      - detection_recall     匹配数 / GT双曲线数
      - mean_match_iou       匹配上的预测-GT 之间的平均IoU
  (B) 参数误差（匹配上的对里参数的回归误差，主要量化 size_head 的回归质量）-- 越小越好
      - mae_x_vertex / mae_y_vertex   顶点坐标 (来自mask解码 + 几何拟合)
      - mae_width / mae_height / mae_thickness   size_head 直接回归的三个参数
      （RMSE 与 MAE 趋势一致，附加 RMSE 仅作参考）

注：
  - x_vertex / y_vertex 严格来说是 mask 解码 + 二次曲线拟合的结果，
    但它们参与了双曲线参数的最终输出，所以仍归入参数头的"产品质量"指标。
  - 真正纯粹来自 size_head 的回归量是 width / height / thickness。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 配置
# =========================================================
INPUT_CSV = r"C:\Users\79152\Desktop\github\shuangquxian\ml\0409\0521-1_0521_2034\test_metrics_summary.csv"
OUTPUT_DIR = r"C:\Users\79152\Desktop\github\shuangquxian\writepaper\analyze-param-head"
# 检测匹配相关指标 (越大越好)
DETECTION_METRICS = [
    "detection_precision",
    "detection_recall",
    "mean_match_iou",
]

# 参数 MAE (越小越好)，主指标
MAE_METRICS = [
    "mae_x_vertex",
    "mae_y_vertex",
    "mae_width",
    "mae_height",
    "mae_thickness",
]

# 参数 RMSE (越小越好)，作为参考
RMSE_METRICS = [
    "rmse_x_vertex",
    "rmse_y_vertex",
    "rmse_width",
    "rmse_height",
    "rmse_thickness",
]

# 标签
METRIC_LABELS = {
    "detection_precision": "Detection Precision",
    "detection_recall":    "Detection Recall",
    "mean_match_iou":      "Mean Match IoU",
    "mae_x_vertex":        "MAE x_vertex (px)",
    "mae_y_vertex":        "MAE y_vertex (px)",
    "mae_width":           "MAE width (px)",
    "mae_height":          "MAE height (px)",
    "mae_thickness":       "MAE thickness (px)",
    "rmse_x_vertex":       "RMSE x_vertex (px)",
    "rmse_y_vertex":       "RMSE y_vertex (px)",
    "rmse_width":          "RMSE width (px)",
    "rmse_height":         "RMSE height (px)",
    "rmse_thickness":      "RMSE thickness (px)",
}


# =========================================================
# 辅助函数
# =========================================================
def aggregate(df: pd.DataFrame, metrics):
    """按 lambda_size 聚合，对 seed 取均值/标准差。"""
    g = df.groupby("lambda_size")[metrics]
    mean_df = g.mean().reset_index().sort_values("lambda_size").reset_index(drop=True)
    std_df  = g.std(ddof=1).reset_index().sort_values("lambda_size").reset_index(drop=True)
    n_seed  = g.size().reset_index(name="n_seeds")
    return mean_df, std_df, n_seed


def build_pretty(mean_df, std_df, metrics):
    """每个指标一列 "mean ± std" 的简表。"""
    out = mean_df[["lambda_size"]].copy()
    for m in metrics:
        out[m] = [f"{mu:.4f} ± {sd:.4f}" if abs(mu) < 1 else f"{mu:.2f} ± {sd:.2f}"
                  for mu, sd in zip(mean_df[m], std_df[m])]
    return out


def plot_metrics(mean_df, std_df, metrics, title_prefix, out_png,
                 ylabel="score", lower_is_better=False):
    """所有指标画在同一张图里。"""
    lambdas = mean_df["lambda_size"].values
    x_labels = [str(int(l)) if l == int(l) else str(l) for l in lambdas]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in metrics:
        ax.errorbar(
            range(len(lambdas)),
            mean_df[m].values,
            yerr=std_df[m].values,
            marker="o", capsize=3, linewidth=1.5,
            label=METRIC_LABELS.get(m, m),
        )
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("lambda_size")
    ax.set_ylabel(ylabel)
    arrow = "↓ lower is better" if lower_is_better else "↑ higher is better"
    ax.set_title(f"{title_prefix}  ({arrow})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_each_metric(mean_df, std_df, metrics, out_dir, prefix, lower_is_better=False):
    """每个指标单独一张图。"""
    lambdas = mean_df["lambda_size"].values
    x_labels = [str(int(l)) if l == int(l) else str(l) for l in lambdas]
    n_seed = "?"
    for m in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(
            range(len(lambdas)),
            mean_df[m].values,
            yerr=std_df[m].values,
            marker="o", capsize=4, linewidth=1.8,
        )
        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("lambda_size")
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        arrow = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"Param Head: {METRIC_LABELS.get(m, m)} vs lambda_size  ({arrow})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_{m}_vs_lambda.png"), dpi=150)
        plt.close(fig)


# =========================================================
# 主逻辑
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, skiprows=[1])
    all_metrics = DETECTION_METRICS + MAE_METRICS + RMSE_METRICS
    df = df[["seed", "lambda_size"] + all_metrics].copy()

    # ---- 检测层面 ----
    det_mean, det_std, n_seed = aggregate(df, DETECTION_METRICS)
    det_pretty = build_pretty(det_mean, det_std, DETECTION_METRICS)
    det_pretty.insert(1, "n_seeds", n_seed["n_seeds"].values)

    # ---- 参数 MAE ----
    mae_mean, mae_std, _ = aggregate(df, MAE_METRICS)
    mae_pretty = build_pretty(mae_mean, mae_std, MAE_METRICS)
    mae_pretty.insert(1, "n_seeds", n_seed["n_seeds"].values)

    # ---- 参数 RMSE ----
    rmse_mean, rmse_std, _ = aggregate(df, RMSE_METRICS)
    rmse_pretty = build_pretty(rmse_mean, rmse_std, RMSE_METRICS)
    rmse_pretty.insert(1, "n_seeds", n_seed["n_seeds"].values)

    # ---- 全量数值表 (合并 mean / std 数值列) ----
    full = det_mean[["lambda_size"]].copy()
    full["n_seeds"] = n_seed["n_seeds"].values
    for m in all_metrics:
        if m in DETECTION_METRICS:
            mean_src, std_src = det_mean, det_std
        elif m in MAE_METRICS:
            mean_src, std_src = mae_mean, mae_std
        else:
            mean_src, std_src = rmse_mean, rmse_std
        full[f"{m}_mean"] = mean_src[m]
        full[f"{m}_std"]  = std_src[m]

    # ---- 保存 csv ----
    full.to_csv(os.path.join(OUTPUT_DIR, "param_head_lambda_summary_full.csv"),
                index=False, encoding="utf-8-sig")
    det_pretty.to_csv(os.path.join(OUTPUT_DIR, "param_head_detection_summary.csv"),
                      index=False, encoding="utf-8-sig")
    mae_pretty.to_csv(os.path.join(OUTPUT_DIR, "param_head_mae_summary.csv"),
                      index=False, encoding="utf-8-sig")
    rmse_pretty.to_csv(os.path.join(OUTPUT_DIR, "param_head_rmse_summary.csv"),
                       index=False, encoding="utf-8-sig")

    # ---- 控制台输出 ----
    n = int(n_seed["n_seeds"].iloc[0])
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print("=" * 80)
        print(f"Param Head 检测层面 (越大越好, mean ± std over {n} seeds)")
        print("=" * 80)
        print(det_pretty.to_string(index=False))

        print("\n" + "=" * 80)
        print(f"Param Head 参数 MAE (越小越好, mean ± std over {n} seeds)")
        print("=" * 80)
        print(mae_pretty.to_string(index=False))

        print("\n" + "=" * 80)
        print(f"Param Head 参数 RMSE (越小越好, mean ± std over {n} seeds)")
        print("=" * 80)
        print(rmse_pretty.to_string(index=False))

    # ---- 绘图 ----
    plot_metrics(det_mean, det_std, DETECTION_METRICS,
                 title_prefix="Param Head: detection metrics vs lambda_size",
                 out_png=os.path.join(OUTPUT_DIR, "param_head_detection_vs_lambda.png"),
                 ylabel="score", lower_is_better=False)
    plot_metrics(mae_mean, mae_std, MAE_METRICS,
                 title_prefix="Param Head: parameter MAE vs lambda_size",
                 out_png=os.path.join(OUTPUT_DIR, "param_head_mae_vs_lambda.png"),
                 ylabel="MAE (pixels)", lower_is_better=True)
    plot_metrics(rmse_mean, rmse_std, RMSE_METRICS,
                 title_prefix="Param Head: parameter RMSE vs lambda_size",
                 out_png=os.path.join(OUTPUT_DIR, "param_head_rmse_vs_lambda.png"),
                 ylabel="RMSE (pixels)", lower_is_better=True)

    # 每个指标单独画
    plot_each_metric(det_mean, det_std, DETECTION_METRICS, OUTPUT_DIR,
                     prefix="param_head_det", lower_is_better=False)
    plot_each_metric(mae_mean, mae_std, MAE_METRICS, OUTPUT_DIR,
                     prefix="param_head_mae", lower_is_better=True)

    # ---- 标出每个指标的最佳 lambda ----
    print("\n" + "=" * 80)
    print("Best lambda for each Param-Head metric:")
    print("=" * 80)
    best_records = []
    for m in DETECTION_METRICS:
        idx = det_mean[m].idxmax()
        bl, bv, bs = det_mean.loc[idx, "lambda_size"], det_mean.loc[idx, m], det_std.loc[idx, m]
        print(f"  [↑ better] {METRIC_LABELS[m]:<22s} -> lambda={bl:<6}  {bv:.4f} ± {bs:.4f}")
        best_records.append({"metric": m, "direction": "max",
                             "best_lambda": bl, "best_mean": bv, "best_std": bs})
    for m in MAE_METRICS:
        idx = mae_mean[m].idxmin()
        bl, bv, bs = mae_mean.loc[idx, "lambda_size"], mae_mean.loc[idx, m], mae_std.loc[idx, m]
        print(f"  [↓ better] {METRIC_LABELS[m]:<22s} -> lambda={bl:<6}  {bv:.4f} ± {bs:.4f}")
        best_records.append({"metric": m, "direction": "min",
                             "best_lambda": bl, "best_mean": bv, "best_std": bs})
    for m in RMSE_METRICS:
        idx = rmse_mean[m].idxmin()
        bl, bv, bs = rmse_mean.loc[idx, "lambda_size"], rmse_mean.loc[idx, m], rmse_std.loc[idx, m]
        best_records.append({"metric": m, "direction": "min",
                             "best_lambda": bl, "best_mean": bv, "best_std": bs})

    pd.DataFrame(best_records).to_csv(
        os.path.join(OUTPUT_DIR, "param_head_best_lambda_per_metric.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
