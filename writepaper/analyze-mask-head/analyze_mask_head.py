# -*- coding: utf-8 -*-
"""
Head 1: Mask / Heatmap 头评价分析
读取 test_metrics_summary.csv，针对不同 lambda_size 在多个随机种子上的结果取均值与标准差，
并对适合评估 mask 头的指标进行可视化对比。

适合评估 mask 头的指标（直接反映分割mask质量）:
    - pixel_precision  : intersection / pred_pixels  (像素精确率)
    - pixel_recall     : intersection / gt_pixels    (像素召回率)
    - pixel_f1         : 2*P*R/(P+R)                (像素F1，综合)
    - global_iou       : intersection / union       (全局IoU，综合)
    - mean_image_iou   : 每张图IoU均值

注：
    - global_overlap 与 pixel_recall 数值上等价，已剔除
    - mean_image_overlap 与 mean_image_iou 高度相关，作为参考可选
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 配置
# =========================================================
INPUT_CSV = r"C:\Users\79152\Desktop\github\shuangquxian\ml\0409\0521-1_0521_2034\test_metrics_summary.csv"
OUTPUT_DIR = r"C:\Users\79152\Desktop\github\shuangquxian\writepaper\fenxi-2head-differentlamda"

# 适合 mask 头评估的指标 (越大越好)
MASK_METRICS = [
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "global_iou",
]

# 中文标签（用于绘图标题）
METRIC_LABELS = {
    "pixel_precision": "Pixel Precision",
    "pixel_recall":    "Pixel Recall",
    "pixel_f1":        "Pixel F1",
    "global_iou":      "Global IoU",
}


# =========================================================
# 主逻辑
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 跳过第2行（中英文说明行）
    df = pd.read_csv(INPUT_CSV, skiprows=[1])

    # 仅保留需要的列
    keep_cols = ["seed", "lambda_size"] + MASK_METRICS
    df = df[keep_cols].copy()

    # ---- 1) 按 lambda 分组，对 seed 取均值与标准差 ----
    grouped = df.groupby("lambda_size")[MASK_METRICS]
    mean_df = grouped.mean().reset_index()
    std_df  = grouped.std(ddof=1).reset_index()
    n_seed  = grouped.size().reset_index(name="n_seeds")

    # 拼成 "mean±std" 形式便于查看
    summary = mean_df[["lambda_size"]].copy()
    summary["n_seeds"] = n_seed["n_seeds"]
    for m in MASK_METRICS:
        summary[f"{m}_mean"] = mean_df[m]
        summary[f"{m}_std"]  = std_df[m]
        summary[f"{m}"] = [f"{mu:.4f} ± {sd:.4f}"
                          for mu, sd in zip(mean_df[m], std_df[m])]

    # 按 lambda 升序
    summary = summary.sort_values("lambda_size").reset_index(drop=True)

    # ---- 2) 保存表格 ----
    # (a) 全量数值表（带 mean / std 数值列）
    full_csv = os.path.join(OUTPUT_DIR, "mask_head_lambda_summary_full.csv")
    summary.to_csv(full_csv, index=False, encoding="utf-8-sig")

    # (b) 简表：每个指标一列，内容为 "mean ± std"
    nice_cols = ["lambda_size"] + MASK_METRICS
    nice = summary[nice_cols].copy()
    nice_csv = os.path.join(OUTPUT_DIR, "mask_head_lambda_summary_nice.csv")
    nice.to_csv(nice_csv, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("Mask Head: 不同 lambda 下各指标(均值±标准差, 在", int(n_seed['n_seeds'].iloc[0]), "个seed上)")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(nice.to_string(index=False))

    # ---- 3) 绘图：每个指标一张折线图（mean + std误差棒） ----
    lambdas = mean_df["lambda_size"].values
    x_labels = [str(int(l)) if l == int(l) else str(l) for l in lambdas]

    # 单图
    for m in MASK_METRICS:
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
        ax.set_ylabel(METRIC_LABELS[m])
        ax.set_title(f"Mask Head: {METRIC_LABELS[m]} vs lambda_size "
                     f"(mean ± std over {int(n_seed['n_seeds'].iloc[0])} seeds)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = os.path.join(OUTPUT_DIR, f"mask_head_{m}_vs_lambda.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    # 综合图（一张图里画所有指标，便于横向看趋势）
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in MASK_METRICS:
        ax.errorbar(
            range(len(lambdas)),
            mean_df[m].values,
            yerr=std_df[m].values,
            marker="o", capsize=3, linewidth=1.5,
            label=METRIC_LABELS[m],
        )
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("lambda_size")
    ax.set_ylabel("score")
    ax.set_title(f"Mask Head: all metrics vs lambda_size "
                 f"(mean ± std over {int(n_seed['n_seeds'].iloc[0])} seeds)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    overall_png = os.path.join(OUTPUT_DIR, "mask_head_all_metrics_vs_lambda.png")
    fig.savefig(overall_png, dpi=150)
    plt.close(fig)

    # ---- 4) 标出每个指标的最佳 lambda ----
    print("\n" + "=" * 70)
    print("Best lambda for each Mask-Head metric (higher is better):")
    print("=" * 70)
    best_records = []
    for m in MASK_METRICS:
        idx = mean_df[m].idxmax()
        best_lambda = mean_df.loc[idx, "lambda_size"]
        best_val = mean_df.loc[idx, m]
        best_std = std_df.loc[idx, m]
        print(f"  {METRIC_LABELS[m]:<20s} -> lambda={best_lambda:<6}  "
              f"value={best_val:.4f} ± {best_std:.4f}")
        best_records.append({
            "metric": m,
            "best_lambda": best_lambda,
            "best_mean": best_val,
            "best_std":  best_std,
        })
    pd.DataFrame(best_records).to_csv(
        os.path.join(OUTPUT_DIR, "mask_head_best_lambda_per_metric.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
