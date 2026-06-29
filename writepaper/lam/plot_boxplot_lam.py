# -*- coding: utf-8 -*-
"""
none vs abs(不同 lam_att 取值)的 BBox F1 对比箱线图。

数据结构 (merged_results-*.csv):
    train_frac, seed, config, lam_att, bbox_P, bbox_R, bbox_F1, attn_band_iou, attn_gap
    - config == "none": 基线, 没有 lam (lam_att 为 "-")
    - config == "abs" : 每个 lam_att 取值 0.1/0.3/0.5/0.7/1/3/5, 各 10 个 seed

每个箱子 = 同一配置在 10 个 seed 上的 F1 分布。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------- 配置
HERE = Path(__file__).resolve().parent
CSV = HERE / "merged_results-06281917.csv"
OUT = HERE / "boxplot_f1_none_vs_abs_lam.png"

METRIC = "bbox_F1"
LAM_ORDER = ["0.1", "0.3", "0.5", "0.7", "1", "3", "5"]   # abs 的 lam 排序
ORDER = ["none"] + LAM_ORDER                               # x 轴顺序, none 作基线

# 中文字体, 避免出现方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="talk", font="Microsoft YaHei")


def main() -> None:
    df = pd.read_csv(CSV)
    # 统一分组标签: none 单独一组, abs 用其 lam 值
    df["group"] = np.where(df["config"] == "none", "none", df["lam_att"].astype(str))

    # 颜色: none 用灰色, abs 各 lam 用渐变色带
    abs_colors = sns.color_palette("viridis", len(LAM_ORDER))
    palette = {"none": (0.6, 0.6, 0.6)} | dict(zip(LAM_ORDER, abs_colors))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, x="group", y=METRIC, order=ORDER,
        hue="group", palette=palette, legend=False,
        width=0.6,
        showfliers=False,   # 不画散点, 也不画离群点, 只保留箱体和须线
        ax=ax,
    )

    ax.set_title("none vs abs:不同 lam_att 的 BBox F1 分布")
    ax.set_xlabel("lam_att(none = 基线)")
    ax.set_ylabel("BBox F1")
    ax.set_ylim(0.5, 0.93)   # 纵轴从 0.5 开始

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved: {OUT}")
    plt.close(fig)

    # 顺手打印各组 F1 的中位数/均值/标准差
    stat = df.groupby("group")[METRIC].agg(["median", "mean", "std"]).reindex(ORDER)
    print(stat.round(4))


if __name__ == "__main__":
    main()
