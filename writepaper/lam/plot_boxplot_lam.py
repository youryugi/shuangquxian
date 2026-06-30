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
IEEE_DIR = Path(r"C:\Users\79152\Desktop\github\shuangquxian\writepaper\IEEE")   # PDF 另存目录

METRIC = "bbox_F1"
LAM_ORDER = ["0.1", "0.3", "0.5", "0.7", "1", "3", "5","7","10"]   # abs 的 lam 排序
ORDER = ["none"] + LAM_ORDER                               # x 轴顺序, none 作基线

# ---------------------------------------------------------------- 期刊风格
# 无衬线专业字体 + 紧凑排版, 适合论文单栏插图
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


def main() -> None:
    df = pd.read_csv(CSV)
    # 统一分组标签: none 单独一组, abs 用其 lam 值
    df["group"] = np.where(df["config"] == "none", "none", df["lam_att"].astype(str))
    # 只保留 ORDER 里定义的组, 丢掉空行(config 为 NaN)和未纳入的 lam 值
    df = df[df["group"].isin(ORDER)]

    # 颜色: 所有箱体统一浅灰填充 + 黑色边框; 黑白印刷也清晰
    palette = {g: (0.88, 0.88, 0.88) for g in ORDER}

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sns.boxplot(
        data=df, x="group", y=METRIC, order=ORDER,
        hue="group", palette=palette, legend=False,
        width=0.62,
        linewidth=0.9,
        fliersize=0,            # 不画离群点
        showcaps=True,
        boxprops={"edgecolor": "black"},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
        medianprops={"color": "black", "linewidth": 1.3},
        ax=ax,
    )

    ax.set_xlabel(r"$\lambda$ (none = baseline)")
    ax.set_ylabel("F1")
    ax.set_ylim(0.5, 0.93)

    # 去掉上/右边框, 网格仅保留淡化的 y 方向
    sns.despine(ax=ax, top=True, right=True)
    ax.yaxis.grid(True, color="0.85", linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")   # 矢量图供排版
    # 另存一份 PDF 到 IEEE 目录
    IEEE_DIR.mkdir(parents=True, exist_ok=True)
    ieee_pdf = IEEE_DIR / OUT.with_suffix(".pdf").name
    fig.savefig(ieee_pdf, bbox_inches="tight")
    print(f"saved: {OUT}")
    print(f"saved: {ieee_pdf}")
    plt.close(fig)

    # 各组 F1 的中位数/均值/标准差: 打印并保存为 CSV
    stat = df.groupby("group")[METRIC].agg(["count", "median", "mean", "std"]).reindex(ORDER)
    print(stat.round(4))
    stat_csv = OUT.with_name("boxplot_f1_none_vs_abs_lam_stats.csv")
    stat.round(4).to_csv(stat_csv, index_label="group")
    print(f"saved: {stat_csv}")


if __name__ == "__main__":
    main()
