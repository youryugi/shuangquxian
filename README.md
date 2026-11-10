# Hyperbola-Band Attention Supervision (JSTARS-2026-06074)

GPR B-scan 双曲线检测：HBAT 参数化标注 + 双曲线带掩码显式监督注意力分支。当前处于大修（major revision）阶段。

## 目录结构

```
data/                        数据集（脚本按 <repo>/data/... 相对路径读取）；公开来源见 data/README.md
  primary_roboflow/            主数据集（Roboflow Universe gpr-eywbm/gpr-hhalx）
    all_405/                     原始导出 405 张（135 张原图 × 3 个 Roboflow 增强副本）
    labels_dota_405/             Roboflow 原始 DOTA 格式标签
    selected_318_bbox/           筛选后 318 张（106 张原图 × 3），旧格式标注 width/height/thickness，468 个目标
    selected_318_hbat/           同一批 318 张图，HBAT 格式标注 slope/span/thickness，486 个目标
  additional_utilities/        第二数据集（Mendeley, Mojahid et al.）
    raw_131/                     官方原图 131 张 + 标注（旧格式）
    augmented_247_hbat/          官方增强图 247 张（来自 71 张原图，每张 1–4 个 _aug_N 副本），HBAT 格式

tools/
  annotation/hbat.py           HBAT 标注工具（论文 Fig. 1）
  annotation/rect_annotator.py 矩形框标注 / 双曲线↔矩形查看
  annotation/legacy_bbox_annotator.py  旧版 width/height 标注工具（产生 selected_318_bbox 的标注）
  data_audit/                  修改稿用的数据与结果核查脚本
    revision_data_check.py       按审稿意见逐项检查数据能否满足（来源重叠、排除样本、可见度、分组划分、标注一致性）
    source_clusters.py           按图像相似度聚类“独立来源”，供分组划分使用
    paper_table_check.py         在全部结果 CSV 中查找与论文表格数字一致的运行
  visualize_yolo_labels.py     标签可视化
  git_batch_upload.py          分批 git 提交上传

experiments/                 每个目录内：顶层 .py = 当前代码；runs/<脚本名>_<MMDD_HHMM>/ = 历次运行（含当次脚本快照 + 结果 CSV）
  primary_yolohead/            主数据集，YOLO 式检测头，640 输入，80 epoch
    band_attention.py            本文方法 vs baseline（训练比例、λ 扫描）
    band_attention_vis.py        同上 + 注意力热图可视化
    band_attention_vis_hbat.py   同上，读 HBAT 版标注
    generic_attention_baselines.py  SE / CBAM / Non-local / CA 对比
  additional_centernet/        第二数据集，CenterNet 热图检测头，224 输入
  cross_dataset/               跨数据集测试（CenterNet 头）
  explore_*/                   探索阶段代码，未进入论文；已删除其模型权重和可视化图
    explore_0409_early/          早期探索（脚本用的是旧绝对路径和旧标注，仅供参考，不能直接运行）
    explore_0624_backbones/      ResNet / VGG / EfficientNet / MobileNet 骨干 + 主数据集 CenterNet 头
    explore_0706_param_regression/  双曲线参数回归、关键点、Hough 拟合基线
    explore_0728_hparam_primary/ 主数据集 HBAT 标注上的 λ/LR 搜索
    explore_yolo_ultralytics/    ultralytics YOLO 训练流程

paper/
  manuscript/                  投稿版（JSTARS-2026-06074）：2main.tex + 1references.bib + 图；修改稿在此基础上改
  revision/                    审稿意见 reviewer_comments.txt、回复模板
  figure_scripts/lambda_boxplot/  λ 箱线图（Fig. 6）作图脚本及数据 CSV

_待确认/                      不确定是否还需要的文件，见下文
```

## 论文表格 / 图 ↔ 代码与结果（按 paper/manuscript/2main.tex 编号）

| 论文 | 代码 | 结果 | 核对情况 |
|---|---|---|---|
| Table II（训练比例 0.5–0.9） | `experiments/primary_yolohead/band_attention.py` | 0.7 行：`paper/figure_scripts/lambda_boxplot/merged_results-06281917.csv` | 只有 0.7 行找到一致的结果；0.5/0.6/0.8/0.9 行在现有 CSV 中均未找到 |
| Table III（注意力模块对比） | `experiments/primary_yolohead/generic_attention_baselines.py` | `runs/attn_cnn_yolo_attnblocks_0629_0737`（训练比例 0.8） | 四个模块一致（个别 std 末位差 0.01）；“This work”行取自 Table II 0.8 行 |
| Table IV（第二数据集） | `experiments/additional_centernet/centernet_band_attention.py` | 0.5 行：`runs/attn_cnn_merged_final_0731_0912`；0.7 行：`runs/attn_cnn_merged_final_0729_1127` | 一致；两行来自两次不同运行 |
| Table V（跨数据集，训练比例 0.5） | `experiments/cross_dataset/cross_dataset.py` | `runs/attn_cnn_cross_dataset_0731_1005` | **Baseline 与 This work 两行写反**（见下文） |
| Fig. 4（六个场景注意力图） | `experiments/primary_yolohead/band_attention_vis*.py` | 论文用图在 `paper/manuscript/fig/` | 生成该图的具体运行未确认 |
| Fig. 5（Grad-CAM） | 同上 | 论文用图在 `paper/manuscript/` | 生成该图的具体运行未确认 |
| Fig. 6（λ 箱线图） | `paper/figure_scripts/lambda_boxplot/plot_boxplot_lam.py` | `merged_results-06281917.csv` | 论文引用 `boxplot_f1_none_vs_abs_lam2.pdf`，脚本输出名是 `..._lam.pdf`，`lam2` 版本的生成方式未找到 |

### Table V 行标签写反

`cross_dataset_results.csv` 中按 `mode` 汇总（10 个 seed）：

| 方向 | mode=none（baseline） P / R / F1 | mode=abs（本文） P / R / F1 |
|---|---|---|
| Primary→Additional | 0.260 / 0.080 / 0.114 | 0.271 / 0.090 / 0.125 |
| Additional→Primary | 0.116 / 0.142 / 0.124 | 0.117 / 0.166 / 0.135 |

论文把 abs 的数字填在了 Baseline 行。两个方向上本文方法实际都略好（F1 +0.01），但差距在一个标准差以内。审稿意见 R2-7 引用了这张表。

## 已知问题（修改稿需处理）

- 数据泄漏：现有划分按图片随机打乱，同一原图的增强副本会同时进入训练集和测试集。另外，不同文件名之间也存在同一场景或同一剖面的重复。需要用 `tools/data_audit/source_clusters.py` 聚类后按来源分组划分。
- 主数据集实验与第二数据集 / 跨数据集实验用的是不同检测头、输入尺寸和 epoch。
- Table II/III 用 `selected_318_bbox` 标注，Table V 用 `selected_318_hbat` 标注。

## _待确认

| 文件 | 说明 |
|---|---|
| `utilities_annotation_versions/` | `raw_131` 的另外两个标注版本（164 / 219 个目标）；不确定 `augmented_247_hbat` 由哪个版本生成 |
| `selected_318_old_annotations_rect.json` | 318 张主数据集的旧版矩形标注 |
| `hbat_simple_0701.py` | HBAT 的早期简化版（比 `hbat.py` 早 3 天） |
| `manuscript_unused_figures/` | 投稿版目录中未被 2main.tex 引用的图：`bscan.png`（引用已注释）、`example-011/030`（第二数据集样例）、`11.png`、`unet.pdf` |

`augmented_247_hbat` 的增强脚本不在仓库中；302 条标注里有 55 张图片已被删除，只剩 247 张。

旧草稿（原 `paper/submission_v1`，即 git 中的 `writepaper/IEEE-final`）已删除，可从 git 历史恢复；它的 bib 只比投稿版多一条 `rw-gprsurvey2`（GPR 深度学习综述，Construction and Building Materials 2020）。
