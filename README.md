# Hyperbola-Band Attention Supervision (JSTARS-2026-06074)

GPR B-scan 双曲线检测：HBAT 参数化标注 + 双曲线带掩码显式监督注意力分支。当前处于大修（major revision）阶段。

## 目录结构

```
data/                        数据集（脚本按 <repo>/data/... 相对路径读取）；公开来源见 data/README.md
  primary_roboflow/            数据集 1（Roboflow Universe gpr-eywbm/gpr-hhalx）
    original_GPR.v1i.yolov5-obb/ 【修改稿用】v1 官方导出包，原样保存：train 408 / valid 17 / test 17，YOLOv5-OBB 官方标注
    hbat/                        【修改稿用】HBAT 输出，三个 split 共用一个 annotations.json；初始内容由 selected_318_hbat 按组名映射而来（source=legacy_init）
    all_405/                     旧：早先的导出，405 张（= 官方 train 去掉 line2888_0-95 一组）
    labels_dota_405/             旧：all_405 对应的 DOTA 格式标签
    selected_318_bbox/           旧：原论文筛选的 318 张（106 张原图 × 3），旧格式自标注 width/height/thickness，468 个目标
    selected_318_hbat/           旧：同一批 318 张图，HBAT 格式自标注 slope/span/thickness，486 个目标
  additional_utilities/        数据集 2（Mendeley, Mojahid et al.）
    original/                    官方下载包 GPR_data.rar 的管线部分，原样保存（2026-11-14 重新整理）
      original_Utilities/          【修改稿用】官方原图 131 张，无标注
      augmented_utilities/         官方增强图 786 张（131 × aug_1–6）+ annotations/（YOLO、VOC 各 786 个；官方标注只在这里，且每张增强图是分别手工标的）
    gt_boxes_official_aug6/      【修改稿用】原图的官方框：original/augmented_utilities/annotations/YOLO_format/<id>_aug_6.txt 改名为 <id>.txt，共 169 个框；HBAT 自动读取
    hbat/                        【修改稿用】HBAT 输出（annotations[_标注员].json、timelog、导出框），以后掩码也放这一层；初始内容由 legacy/raw_131 旧标注转换而来（source=legacy_init 表示尚未复核）
    legacy/                      旧论文留下的自标注（2026-11-14 移入；图片都与 original/ 逐字节相同，保留只为可追溯）
      raw_131/                     131 张原图 + 自行标注的 annotations.json（旧 width/height 格式，218 个目标）
      augmented_247_hbat/          原论文用的 247 张官方增强图（来自 71 张原图）+ HBAT 格式自标注 + 旧派生框
  splits/                      修改稿用的分组划分（全部数据、按来源组划分、5 折 + 固定测试集 + 嵌套训练子集），由 tools/splits/make_splits.py 生成

tools/
  annotation/hbat.py           HBAT 标注工具（论文 Fig. 1）。2026-11-14 改版：显示官方 GT 框并记录每条带所属的框（gt_box_id）、
                               四个目标标签（可见度/完整度/畸变/参数化状态）、顶点可放在图外、导出框裁到图内、每个目标的耗时日志（窗口右上角实时显示）、
                               按标注员分文件。用法：python hbat.py --folder <图像目录> --out <输出目录> [--annotator B] [--gt <框目录或json>] [--hide-gt] [--extend]
                               官方数据目录只读，输出一律用 --out 指到 data/<数据集>/hbat/
  annotation/seed_from_legacy.py 把旧自标注转成 HBAT 初值写入 data/<数据集>/hbat/annotations.json（已存在则不覆盖）
                               画带的函数 hyperbola_band_polygon / band_mask 在模块顶层，掩码导出和厚度扫描都要调用它们
  annotation/rect_annotator.py 旧版抛物线(width/height)标注工具加矩形视图，不是画矩形框的工具；与 legacy_bbox_annotator.py 基本相同
  annotation/legacy_bbox_annotator.py  旧版 width/height 标注工具（产生 selected_318_bbox 的标注）
  data_audit/                  修改稿用的数据与结果核查脚本
    revision_data_check.py       按审稿意见逐项检查数据能否满足（来源重叠、排除样本、可见度、分组划分、标注一致性）
    source_clusters.py           按图像相似度聚类“独立来源”，供分组划分使用
    paper_table_check.py         在全部结果 CSV 中查找与论文表格数字一致的运行
  splits/make_splits.py        生成 data/splits/ 分组划分文件（--report 查看不同相似度阈值下的分组）
  visualize_yolo_labels.py     标签可视化
  git_batch_upload.py          分批 git 提交上传

experiments/                 实验代码与结果。导航见 experiments/README.md；每个实验目录内：README.md = 文档（结论在最前），*.py = 代码，runs*/ = 结果与日志
  submission/                投稿版实验（代码冻结，只读；对应论文各表，见 experiments/submission/README.md）
    primary_yolohead/          主数据集，YOLO 式检测头，640 输入 → Table II / III / Fig. 4
    additional_centernet/      第二数据集，CenterNet 热图头，224 输入 → Table IV
    cross_dataset/             跨数据集 → Table V
    explore/                   探索分支，未进论文（已删模型权重和可视化图）
      0409_early/              早期探索（旧绝对路径和旧标注，仅供参考，不能直接运行）
      0624_backbones/          ResNet / VGG / EfficientNet / MobileNet 骨干
      0706_param_regression/   双曲线参数回归、关键点、Hough 拟合基线
      0728_hparam_primary/     主数据集 HBAT 标注上的 λ / LR 搜索
      yolo_ultralytics/        ultralytics YOLO 训练流程
  revision/                  大修阶段实验（见 experiments/revision/README.md）
    detector_comparison/       共享的数据导出 + 统一评估 + 完整检测器对比 / 带监督插件（实验 3）
    basic_ds2/                 基础实验重跑：数据集 2 新标注、干净协议（结论见其 README §0）

paper/
  manuscript/                  投稿版（JSTARS-2026-06074）：2main.tex + 1references.bib + 图；修改稿在此基础上改
  revision/                    审稿意见 reviewer_comments.txt；revision_map.md（逐条意见分析、环节总览、数据集能撑什么）
  figure_scripts/lambda_boxplot/  λ 箱线图（Fig. 6）作图脚本及数据 CSV

_待确认/                      不确定是否还需要的文件，见下文
```

## 论文表格 / 图 ↔ 代码与结果（按 paper/manuscript/2main.tex 编号）

| 论文 | 代码 | 结果 | 核对情况 |
|---|---|---|---|
| Table II（训练比例 0.5–0.9） | `experiments/submission/primary_yolohead/band_attention.py` | 0.7 行：`paper/figure_scripts/lambda_boxplot/merged_results-06281917.csv` | 只有 0.7 行找到一致的结果；0.5/0.6/0.8/0.9 行在现有 CSV 中均未找到 |
| Table III（注意力模块对比） | `experiments/submission/primary_yolohead/generic_attention_baselines.py` | `runs/attn_cnn_yolo_attnblocks_0629_0737`（训练比例 0.8） | 四个模块一致（个别 std 末位差 0.01）；“This work”行取自 Table II 0.8 行 |
| Table IV（第二数据集） | `experiments/submission/additional_centernet/centernet_band_attention.py` | 0.5 行：`runs/attn_cnn_merged_final_0731_0912`；0.7 行：`runs/attn_cnn_merged_final_0729_1127` | 一致；两行来自两次不同运行 |
| Table V（跨数据集，训练比例 0.5） | `experiments/submission/cross_dataset/cross_dataset.py` | `runs/attn_cnn_cross_dataset_0731_1005` | **Baseline 与 This work 两行写反**（见下文） |
| Fig. 4（六个场景注意力图） | `experiments/submission/primary_yolohead/band_attention_vis*.py` | 论文用图在 `paper/manuscript/fig/` | 生成该图的具体运行未确认 |
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
