# 投稿版实验（JSTARS-2026-06074）

**状态：代码冻结，只读。** 这里的脚本和 `runs/` 是论文投稿版各表的来源，修改稿不在这里改代码；新实验一律在 [../revision/](../revision/README.md)。
脚本里的仓库根路径已在 2026-11-15 目录重排时同步修正（多了一层 `submission/`），可以原地运行；`explore/0409_early/` 除外（旧绝对路径、旧标注，仅供参考）。

## 1. 论文表格 ↔ 脚本 ↔ 运行记录

| 论文 | 目录 / 脚本 | 结果位置 | 核对情况 |
|---|---|---|---|
| Table II（主数据集，训练比例 0.5–0.9） | `primary_yolohead/band_attention.py` | 0.7 行：`paper/figure_scripts/lambda_boxplot/merged_results-06281917.csv` | 只有 0.7 行找到一致结果；0.5 / 0.6 / 0.8 / 0.9 行在现有 CSV 中未找到（S3） |
| Table III（注意力模块对比） | `primary_yolohead/generic_attention_baselines.py` | `primary_yolohead/runs/attn_cnn_yolo_attnblocks_0629_0737`（训练比例 0.8） | 四个模块一致；"This work" 行取自 Table II 0.8 行 |
| Table IV（第二数据集） | `additional_centernet/centernet_band_attention.py` | 0.5 行：`additional_centernet/runs/attn_cnn_merged_final_0731_0912`；0.7 行：`.../attn_cnn_merged_final_0729_1127` | 一致；两行来自两次不同运行 |
| Table V（跨数据集，训练比例 0.5） | `cross_dataset/cross_dataset.py` | `cross_dataset/runs/attn_cnn_cross_dataset_0731_1005` | **Baseline 与 This work 两行写反**（S1） |
| Fig. 4（六个场景注意力图） | `primary_yolohead/band_attention_vis*.py` | 论文用图在 `paper/manuscript/fig/` | 生成该图的具体运行未确认 |

`runs/<脚本名>_<MMDD_HHMM>/` 每个目录含当次脚本快照 + 结果 CSV。

## 2. 目录

| 目录 | 内容 | 数据集 | 检测头 |
|---|---|---|---|
| `primary_yolohead/` | 本文方法 vs baseline（训练比例、λ 扫描）、通用注意力模块对比、注意力可视化 | 数据集 1（`data/primary_roboflow/selected_318_*`） | YOLO 式，640 输入，80 epoch |
| `additional_centernet/` | 本文方法 vs baseline；`hparam_search.py` 超参搜索 | 数据集 2（`data/additional_utilities/legacy/augmented_247_hbat`，247 张官方增强图） | CenterNet 热图头，224 输入 |
| `cross_dataset/` | 两个数据集互相迁移 | 1 ↔ 2 | CenterNet 头 |
| `explore/` | 未进论文的探索分支，见下 | | |

### explore/（未进论文）

| 目录 | 内容 | 对修改稿的参考价值 |
|---|---|---|
| `0409_early/` | 最早的探索：CNN 检测头 + 显式注意力监督、门控设计、YOLO 式头手写尝试、Grad-CAM | 关键发现：门控应"增强带"不要"抑制背景"；注意力增益只在弱检测器上出现（+0.03，5 seed p≈0.04），预训练 ResNet18 上反而有害。**不能直接运行** |
| `0624_backbones/` | ResNet / VGG / EfficientNet / MobileNet 骨干 + CenterNet 头 | 预训练骨干上注意力门控有害的证据 |
| `0706_param_regression/` | 双曲线参数回归、关键点回归、Hough 拟合基线 | 带级 / 参数化输出的起点（与"带表示优于框"方向相关） |
| `0728_hparam_primary/` | 主数据集 HBAT 标注上的 λ / LR 搜索 | λ 的取值依据 |
| `yolo_ultralytics/` | ultralytics YOLO 最小训练流程 | 已被 `revision/detector_comparison/` 取代 |

## 3. 投稿版已知问题（修改稿必须处理）

详见 [revision_map.md §4 自查发现](../../paper/revision/revision_map.md#4-自查发现)。

| 编号 | 问题 | 影响 |
|---|---|---|
| S1 | Table V 的 Baseline 与 This work 两行写反 | 跨数据集结论方向 |
| S2 | 监督掩码与论文式 (1) 不一致：代码画的是抛物线，论文写的是 HBAT 双曲线带 | R1-1、R1-4、R2-4 |
| S3 | Table II 只有 0.7 行能找到原始结果 | 可复现性 |
| S4 | 两个数据集用了不同的检测器（YOLO 式头 vs CenterNet 头） | 对比的公平性 |
| S5 | 投稿版实验有同源泄漏（数据集 2 测试图 84% 在训练集有同一原图的增强副本）且 seed 配对是假的；干净协议下注意力监督增益为 +0.004 | **论文主张**，见 `revision/basic_ds2/README.md` |

## 4. 旧实验与新实验的对应

| 投稿版 | 修改稿 |
|---|---|
| `additional_centernet/centernet_band_attention.py`（Table IV） | `revision/basic_ds2/train_basic.py`（同一网络、同一损失；改了数据、划分、掩码来源、初始化配对、评估口径） |
| 随机 50/50 切、无验证集、3 seed | 按来源分组 5 折、合并验证集选阈值、10 seed 配对 + CI |
| 自画抛物线掩码 | HBAT 导出的双曲线带 PNG |
| 247 张官方增强图 | 131 张原图（增强只在训练时做） |
