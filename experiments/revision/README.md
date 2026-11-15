# 大修阶段实验（2026-11 起）

计划来源：[paper/revision/revision_map.md](../../paper/revision/revision_map.md)（审稿意见逐条分析 → 第 3 步实验表）。本页只做导航和状态，数字和分析在各实验自己的 README 里。

## 1. 目录

| 目录 | 作用 | 文档 | 代码 | 结果 |
|---|---|---|---|---|
| `detector_comparison/` | **共享基础设施** + 实验 3。数据导出（划分 → YOLO 目录 + 官方框 + HBAT 带掩码）、统一评估口径、完整检测器对比（YOLOv8 / RT-DETR / Faster R-CNN）、YOLOv8 带监督插件 | [README](detector_comparison/README.md) | `prepare_data.py`、`evaluate.py`、`common.py`、`band_yolo.py`、`train_*.py`、`run_matrix.py`、`model_stats.py` | `datasets/`（导出的数据目录，不入库）、`runs_smoke/`（冒烟）；正式矩阵未跑 |
| `basic_ds2/` | 基础实验重跑：数据集 2 新标注、有 / 无注意力监督、干净协议；以及为什么没增益的诊断 | [README](basic_ds2/README.md)（按轮次时间线） | `train_basic.py`、`run_seeds.py`、`aggregate_cv.py`、`diag_*.py`、`refine_with_band.py` | `runs_cv/`（正式，200 run + 带修正）、`runs/`（旧协议，仅对照） |

所有 revision 实验都用 `detector_comparison/` 的数据目录和 `evaluate.py`，口径一致：IoU 0.5 贪心匹配、验证集选阈值、测试集 F1 为主指标、另报 AP50；正式结果一律 5 折合并 + 10 seed 配对 + bootstrap CI + Wilcoxon。

## 2. 实验总表（对应 revision_map 第 3 步）

| 优先级 | 实验 | 回应 | 状态 | 结论 / 位置 |
|---|---|---|---|---|
| 1 | 主表重跑：有 / 无注意力监督，两个数据集 | R2-5、R2-6 | **数据集 2 完成；数据集 1 未跑** | ds2：ΔF1 = +0.004 [−0.021, +0.028]，无增益 → `basic_ds2/README.md` 第 4 轮 |
| 1 | 几何先验消融：矩形框 / 椭圆 / 平移带 / 抛物线带 / 只加辅助头 | R1-1 | 只有框掩码单种子对照（0.308，更差） | `basic_ds2/README.md` 第 2 轮 |
| 1 | 困难子集、按可见度分层、失败案例 | R2-1、R2-2、R2-10 | 部分：漏检分类与难目标特征 | `basic_ds2/README.md` 第 5 轮 |
| 1 | 数据集 1 按类别报告（管线 / 钢筋排） | R2-1、R1-2 | 未开始（等数据集 1 带标注补齐） | |
| 1 | 完整检测器对比 + 插件实验 | R1-3、R2-3 | 代码与冒烟完成，正式矩阵未跑 | `detector_comparison/README.md` |
| 1 | 敏感性：thickness ×0.25–2、span ×0.75 / 1.25 | R1-4、R2-4 | 未开始 | |
| 2 | 敏感性：顶点偏移、slope 扰动 | R1-4 | 未开始 | |
| 2 | 跨数据集：零样本 + 少样本 | R2-7 | 未开始 | |
| 2 | 带修正框范围（输出机制，不进论文主张） | 解释 R1-1 增益来源 | ds2 完成 | F1 0.479 → 0.541，10 seed 全胜；真值带上限 0.587 → `basic_ds2/README.md` 第 6 轮 |
| 3 | 同等标注预算比较 | R2-4 | 未开始 | |

## 3. 当前结论与下一步

见 [basic_ds2/README.md](basic_ds2/README.md) 开头与一页总览。要点：

1. 数据集 2 上注意力监督对检测没有可测出的增益；投稿版的增益来自同源泄漏和假配对。
2. 注意力分支学到了框所缺的"范围"信息，但没进入框回归（带修正框实验证明）。
3. **下一步优先级**：① 数据集 1 干净协议复跑（决定论文主张）→ ② 范围信息做进网络（带外接框监督 wh / GIoU）→ ③ 提升带完整度（注意力头移到 112×112、软目标）→ ④ λ 扫描、gate 融合、上限诊断。

## 4. 新开一个实验时

1. 在本目录下建 `<name>/`，README 按 [../README.md](../README.md) 的约定写（开头给答案，正文按轮次时间线）。
2. 数据用 `detector_comparison/prepare_data.py` 导出，评估调 `detector_comparison/evaluate.py`，不要另写口径。
3. 多 seed 用 `basic_ds2/run_seeds.py` / `aggregate_cv.py` 的模式：每个 run 一个目录、`run_info.json` 记全部参数、已完成的自动跳过、按 seed 配对统计。
4. 做完在本页第 2 节的表里更新状态和结论位置。
