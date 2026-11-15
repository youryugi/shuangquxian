# 大修阶段实验（2026-11 起）

计划来源：[paper/revision/revision_map.md](../../paper/revision/revision_map.md)（审稿意见逐条分析 → 第 3 步实验表）。本页只做导航和状态，数字和分析在各实验自己的 README 里。

## 1. 目录

| 目录 | 作用 | 文档 | 代码 | 结果 |
|---|---|---|---|---|
| `detector_comparison/` | **共享基础设施** + 实验 3。数据导出（划分 → YOLO 目录 + 官方框 + HBAT 带掩码）、统一评估口径、完整检测器对比（YOLOv8 / RT-DETR / Faster R-CNN）、YOLOv8 带监督插件 | [README](detector_comparison/README.md) | `prepare_data.py`、`evaluate.py`、`common.py`、`band_yolo.py`、`train_*.py`、`run_matrix.py`、`model_stats.py` | `datasets/`（导出的数据目录，不入库）、`runs_smoke/`（冒烟）；正式矩阵未跑 |
| `basic_ds2/` | 基础实验重跑：数据集 2 新标注、有 / 无注意力监督、干净协议；为什么没增益的诊断；band 定框做进推理及其对照 | [README](basic_ds2/README.md)（按轮次时间线，第 1–13 轮） | `train_basic.py`、`run_seeds.py`、`aggregate_cv.py`、`diag_*.py`、`diag_strata.py`、`refine_with_band.py`、`infer_bandbox.py` | `runs_cv/`（正式：16 个训练实验 × 50 run，另 6 个由推理生成的实验；`cv_summary.md` 全部配对）、`runs/`（旧协议，仅对照） |

所有 revision 实验都用 `detector_comparison/` 的数据目录和 `evaluate.py`，口径一致：IoU 0.5 贪心匹配、验证集选阈值、测试集 F1 为主指标、另报 AP50；正式结果一律 5 折合并 + 10 seed 配对 + bootstrap CI + Wilcoxon。

## 2. 实验总表（对应 revision_map 第 3 步）

| 优先级 | 实验 | 回应 | 状态 | 结论 / 位置 |
|---|---|---|---|---|
| 1 | 主表重跑：有 / 无注意力监督，两个数据集 | R2-5、R2-6 | **数据集 2 完成（两次硬件复现）；数据集 1 未跑（等 band 标注）** | ds2 fusion：ΔF1 = +0.004 [−0.021, +0.028]（5060 Ti）/ +0.019 [+0.004, +0.035]（5090），与换卡噪声同量级 → 第 4 轮；**+ band 定框：0.473 → 0.545，+0.072 [+0.059, +0.086]，10/10** → 第 12 轮 |
| 1 | 几何先验消融：矩形框 / 椭圆 / 平移带 / 抛物线带 / 只加辅助头 | R1-1 | **矩形框掩码按正式协议完成**（10 seed，含同样的 mask → 框步骤）；其余形状未做 | 框掩码 fusion +0.029 ≥ band fusion +0.019（检出增益与形状无关）；**方法比框掩码全流程高 +0.025 [+0.002, +0.046]，7/3，p = 0.11**（band 形状的贡献在框范围）→ 第 13 轮 |
| 1 | 困难子集、按可见度分层、失败案例 | R2-1、R2-2、R2-10 | 分层 recall 完成（可见度 / 完整度 / 畸变 / 宽度 / 图像上下文，10 seed 配对）；失败案例图未做 | 收益在清晰、宽、孤立目标；weak（n = 29）、成簇、窄目标 CI 盖 0，不支持"弱 / 不完整反射更不易漏检" → 第 5、11、12 轮，`diag_strata.py`，`runs_cv/strata*/summary.md` |
| 1 | 数据集 1 按类别报告（管线 / 钢筋排） | R2-1、R1-2 | 未开始（等数据集 1 带标注补齐） | |
| 1 | 完整检测器对比 + 插件实验 | R1-3、R2-3 | 代码与冒烟完成，正式矩阵未跑 | `detector_comparison/README.md` |
| 1 | 敏感性：thickness ×0.25–2、span ×0.75 / 1.25 | R1-4、R2-4 | 未开始 | |
| 2 | 敏感性：顶点偏移、slope 扰动 | R1-4 | 未开始 | |
| 2 | 跨数据集：零样本 + 少样本 | R2-7 | 未开始 | |
| 2 | **band 定框做进推理**（方法的推理步骤）+ 对照（框统一放大、框掩码）+ 上限（真实 band） | R1-1 增益来源；主表 | ds2 完成 | 方法 0.545（+0.072，10/10）；放大 1.05 对照 0.501（+0.029）；框掩码对照 0.519；真实 band 0.589 / 0.602；union 20/20 组参数有效 → 第 6、12、13 轮，`infer_bandbox.py` |
| 2 | 上限与结构诊断：oracle（真实 band 喂 fusion）、extent fusion、stride 4、λ 0.1–1.5、gate、训练集 25–100% | 解释 R1-1 | ds2 完成 | 都不改变"fusion 只有 +0.02"的结论：oracle +0.032、extent 上限 +0.043、stride 4 更差、λ / gate 无差别、数据越少 Δ 越小 → 第 7–10 轮 |
| 3 | 同等标注预算比较 | R2-4 | 未开始 | |

## 3. 当前结论与下一步

见 [basic_ds2/README.md](basic_ds2/README.md) 开头与一页总览。要点：

1. 数据集 2 上注意力监督走 fusion 的增益是 +0.004 / +0.019（两次硬件复现），和换显卡的噪声同量级；投稿版的 +0.06 来自同源泄漏和假配对。λ、fusion 方式、stride、数据量、真实 band 上限都不改变这一点（第 7–10 轮）。
2. **band 定框做进推理**（第 12 轮）：0.473 → 0.545，+0.072 [+0.059, +0.086]，10/10 胜，修的是"框画小了"的 16% 目标。
3. **框掩码对照**（第 13 轮）把 +0.072 拆开：+0.029 是任意稠密监督都有的检出增益（框掩码比 band 还多一点），+0.018 是稠密掩码定框（不显著），**band 形状特有 +0.025 [+0.002, +0.046]**，集中在宽的、孤立的双曲线；成簇 / 窄目标没动。论文主张要按这个口径改：band 的贡献在框范围，不在检出；效应 0.02–0.03 量级，和标注成本一起报。
4. **下一步优先级**：① 数据集 1 干净协议复跑（等 band 标注）→ ② 公平性补充：框宽高分开放大的对照、YOLOv8（HBAT 框）上复现并接插件 → ③ 窄 / 成簇目标的 band 预测（receptive field、软目标、换掉 Voronoi 切分）→ ④ 把 `infer_bandbox.py` 并进 `train_basic.py`、作图、写作（两次硬件复现、Discussion）。

## 4. 新开一个实验时

1. 在本目录下建 `<name>/`，README 按 [../README.md](../README.md) 的约定写（开头给答案，正文按轮次时间线）。
2. 数据用 `detector_comparison/prepare_data.py` 导出，评估调 `detector_comparison/evaluate.py`，不要另写口径。
3. 多 seed 用 `basic_ds2/run_seeds.py` / `aggregate_cv.py` 的模式：每个 run 一个目录、`run_info.json` 记全部参数、已完成的自动跳过、按 seed 配对统计。
4. 做完在本页第 2 节的表里更新状态和结论位置。
