# 大修阶段实验（2026-11 起）

计划来源：[paper/revision/revision_map.md](../../paper/revision/revision_map.md)（审稿意见逐条分析 → 第 3 步实验表）。本页只做导航和状态，数字和分析在各实验自己的 README 里。

## 1. 目录（2026-11-22 按实验整理）

| 目录 | 实验 | 作用 | 文档 | 状态 |
|---|---|---|---|---|
| `exp1_basic/` | **实验 1：基础实验** | 本文轻量 CenterNet 上有无 band attention（严格协议）→ band 用来定框（band + hyperbola fit）→ 和 box-mask 对照比、数据效率、grouped targets | [README](exp1_basic/README.md)（答案、总览、下一步、文件位置）+ [rounds.md](exp1_basic/rounds.md)（逐轮记录） | **数据集 2 完成**；数据集 1 等 band 标注 |
| `exp2_annotation_sensitivity/` | **实验 2：标注敏感性** | 扰动 HBAT 参数（厚度、跨度、顶点、slope、重复标注误差）重画监督掩码，看 band + fit 的提升还在不在（R1-4、R2-4、R1-2） | [README](exp2_annotation_sensitivity/README.md) | **数据集 2 完成**；两位标注员之间的一致性等第二位标注员 |
| `exp3_comparison/` | **实验 3：对比** | 和成熟检测器比：`detectors/` 是 YOLOv8s / Faster R-CNN 等的对比表和 band 插件代码（`band_yolo.py`）；`yolo_band/` 是 band 辅助监督插进 YOLOv8s 的实验（预训练、从零，box-mask 对照，不重训的诊断） | [detectors/README](exp3_comparison/detectors/README.md)、[yolo_band/README](exp3_comparison/yolo_band/README.md) | **数据集 2：和成熟模型的对比完成**（YOLOv8s 预训练 / 从零、Faster R-CNN 预训练） |
| `exp4_attention/` | **实验 4：注意力机制对比** | 投稿版 Table III 的四个通用注意力模块（SE、CBAM、Non-local、CA）在实验 1 的网络和协议下重做，加 attention 分支不监督的对照；和 band fusion、band + fit 配对比较 | [README](exp4_attention/README.md) | **数据集 2 完成** |
| `shared/` | 共用 | 数据导出（`prepare_data.py` → `datasets/`，不入库）、统一评估（`evaluate.py`）、`common.py` | [README](shared/README.md) | |

依赖关系：四个实验都读 `shared/datasets/`、调 `shared/evaluate.py`。实验 2 的训练和后处理借用实验 1 的管线（`run_seeds.py`、`infer_bandbox.py`、`aggregate_cv.py`），750 个 run 也放在 `exp1_basic/runs_cv/ds2/` 下以便和实验 1 的 baseline、box-mask 配对；实验 3 的 `yolo_band/` 用 `detectors/` 的训练脚本，聚合和分层分析借用实验 1 的 `aggregate_cv.py`、`diag_strata.py`、`fit_hyperbola.py`。实验 4 同实验 2：训练和后处理用实验 1 的 `train_basic.py --attn-block`、`run_seeds.py`、`infer_bandbox.py`，250 个 run 放在 `exp1_basic/runs_cv/ds2/hbatbox_flipfix_{se,cbam,nonlocal,coord,unsup}*`，模块代码在 `shared/attn_blocks.py`。

所有 revision 实验的评估口径一致：预测框先裁到图内再匹配（2026-11-21 起）、IoU 0.5 greedy matching、validation set 选 threshold、test F1 为主指标、另报 AP50；正式结果一律 5-fold 合并 + 10 seeds 配对 + bootstrap CI + Wilcoxon。

## 2. 实验总表（对应 revision_map 第 3 步）

| 优先级 | 实验 | 回应 | 状态 | 结论 / 位置（Round 见 exp1_basic/rounds.md） |
|---|---|---|---|---|
| 1 | 主表重跑：有 / 无 attention，两个数据集 | R2-5、R2-6 | **数据集 2 完成**（Round 4 两次硬件复现；Round 18 修好翻转后的正式结果）；数据集 1 未跑（等 band 标注） | fusion +0.023（8/2），和换卡噪声同量级；**band + hyperbola fit 0.472 → 0.585（+0.112，10/10）** → Round 4、18 |
| 1 | 固定测试集 + 递增训练集（替换 Table II） | R2-6、R2-8 | **数据集 2 完成** | 训练数据 25 / 50 / 75 / 100%：band + fit 每个比例都赢 baseline（+0.170 / +0.138 / +0.098 / +0.112，10/10）和 box-mask 最好的后处理（+0.044 到 +0.055）；fusion 任何比例都不显著 → Round 19 |
| 1 | 几何先验消融：矩形框 / 椭圆 / 平移带 / 抛物线带 / 只加辅助头 | R1-1 | 矩形框（box-mask control）按正式协议完成；其余形状未做 | fusion 上 band 对 box-mask −0.008（n.s.）；按形状定框后 band 对 box-mask + fit **+0.086**、对 box-mask 最好的后处理 **+0.044**（都 10/10），同参数 145/180 → Round 13、18 |
| 1 | 困难子集、按可见度分层、失败案例 | R2-1、R2-2、R2-10 | 数据集 2 分层 recall 完成；CNR / 杂波未算；失败案例图有工具（`vis_errors.py`），未选图 | baseline → band + fit：单翼 +0.142（8/2）、畸变 +0.079（6/2/2）、weak +0.069（n.s.）；grouped 对 baseline +0.011（n.s.），对 box-mask +0.105 → Round 18 |
| 1 | 数据集 1 按类别报告（管线 / 钢筋排） | R2-1、R1-2 | 未开始（等 band 标注） | |
| 1 | 完整检测器对比 + 插件实验 | R1-3、R2-3 | **数据集 2 完成**（实验 3）：YOLOv8s 预训练 / 从零、Faster R-CNN 预训练，5 折 × 10 seeds；从零 YOLOv8s 的 box-mask 对照、预训练只翻转 YOLOv8s + band 未跑 | 条件对齐（都不预训练、只翻转）时本文 + band 0.496 对 YOLOv8s 0.452；band 在从零 YOLOv8s 上 +0.034（8/2），在预训练 YOLOv8s 上不显著；Faster R-CNN 预训练 0.724 最高 → [detectors/README](exp3_comparison/detectors/README.md) 0.7 节、[yolo_band/README](exp3_comparison/yolo_band/README.md) |
| 1 | 通用注意力模块对比（SE、CBAM、Non-local、CA，替换 Table III） | R1-3、R2-3、R1-1 | **数据集 2 完成**（实验 4） | 只比注意力模块：四个模块都赢 baseline（+0.031 到 +0.058），band fusion 对 SE / CBAM / CA 打平、对 Non-local −0.034（0/10）；同结构不监督对 band 监督 +0.006（n.s.）。**band + fit 赢全部四个模块**（+0.055 到 +0.081；模块加 scale control 后仍 +0.049 到 +0.059，10/10）；不监督的 attention 走 fit 只有 0.466 → [实验 4](exp4_attention/README.md) |
| 1 | 敏感性：thickness ×0.25–2、统一厚度、span ×0.75 / 1.25 | R1-4、R2-4、R1-2 | **数据集 2 完成**（实验 2） | 厚度 ×0.5–×1.25 和统一厚度不显著（0.578–0.583 对 0.585）；跨度 ±25% 掉 −0.047；全部档位赢 baseline → [实验 2](exp2_annotation_sensitivity/README.md) |
| 2 | 敏感性：顶点偏移、slope 扰动、重复标注误差 | R1-4 | **数据集 2 完成**（实验 2） | 实测误差水平（顶点 2 px、slope ±10%、同一人重复标注）下不显著，仍对 box-mask 最好的后处理 +0.03；误差翻倍后对 box-mask 不再显著 → [实验 2](exp2_annotation_sensitivity/README.md) |
| 2 | 跨数据集：零样本 + 少样本 | R2-7 | 未开始 | |
| 2 | box refinement 做进推理 + scale control + upper bound | R1-1 增益来源 | ds2 完成（Round 18 取代 Round 6、12 的数字） | band + fit 对 scale control +0.093（10/10）；真实 band + fit 上限 0.662 → Round 6、12、18 |
| 2 | 上限与结构诊断：oracle fusion、extent、stride 4、λ、gate、训练集 25–100% | 解释 R1-1 | ds2 完成 | 都不改变"fusion 只有 +0.02" → Round 7–10（训练集比例由 Round 19 取代） |
| 2 | stage 2 框回归（按每条双曲线，只用框标注）+ band / boxmask / oracle arm | 对照（11-21 决定不在主线上） | ds2 完成 | +0.159（10/10）；band 在 stage 2 上无额外作用。它不用 band，证明的是"再回归一次框有用"，不作为 band 方法要超过的对照 → Round 14 |
| 2 | **hyperbola fit** 代替 Voronoi split；**U-shaped decoder** attention head | grouped targets；band 质量 | ds2 完成 | fit 是 Round 18 主结论的后处理；预测 band 测试 IoU 只到 0.42（受数据量限制）→ Round 15、16、18 |
| 2 | 参数化检测头（网络直接输出双曲线参数） | band 标注的其他用法 | ds2 完成 | 参数化 +0.08，但只用框反推参数的 pseudo 拿到几乎全部；顺带发现翻转增强 bug → Round 17 |
| 2 | 形状外推、曲线归属、加权损失；fit 之后的配对规则 | grouped targets | ds2 完成（实验 1 Round 20、22） | Round 20 四种补法都没有提升；Round 22 只有"解释掉再拟合"对 band 有小幅提升（+0.010 / +0.012，8/10）→ [实验 1 README](exp1_basic/README.md) |
| 3 | 同等标注预算比较 | R2-4 | 未开始 | |

## 3. 当前结论与下一步

每个实验的结论只写在它自己 README 的开头："答案"和"下一步"：[实验 1](exp1_basic/README.md)（逐轮过程在 [rounds.md](exp1_basic/rounds.md)）、[实验 2](exp2_annotation_sensitivity/README.md)、实验 3（[detectors](exp3_comparison/detectors/README.md)、[yolo_band](exp3_comparison/yolo_band/README.md)）。上面第 2 节只放一句话结论和链接。

## 4. 新开一个实验时

1. 在本目录下建 `exp<N>_<name>/`（编号接着现有实验），README 按[根目录 README 的"写文档的约定"](../../README.md#写文档的约定)写：开头给答案，正文按 Round 时间线，每轮"为什么做 / 做了什么 / 结果 / 结论"，次要的大表放进 `<details>` 折叠，术语用英文并在开头放术语表。
2. 数据用 `shared/prepare_data.py` 导出，评估调 `shared/evaluate.py`，不要另写口径。
3. 多 seed 用 `exp1_basic/run_seeds.py` / `aggregate_cv.py` 的模式：每个 run 一个目录、`run_info.json` 记全部参数、已完成的自动跳过、按 seed 配对统计。推理步骤的参数在合并 validation set 上全局选。
4. 做完在本页第 2 节更新状态和结论位置。
