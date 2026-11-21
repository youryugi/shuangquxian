# 大修阶段实验（2026-11 起）

计划来源：[paper/revision/revision_map.md](../../paper/revision/revision_map.md)（审稿意见逐条分析 → 第 3 步实验表）。本页只做导航和状态，数字和分析在各实验自己的 README 里。

## 1. 目录

| 目录 | 作用 | 文档 | 代码 | 结果 |
|---|---|---|---|---|
| `detector_comparison/` | **共享基础设施** + 实验 3。数据导出（划分 → YOLO 目录 + official box + HBAT band mask）、统一评估口径、完整检测器对比（YOLOv8 / RT-DETR / Faster R-CNN）、YOLOv8 band 辅助监督插件 | [README](detector_comparison/README.md) | `prepare_data.py`、`evaluate.py`、`common.py`、`band_yolo.py`、`train_*.py`、`run_matrix.py`、`model_stats.py` | `datasets/`（不入库）、`runs_smoke/`；检测器对比矩阵未跑（插件的正式结果在 `yolo_band_ds2/`） |
| `yolo_band_ds2/` | 第二个实验：band 监督插进 COCO 预训练的 YOLOv8s（approach A），同样的严格协议；box-mask control、A/A control、band IoU；不重训的诊断（真实 band + fit 修框、截断双曲线的参数估计） | [README](yolo_band_ds2/README.md) | `run_cv.py`（训练用 `detector_comparison/train_ultralytics.py`、`band_yolo.py`） | `runs_cv/` |
| `basic_ds2/` | 基础实验重跑（数据集 2，新标注，严格协议）：attention 有没有用 → 为什么没有 → band 用在 box refinement → band 形状有没有不可替代的价值 | [README](basic_ds2/README.md)（答案、总览、下一步、文件位置）+ [rounds.md](basic_ds2/rounds.md)（Round 1–19 逐轮记录） | `train_basic.py`、`run_seeds.py`、`aggregate_cv.py`、`infer_bandbox.py`、`stage2_instance.py`、`fit_hyperbola.py`、`diag_*.py` | `runs_cv/`（正式）、`runs/`（旧协议，仅对照） |

所有 revision 实验都用 `detector_comparison/` 的数据目录和 `evaluate.py`：预测框先裁到图内再匹配（2026-11-21 起）、IoU 0.5 greedy matching、validation set 选 threshold、test F1 为主指标、另报 AP50；正式结果一律 5-fold 合并 + 10 seeds 配对 + bootstrap CI + Wilcoxon。

## 2. 实验总表（对应 revision_map 第 3 步）

| 优先级 | 实验 | 回应 | 状态 | 结论 / 位置（Round 见 basic_ds2/rounds.md） |
|---|---|---|---|---|
| 1 | 主表重跑：有 / 无 attention，两个数据集 | R2-5、R2-6 | **数据集 2 完成**（Round 4 两次硬件复现；Round 18 修好翻转后的正式结果）；数据集 1 未跑（等 band 标注） | fusion +0.023（8/2），和换卡噪声同量级；**band + hyperbola fit 0.472 → 0.585（+0.112，10/10）** → Round 4、18 |
| 1 | 固定测试集 + 递增训练集（替换 Table II） | R2-6、R2-8 | **数据集 2 完成** | 训练数据 25 / 50 / 75 / 100%：band + fit 每个比例都赢 baseline（+0.170 / +0.138 / +0.098 / +0.112，10/10）和 box-mask 最好的后处理（+0.044 到 +0.055）；fusion 任何比例都不显著 → Round 19 |
| 1 | 几何先验消融：矩形框 / 椭圆 / 平移带 / 抛物线带 / 只加辅助头 | R1-1 | 矩形框（box-mask control）按正式协议完成；其余形状未做 | fusion 上 band 对 box-mask −0.008（n.s.）；按形状定框后 band 对 box-mask + fit **+0.086**、对 box-mask 最好的后处理 **+0.044**（都 10/10），同参数 145/180 → Round 13、18 |
| 1 | 困难子集、按可见度分层、失败案例 | R2-1、R2-2、R2-10 | 数据集 2 分层 recall 完成；CNR / 杂波未算；失败案例图有工具（`vis_errors.py`），未选图 | baseline → band + fit：单翼 +0.142（8/2）、畸变 +0.079（6/2/2）、weak +0.069（n.s.）；grouped 对 baseline +0.011（n.s.），对 box-mask +0.105 → Round 18 |
| 1 | 数据集 1 按类别报告（管线 / 钢筋排） | R2-1、R1-2 | 未开始（等 band 标注） | |
| 1 | 完整检测器对比 + 插件实验 | R1-3、R2-3 | **插件（YOLOv8s 预训练）完成**，10 seeds × 5 folds + box-mask control；从零训练的 YOLOv8s 未跑；检测器对比矩阵未跑 | 插件无增益：0.697 → 0.711（+0.014，n.s.），band 对 box-mask +0.008；band 独有的是截断双曲线的参数（slope ×1.09 vs 框 ×1.24）→ `yolo_band_ds2/README.md`；对比 → `detector_comparison/README.md` |
| 1 | 敏感性：thickness ×0.25–2、span ×0.75 / 1.25 | R1-4、R2-4 | 未开始 | |
| 2 | 敏感性：顶点偏移、slope 扰动 | R1-4 | 未开始 | |
| 2 | 跨数据集：零样本 + 少样本 | R2-7 | 未开始 | |
| 2 | box refinement 做进推理 + scale control + upper bound | R1-1 增益来源 | ds2 完成（Round 18 取代 Round 6、12 的数字） | band + fit 对 scale control +0.093（10/10）；真实 band + fit 上限 0.662 → Round 6、12、18 |
| 2 | 上限与结构诊断：oracle fusion、extent、stride 4、λ、gate、训练集 25–100% | 解释 R1-1 | ds2 完成 | 都不改变"fusion 只有 +0.02" → Round 7–10（训练集比例由 Round 19 取代） |
| 2 | stage 2 框回归（按每条双曲线，只用框标注）+ band / boxmask / oracle arm | 对照（11-21 决定不在主线上） | ds2 完成 | +0.159（10/10）；band 在 stage 2 上无额外作用。它不用 band，证明的是"再回归一次框有用"，不作为 band 方法要超过的对照 → Round 14 |
| 2 | **hyperbola fit** 代替 Voronoi split；**U-shaped decoder** attention head | grouped targets；band 质量 | ds2 完成 | fit 是 Round 18 主结论的后处理；预测 band 测试 IoU 只到 0.42（受数据量限制）→ Round 15、16、18 |
| 2 | 参数化检测头（网络直接输出双曲线参数） | band 标注的其他用法 | ds2 完成 | 参数化 +0.08，但只用框反推参数的 pseudo 拿到几乎全部；顺带发现翻转增强 bug → Round 17 |
| 2 | 形状外推（补翼尖和漏画的 band） | grouped targets | 未开始（下一步） | 目标：缩小预测 band + fit（0.585）和真实 band + fit（0.662）的差距 |
| 3 | 同等标注预算比较 | R2-4 | 未开始 | |

## 3. 当前结论与下一步

只维护一处：[basic_ds2/README.md](basic_ds2/README.md) 开头的"答案"和"下一步"（逐轮过程在 [basic_ds2/rounds.md](basic_ds2/rounds.md)）。本页不复制数字。

## 4. 新开一个实验时

1. 在本目录下建 `<name>/`，README 按[根目录 README 的"写文档的约定"](../../README.md#写文档的约定)写：开头给答案，正文按 Round 时间线，每轮"为什么做 / 做了什么 / 结果 / 结论"，次要的大表放进 `<details>` 折叠，术语用英文并在开头放术语表。
2. 数据用 `detector_comparison/prepare_data.py` 导出，评估调 `detector_comparison/evaluate.py`，不要另写口径。
3. 多 seed 用 `basic_ds2/run_seeds.py` / `aggregate_cv.py` 的模式：每个 run 一个目录、`run_info.json` 记全部参数、已完成的自动跳过、按 seed 配对统计。推理步骤的参数在合并 validation set 上全局选。
4. 做完在本页第 2 节更新状态和结论位置。
