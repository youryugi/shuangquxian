# 大修阶段实验（2026-11 起）

计划来源：[paper/revision/revision_map.md](../../paper/revision/revision_map.md)（审稿意见逐条分析 → 第 3 步实验表）。本页只做导航和状态，数字和分析在各实验自己的 README 里。

## 1. 目录

| 目录 | 作用 | 文档 | 代码 | 结果 |
|---|---|---|---|---|
| `detector_comparison/` | **共享基础设施** + 实验 3。数据导出（划分 → YOLO 目录 + official box + HBAT band mask）、统一评估口径、完整检测器对比（YOLOv8 / RT-DETR / Faster R-CNN）、YOLOv8 band 辅助监督插件 | [README](detector_comparison/README.md) | `prepare_data.py`、`evaluate.py`、`common.py`、`band_yolo.py`、`train_*.py`、`run_matrix.py`、`model_stats.py` | `datasets/`（不入库）、`runs_smoke/`；正式矩阵未跑 |
| `yolo_band_ds2/` | 第二个实验：band 监督插进 COCO 预训练的 YOLOv8s（approach A），同样的严格协议；box-mask control、A/A control、band IoU | [README](yolo_band_ds2/README.md) | `run_cv.py`（训练用 `detector_comparison/train_ultralytics.py`、`band_yolo.py`） | `runs_cv/` |
| `basic_ds2/` | 基础实验重跑（数据集 2，新标注，严格协议）：attention 有没有用 → 为什么没有 → band 用在 box refinement → band 形状有没有不可替代的价值 | [README](basic_ds2/README.md)（Round 1–16，按时间线；开头有答案和"结论怎么变的"） | `train_basic.py`、`run_seeds.py`、`aggregate_cv.py`、`infer_bandbox.py`、`stage2_instance.py`、`fit_hyperbola.py`、`diag_*.py` | `runs_cv/`（正式）、`runs/`（旧协议，仅对照） |

所有 revision 实验都用 `detector_comparison/` 的数据目录和 `evaluate.py`：IoU 0.5 greedy matching、validation set 选 threshold、test F1 为主指标、另报 AP50；正式结果一律 5-fold 合并 + 10 seeds 配对 + bootstrap CI + Wilcoxon。

## 2. 实验总表（对应 revision_map 第 3 步）

| 优先级 | 实验 | 回应 | 状态 | 结论 / 位置（basic_ds2 README） |
|---|---|---|---|---|
| 1 | 主表重跑：有 / 无 attention，两个数据集 | R2-5、R2-6 | **数据集 2 完成（两次硬件复现）；数据集 1 未跑（等 band 标注）** | fusion ΔF1 = +0.004（5060 Ti）/ +0.019（5090），和换卡噪声同量级 → Round 4 |
| 1 | 几何先验消融：矩形框 / 椭圆 / 平移带 / 抛物线带 / 只加辅助头 | R1-1 | 矩形框（box-mask control）按正式协议完成；其余形状未做 | box-mask fusion +0.029 ≥ band fusion +0.019；band 形状特有 +0.025 [+0.002, +0.046] → Round 13；stage 2 上 band vs boxmask +0.012（CI 含 0）→ Round 14 |
| 1 | 困难子集、按可见度分层、失败案例 | R2-1、R2-2、R2-10 | 分层 recall 完成；失败案例图有工具（`vis_errors.py`），未选图 | 收益在清晰、宽、孤立目标；grouped targets 是所有方法共同的瓶颈 → Round 11、12、14、15 |
| 1 | 数据集 1 按类别报告（管线 / 钢筋排） | R2-1、R1-2 | 未开始（等 band 标注） | |
| 1 | 完整检测器对比 + 插件实验 | R1-3、R2-3 | 插件：代码审查 + 冒烟完成，ds2 先导（3 seeds × 5 folds）换电脑重跑中（见 yolo_band_ds2 README "交接"）；检测器对比矩阵未跑 | 插件 → `yolo_band_ds2/README.md`；对比 → `detector_comparison/README.md` |
| 1 | 敏感性：thickness ×0.25–2、span ×0.75 / 1.25 | R1-4、R2-4 | 未开始 | |
| 2 | 敏感性：顶点偏移、slope 扰动 | R1-4 | 未开始 | |
| 2 | 跨数据集：零样本 + 少样本 | R2-7 | 未开始 | |
| 2 | box refinement 做进推理 + scale control + upper bound | R1-1 增益来源 | ds2 完成 | 0.473 → 0.545（+0.072，10/10）；scale control +0.029；真实 band 0.589 / 0.602 → Round 6、12 |
| 2 | 上限与结构诊断：oracle fusion、extent、stride 4、λ、gate、训练集 25–100% | 解释 R1-1 | ds2 完成 | 都不改变"fusion 只有 +0.02"；数据越少 Δ 越小 → Round 7–10 |
| 2 | **stage 2 框回归**（按每条双曲线，只用框标注）+ band / boxmask / oracle arm | 最强的对照 | ds2 完成 | **+0.159（10/10），超过所有 band 方案**；band 在 stage 2 上无额外作用 → Round 14 |
| 2 | **hyperbola fit** 代替 Voronoi split；**U-shaped decoder** attention head | grouped targets；band 质量 | ds2 完成 | 真实 band 时 grouped recall 比框回归 +0.10；预测 band 测试 IoU 只到 0.42（受数据量限制）→ Round 15、16 |
| 3 | 同等标注预算比较 | R2-4 | 未开始 | |

## 3. 当前结论与下一步

详见 [basic_ds2/README.md](basic_ds2/README.md) 开头和"结论是怎么一步步变的"。

1. 数据集 2 上 attention 走 fusion 的增益是 +0.004 / +0.019，和换显卡同量级；投稿版的 +0.06 来自 data leakage 和没对齐的初始化。
2. band 用在 box refinement 上 +0.072（10/10），其中 band 形状特有 +0.025；其余是任何 dense 监督都有的。
3. **不用 band 的 stage 2 框回归 +0.159，超过所有用 band 的方案。** band 形状唯一不可替代处：band 准确时 grouped targets 的 recall 比框回归高 0.10；但预测 band 的测试 IoU 停在 0.38–0.48，受数据量限制。
4. **需要你决定**：论文怎么面对 stage 2 这个对照（revision_map S5 的定位是 Round 14 之前定的）。
5. **可以直接做**：数据集 1 严格协议重跑（等 band 标注）；给 band 更多数据（合并数据集、合成双曲线预训练）；hyperbola fit 接在 stage 2 之后。

## 4. 新开一个实验时

1. 在本目录下建 `<name>/`，README 按 [../README.md](../README.md) 的约定写：开头给答案，正文按 Round 时间线，每轮"为什么做 / 做了什么 / 结果 / 结论"，次要的大表放进 `<details>` 折叠，术语用英文并在开头放术语表。
2. 数据用 `detector_comparison/prepare_data.py` 导出，评估调 `detector_comparison/evaluate.py`，不要另写口径。
3. 多 seed 用 `basic_ds2/run_seeds.py` / `aggregate_cv.py` 的模式：每个 run 一个目录、`run_info.json` 记全部参数、已完成的自动跳过、按 seed 配对统计。推理步骤的参数在合并 validation set 上全局选。
4. 做完在本页第 2 节更新状态和结论位置。
