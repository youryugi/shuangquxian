# 实验 4：注意力机制对比（数据集 2）

**问题**：投稿版 Table III 说本文方法比四个通用注意力模块（SE、CBAM、Non-local、CA）都好（F1 0.89 对次优 0.82）。那张表用的是数据集 1、旧划分（有同源泄漏）、随机切分、YOLO 式头。在修改稿的严格协议下（数据集 2 新 HBAT 标注、按来源分组 5 折合并、10 seeds 配对、验证集选阈值），这个结论还成立吗？顺带回应 R1-1：收益来自 band 监督本身，还是来自多了一个 attention 分支？

**答案（2026-11-22，5 折合并 × 10 seeds，和实验 1 Round 18 配对）**

1. **只比注意力模块本身，投稿版的结论不成立。** 四个通用模块都比 baseline 好（F1 +0.031 到 +0.058），band 监督的 fusion 只 +0.023。band fusion 对 SE / CBAM / CA 打平（−0.008 到 −0.024，不显著），对 Non-local **输**（−0.034，0/10）。
2. **同一结构只差监督：band 监督没有带来检测提升。** attention 分支不监督（λ = 0）0.490，band 监督 0.496（+0.006，n.s.）。和实验 1 的 box-mask 对照一致：fusion 这一步 band 的作用和"多一个分支"分不开。
3. **完整方法（band + fit）赢全部四个模块。** 0.585 对 0.504–0.530：+0.055 到 +0.081（9/10 到 10/10），AP50 +0.135 到 +0.152（10/10）。给四个模块也加上不需要 band 的框修正（scale control，k 在验证集上选）后仍然 +0.049 到 +0.059（全部 10/10）。
4. **fit 的收益需要 band 监督。** 把不监督的 attention 图当 band 走同一个 fit：0.466，比不监督本身还低（−0.024，n.s.），比 band + fit 低 0.119（10/10）。不监督的 attention 图和真实 band 的像素 IoU 只有 0.04，band 监督的是 0.38。
5. **分层**：CBAM / Non-local 的 weak recall 最高（0.32 / 0.29，band + fit 0.20），band + fit 的单翼 recall 最高（0.50，最好的通用模块 0.44）。grouped 上 band + fit（0.44）不如 CBAM（0.48）。

**对论文的含义**：Table III 要改写。不能再说"band 监督的 attention 比通用注意力模块好"；能说的是"band 监督给出了通用模块给不了的东西——可以用来定框的 band，band + fit 比四个模块（各自加上框修正）都高 0.05–0.06 F1、0.11–0.12 AP50"。第 2 条要如实写进 R1-1 的回复。

---

## 术语

| 术语 | 含义 |
|---|---|
| baseline | 实验 1 的三层 backbone + CenterNet 头，无注意力（`hbatbox_flipfix_none`） |
| 通用模块 | SE、CBAM、Non-local、CA（Coordinate Attention），无监督，插在 bottleneck 检测特征上 —— band 融合的同一位置 |
| unsup | 结构和 band 臂完全相同（attention 头读 f3 → sigmoid → 池化后 concat 融合），attention 损失权重 λ = 0 |
| band fusion | 实验 1 的 band 监督 attention（`hbatbox_flipfix_abs`），只看检测头的输出 |
| band + fit | 实验 1 的方法：band fusion + 沿预测的 band 拟合双曲线定框（`hbatbox_flipfix_abs_fitbox`，Round 15、18） |
| scale control | 检测框绕中心等比放大 k 倍（k ∈ 1.0–1.6，在合并验证集上选），不用 band 的框修正 |
| box-mask | attention 用填实的 HBAT 框监督（实验 1 Round 13 的对照） |
| 配对 Δ / 胜/平/负 | 同一 seed（同一初始化、同一数据顺序和翻转）下两个 arm 的 5 折合并 F1 之差；10 个 seed 的均值、bootstrap 95% CI、赢 / 平 / 输次数 |

## 设置

- **数据、划分、评估**与实验 1 Round 18 完全相同：数据集 2 的 131 张原图、HBAT 派生框（222 个测试目标）、按来源分组 5 折、训练时水平翻转（几何翻转，修过 8 px 的 bug）、120 轮取最后一轮、阈值在 5 折合并验证集上选、IoU 0.5。
- **模块**：[../shared/attn_blocks.py](../shared/attn_blocks.py)，从投稿版 `experiments/submission/primary_yolohead/generic_attention_baselines.py` 原样搬来，超参数不变（SE / CBAM r = 16、CBAM 空间核 7、CA r = 32、Non-local 2×2 子采样且输出卷积零初始化）。插入位置也和投稿版一样：bottleneck 之后、检测头之前（28×28，256 通道）。
- **配对初始化**：backbone 和三个检测头用 seed 初始化，和 baseline 逐参数相同（init 指纹相同）；模块参数用 seed + 10000 单独初始化，和 band 臂的 attention 分支同一规则。`--workers 1`，数据顺序和翻转与实验 1 相同。
- **回归检查**：改过的 `train_basic.py` 重跑 baseline（f0 seed0）的 `preds_test.json` 与原 run md5 相同；重推理与训练时导出的预测最大差 0。
- **参数量**：baseline 1.911 M；SE / CBAM +0.009 M，CA +0.007 M，Non-local +0.132 M，band 臂（和 unsup）+0.140 M。
- **后处理**（`infer_bandbox.py build`，参数在合并验证集上全局选，测试集不参与）：
  - 四个模块 + scale control（选中 k：SE 1.1、CBAM 1.05、Non-local 1.1、CA 1.05；baseline 1.1）
  - unsup + fit、unsup + Voronoi + NMS：参数网格和 band + fit 完全相同（180 组），选中 (t 0.7, s 0.5, union, NMS 0.5) 和 (0.7, 0.1, union, 0.5)
- 共 250 个新 run（5 arm × 5 折 × 10 seeds），IWAC1 一张 RTX 5090，10 个并行，31 分钟。

## 结果

**结果 1：各 arm 的检测结果**（5 折合并，10 seeds 均值 ± std）

| 方法 | 参数量 (M) | Precision | Recall | F1 | AP50 |
|---|---|---|---|---|---|
| baseline | 1.911 | 0.511 ± 0.066 | 0.451 ± 0.064 | 0.472 ± 0.030 | 0.371 ± 0.032 |
| SE | 1.920 | 0.555 ± 0.058 | 0.467 ± 0.056 | 0.504 ± 0.033 | 0.389 ± 0.045 |
| CBAM | 1.920 | 0.523 ± 0.060 | 0.524 ± 0.048 | 0.520 ± 0.032 | 0.396 ± 0.030 |
| Non-local | 2.043 | 0.550 ± 0.037 | 0.514 ± 0.045 | 0.530 ± 0.027 | 0.406 ± 0.027 |
| CA | 1.918 | 0.590 ± 0.063 | 0.467 ± 0.058 | 0.517 ± 0.039 | 0.400 ± 0.039 |
| attention 分支，不监督 | 2.051 | 0.498 ± 0.062 | 0.492 ± 0.051 | 0.490 ± 0.021 | 0.393 ± 0.028 |
| box-mask 监督 | 2.051 | 0.529 ± 0.036 | 0.482 ± 0.036 | 0.504 ± 0.029 | 0.395 ± 0.031 |
| band 监督（fusion） | 2.051 | 0.501 ± 0.042 | 0.496 ± 0.044 | 0.496 ± 0.021 | 0.388 ± 0.030 |
| *加推理后处理* | | | | | |
| baseline + scale | 1.911 | 0.519 ± 0.052 | 0.476 ± 0.061 | 0.491 ± 0.019 | 0.406 ± 0.025 |
| SE + scale | 1.920 | 0.598 ± 0.067 | 0.475 ± 0.039 | 0.526 ± 0.020 | 0.433 ± 0.034 |
| CBAM + scale | 1.920 | 0.534 ± 0.051 | 0.538 ± 0.037 | 0.533 ± 0.024 | 0.422 ± 0.028 |
| Non-local + scale | 2.043 | 0.564 ± 0.045 | 0.514 ± 0.045 | 0.536 ± 0.029 | 0.425 ± 0.037 |
| CA + scale | 1.918 | 0.602 ± 0.069 | 0.477 ± 0.053 | 0.528 ± 0.032 | 0.422 ± 0.039 |
| 不监督 + fit | 2.051 | 0.531 ± 0.121 | 0.424 ± 0.058 | 0.466 ± 0.069 | 0.366 ± 0.087 |
| box-mask + Voronoi + NMS | 2.051 | 0.564 ± 0.050 | 0.524 ± 0.044 | 0.541 ± 0.028 | 0.473 ± 0.032 |
| **band + fit（本文）** | 2.051 | **0.641 ± 0.048** | **0.541 ± 0.039** | **0.585 ± 0.019** | **0.541 ± 0.031** |

**结果 2：配对 ΔF1**（B − A，95% CI，胜/平/负；ΔAP50 见 [results/summary.md](results/summary.md)）

| A → B | ΔF1 |
|---|---|
| baseline → SE / CBAM / Non-local / CA | +0.031（7/3）/ +0.048（8/2）/ +0.058（10/0）/ +0.045（8/2），全部显著 |
| baseline → 不监督 | +0.018 [−0.008, +0.043]，n.s. |
| baseline → band fusion | +0.023 [+0.006, +0.040]，8/2 |
| SE / CBAM / CA → band fusion | −0.008 / −0.024 / −0.022，均 n.s. |
| Non-local → band fusion | **−0.034 [−0.051, −0.018]，0/10** |
| 不监督 → band fusion（只差监督） | +0.006 [−0.015, +0.026]，n.s. |
| SE / CBAM / Non-local / CA → band + fit | +0.081 / +0.065 / +0.055 / +0.067（10/0、9/1、10/0、10/0） |
| 各模块 + scale → band + fit | +0.059 / +0.051 / +0.049 / +0.057，全部 10/0 |
| 不监督 + fit → band + fit | **+0.119 [+0.084, +0.159]，10/0** |

**结果 3：分层 recall**（验证集选的阈值，10 seeds 均值）

| 方法 | 全部 | weak (29) | 单翼 (50) | 畸变 (39) | grouped (93) |
|---|---|---|---|---|---|
| baseline | 0.451 | 0.131 | 0.358 | 0.362 | 0.428 |
| SE | 0.467 | 0.210 | 0.394 | 0.374 | 0.441 |
| CBAM | 0.524 | **0.321** | 0.436 | **0.490** | **0.483** |
| Non-local | 0.514 | 0.286 | 0.406 | 0.451 | 0.474 |
| CA | 0.467 | 0.176 | 0.412 | 0.382 | 0.446 |
| 不监督 | 0.492 | 0.190 | 0.444 | 0.372 | 0.478 |
| band fusion | 0.496 | 0.197 | 0.422 | 0.408 | 0.482 |
| band + fit | **0.541** | 0.200 | **0.500** | 0.441 | 0.439 |

**结果 4：attention 图与真实 band 的像素 IoU**（测试集，56×56，t = 0.5，50 个 run 均值）：不监督 0.042（band 内外均值差 −0.001），box-mask 0.455，band 0.383。不监督的 attention 分支学出来的东西和双曲线没有空间对应。

## 解读

- **在这个从零训练的小网络上，"多一点容量 / 全局上下文"本身就有用。** 四个模块都提升了 recall 或 precision；Non-local（全局 self-attention，参数最多）最好。band 监督的 attention 在检测输出上没有比它们多给什么——和实验 1 的结论一致：fusion 这一步 band 和 box-mask、不监督都分不开。
- **band 的价值在输出端，不在特征端。** 通用模块只能重加权特征；band 监督额外给出一张和双曲线对齐的图，fit 用它重新定框。这一步是通用模块拿不到的（它们没有 band 输出），而且必须是监督出来的 band：不监督的图走 fit 反而变差。
- **局限**：band + fit 在 weak 和 grouped 上不如 CBAM / Non-local。两者作用的位置不同（特征 vs 定框），可能可以叠加，见下一步。

## 诊断：band + fit 为什么在 weak 和 grouped 上弱（2026-11-23）

[diag_weak_grouped.py](diag_weak_grouped.py) → [results/weak_grouped.md](results/weak_grouped.md)。每个 GT 在验证集选的阈值下归为一类：hit；taken（有 IoU ≥ 0.5 的框，但匹配给了相邻目标）；near（框 IoU 0.3–0.5）；low_score（有 IoU ≥ 0.5 的框但分数低于阈值）；none（全部输出里都没有）。

**weak（29 个）：是分数问题，band 在 weak 目标上基本没画出来。**

| | hit | near | low_score | none |
|---|---|---|---|---|
| baseline | 0.131 | 0.048 | 0.507 | 0.314 |
| CBAM | **0.321** | 0.103 | **0.266** | 0.307 |
| Non-local | 0.286 | 0.107 | 0.321 | 0.286 |
| band fusion | 0.197 | 0.069 | 0.466 | 0.269 |
| band + fit | 0.200 | 0.072 | 0.445 | 0.283 |

- weak 漏检的一半是 low_score：检测器在那里出了框，分数不够。CBAM / Non-local 的提升几乎全部来自把 low_score 变成 hit（0.51 → 0.27 / 0.32）；none 各 arm 都在 0.27–0.31，谁也没多找到候选。
- band 臂在 weak 目标上的预测 band 覆盖率只有 0.11（clear 0.58），30% 的 weak band 整条没画出来（clear 2%）。band 监督主要从清晰目标学到，weak 处没有信号，fusion 也就提不了分。
- fit 只能改已有检测的框，near 只占 7%，所以 fit 对 weak 没有作用。
- 只有 29 个目标，CBAM 和 band + fit 的差别约每个 seed 3–4 个目标。

**grouped（93 个）：检测阶段不弱，fit 之后丢的。**

| | hit | taken | near | low_score | none |
|---|---|---|---|---|---|
| CBAM | 0.483 | 0.034 | 0.194 | 0.104 | 0.185 |
| band fusion | 0.482 | 0.028 | 0.213 | 0.113 | 0.165 |
| band + fit | **0.439** | **0.070** | 0.208 | 0.098 | 0.186 |

- band fusion 在 grouped 上和 CBAM 一样（0.482 / 0.483）。从 fusion 到 band + fit，hit 少了 0.043（每个 seed 4.0 个目标），taken 正好多了 0.042（3.9 个）：fit 之后相邻两条双曲线的框变得几乎一样，NMS 只留下一个，它只能匹配给其中一个目标，另一个就丢了（和 Round 20 "fit 后去重丢 grouped" 一致）。
- 不是 band 画不出：grouped 和 alone 的覆盖率一样（0.457 / 0.469），整条没画出来的比例也一样（7.2% / 7.6%）。
- 同样的 fit 在 alone 上是净收益（hit 0.507 → 0.616，near 0.151 → 0.075）。

**顺带核对 Round 22 留下的问题**：70 对框重叠的目标里，整条没画出来的平均每个 seed 9.2 条：medium 6.1、clear 1.7、weak 1.4。按比例算，medium / weak 各 17%，clear 3%。漏画主要是因为那一条本身不清楚，而不是分不开两条相邻的 band。实例感知的 band 监督（Round 20 想法 4）针对的不是主要原因。

## 下一步

1. **通用模块 + band + fit 叠加**（建议做）：CBAM 或 Non-local 和 band 监督 attention 同时用，再走 fit。如果能叠加，论文的说法变成"band 监督和通用注意力互补"，比单纯对比更强，也正好补 band + fit 在 weak / grouped 上的短板。需要让 `train_basic.py` 允许 `--attn-block` 和 `--mode abs` 同时用（现在 assert 禁止），2 个 arm × 50 run 约 15 分钟 + 后处理。
2. 注意力图可视化（替换投稿版 Fig. 4 / 5 的对比）：CBAM 空间图、CA 的 H/W 权重、band attention 放在同一组测试图上。
3. 数据集 1 等 band 标注完成后重跑同一套。

## 文件与复现

| 文件 | 作用 |
|---|---|
| [../shared/attn_blocks.py](../shared/attn_blocks.py) | 四个通用模块 |
| `../exp1_basic/train_basic.py --attn-block {se,cbam,nonlocal,coord}` | 训练（只和 `--mode none` 组合）；unsup = `--mode abs --lam-att 0` |
| `../exp1_basic/run_seeds.py` 的 `hbatbox_flipfix_{se,cbam,nonlocal,coord,unsup}` | 5 折 × 10 seeds 队列 |
| `../exp1_basic/infer_bandbox.py` 的 `*_scale`、`hbatbox_flipfix_unsup_{fitbox,bandbox_nms}` | 后处理 |
| [attn_compare.py](attn_compare.py) → [results/summary.md](results/summary.md)、`summary.json` | 汇总表 |
| [diag_weak_grouped.py](diag_weak_grouped.py) → [results/weak_grouped.md](results/weak_grouped.md) | weak / grouped 漏检拆解、band 覆盖率 |
| `../exp1_basic/runs_cv/ds2/hbatbox_flipfix_{se,cbam,nonlocal,coord,unsup}*` | run 目录（和实验 1、2 的 run 放在一起以便配对） |
| `../exp1_basic/runs_cv/exp4_queue_20261122.log`、`runs_cv/bandbox/exp4_post_20261122.log` | 训练队列和后处理日志 |

```bash
cd experiments/revision/exp1_basic
python run_seeds.py --exps hbatbox_flipfix_se hbatbox_flipfix_cbam hbatbox_flipfix_nonlocal hbatbox_flipfix_coord hbatbox_flipfix_unsup \
    --workers 1 --jobs 10            # 必须 --workers 1，和实验 1 配对
python infer_bandbox.py infer --exps hbatbox_flipfix_se hbatbox_flipfix_cbam hbatbox_flipfix_nonlocal hbatbox_flipfix_coord hbatbox_flipfix_unsup
python infer_bandbox.py prefit --variants hbatbox_flipfix_unsup_fitbox
python infer_bandbox.py build --variants hbatbox_flipfix_{se,cbam,nonlocal,coord}_scale hbatbox_flipfix_unsup_fitbox hbatbox_flipfix_unsup_bandbox_nms
python aggregate_cv.py
cd ../exp4_attention && python attn_compare.py
```
