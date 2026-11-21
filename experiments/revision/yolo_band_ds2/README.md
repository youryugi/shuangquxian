# YOLOv8 + band 辅助监督（数据集 2）

**要回答的问题**：band 监督插进一个成熟检测器（COCO 预训练的 YOLOv8s）还有没有用？

> **更新（2026-11-22）：下面的"没有用"只对"COCO 预训练 + 全套增广"的 YOLOv8s 成立。**
>
> 检测器对比实验里加跑了"从零训练、只左右翻转"的 YOLOv8s，条件和轻量模型对齐。在这个条件下 band 辅助监督**显著有效**：
> - F1 0.452 → 0.485，+0.034 [+0.013, +0.056]，8 胜 2 负，Wilcoxon p = 0.020；
> - AP50 +0.030 [+0.011, +0.047]。
>
> 幅度和轻量模型上的 +0.023 [+0.006, +0.040] 相当（flipfix 重跑；最后一行"basic_ds2 无增益"是 flipfix 之前的说法）。
>
> 现在还分不清是预训练还是全套增广让 band 失效。另外，从零 YOLOv8s 的 box-mask 对照还没跑。
>
> 数字和 run 名见 [detector_comparison/README.md](../detector_comparison/README.md) 0.7 节。这些 run 也放在本目录的 `runs_cv/ds2/` 下：`y8s_*_flip_*`、`frcnn_pre`。

## 简短回答（2026-11-21，10 seeds × 5 folds，按 seed 配对）

**没有用。** 加 band 辅助监督（approach A）以后，YOLOv8s 的检测没有变好：

| arm | F1@IoU0.5（5 折合并） | AP50 | F1@IoU0.7 |
|---|---|---|---|
| baseline（`y8s_pre_none`） | 0.697 ± 0.019 | 0.726 | 0.496 |
| + band（`y8s_pre_band`） | 0.711 ± 0.021 | 0.733 | 0.481 |
| + box mask 对照（`y8s_pre_boxmask`） | 0.703 ± 0.022 | 0.733 | 0.501 |

- baseline → band：ΔF1 +0.014 [−0.008, +0.034]，8/0/2，p = 0.16；ΔAP50 +0.007 [−0.009, +0.024]。
- box mask → band（band **形状**特有的作用）：ΔF1 +0.008 [−0.016, +0.031]，6/0/4；ΔAP50 +0.000。
- band arm 的 recall 高 0.054（9/10 seeds），但不是 band 形状的作用：约 0.03 来自验证集上选出的阈值更低（0.41，baseline 0.48，box mask 0.50），FP 同步增加；剩下约 0.02 在同一阈值下也有，box mask 也有同样多。把所有预测都算上的 recall（0.905 vs 0.907）三个 arm 一样。
- **换阈值不改变结论**：score 阈值从 0.05 到 0.8、六种阈值选法（包括不合规的"在测试集上挑最好的"）、IoU 阈值 0.3 到 0.6，band − box mask 都在 ±0.011 以内、都不显著；IoU 0.7 时 band 反而差 0.020。band − baseline 在个别阈值上 CI 不含 0（+0.016 到 +0.024），但 box mask 在同样的阈值上也一样高，而且是在 12 个阈值里挑出来的，不能当证据。
- 预测 band 的质量：测试集 band IoU 0.47，和 basic_ds2 的 CenterNet（0.38–0.46）差不多，**预训练 backbone 没有突破数据量的限制**。在框内，band 图区分 TP 和 FP 的能力是随机水平（AUC 0.50），所以也不能用来给检测重新打分。
和 basic_ds2 的结论一致：band 作为辅助监督或特征融合，在数据集 2 上没有增益。

**band 在哪里有用**（Round 3，不重训的诊断，10 seeds）：

- **检测 F1 上没有空间。** 给真实 band、用 basic_ds2 Round 15 的 hyperbola fit 修 YOLO 的框，F1 只 +0.016（10/10 seeds），即使在测试集上挑最好的参数也只 +0.035。grouped recall 反而下降（basic_ds2 里 band 唯一不可替代的证据，换到 YOLO 上不成立）。用预测的 band 修框是 −0.029。
- **原因是信息冗余。** HBAT box 就是 band 的外接框。没被图像边缘截断的目标占 70%，一个框加默认厚度就能把 slope 还原到 ±3%。所以对"框"这个检测目标，band 标注几乎没有多出来的信息。这也解释了为什么 band 和 box mask 的每一组对照都打平。
- **框给不了、band 给得了的：被图像边缘截断的双曲线（30%）的曲线参数。** 在 YOLO 自己预测的 band 上做 hyperbola fit：

  | 估计方法 | slope（= 2/v，速度）误差 | 顶点误差 |
  |---|---|---|
  | 预测 band + fit | ×1.09 | 4.5 px |
  | 框的几何 | ×1.24 | 7.2 px |
  | 借同图其他目标的 slope | ×1.18 | — |
  | box-mask 图 + 同样的 fit | ×2.15 | — |

  框的几何比直接猜数据集中位数（×1.20）还差，10 个 seed 的范围和 band + fit 不重叠。完整的双曲线上，框反而更准（×1.07 vs ×1.10）。

**当前状态**：Round 0–3 完成。后续选项见 [下一步](#下一步)。

和 [basic_ds2](../basic_ds2/README.md) 的关系：basic_ds2 用三层卷积的 CenterNet 做了 16 轮，结论是：
- fusion 没有增益；
- band 的增益主要来自"把框画对"；
- 这件事一个普通的 stage 2 框回归（不用 band）做得更好（basic_ds2 Round 14）。

弱 baseline 会放大增益，所以这里换成强 baseline 再测一次。审稿意见 R1-3 / R2-3 也要求这个实验。

## 目录

- [术语表](#术语表)
- [实验设计](#实验设计)
- [Round 0：代码审查和改动](#round-0代码审查和改动2026-11-21)
- [Round 1：先导，3 seeds](#round-1先导3-seeds2026-11-21)
- [Round 2：10 seeds 和诊断](#round-210-seeds-和诊断2026-11-21)
- [Round 3：band 在哪里有用](#round-3band-在哪里有用2026-11-21)
- [下一步](#下一步)
- [附录 A：文件](#附录-a文件)
- [附录 B：复现步骤](#附录-b复现步骤)
- [附录 C：运行记录](#附录-c运行记录)

## 术语表

| 术语 | 意思 |
|---|---|
| arm | 一组对比里的一个设置，比如 baseline arm、band arm |
| baseline（`y8s_pre_none`） | YOLOv8s，COCO 预训练，不加任何辅助监督 |
| aux head | 接在 YOLO neck P3（layer 15，stride 8）上的小头（3×3 Conv-BN-SiLU + 1×1 Conv），输出一张"band 在哪"的图 |
| approach A | 只加 aux head 的损失（BCE + Dice），输出不回传给检测分支；推理时 aux head 可以删掉，速度和参数不变 |
| band arm（`y8s_pre_band`） | aux head 用 HBAT band mask 监督 |
| box-mask control（`y8s_pre_boxmask`） | 同一个 aux head，监督换成填实的 HBAT box。band − boxmask = band 形状特有的作用（basic_ds2 Round 13 的思路） |
| A/A control（`y8s_pre_g0`） | aux head 接上但损失权重为 0。它和 baseline 的差只可能来自代码管线和随机性 |
| fingerprint | 两段 hash：训练开始时检测权重的 hash / 第一个 batch（增广后的图和框）的 hash。同一 fold、同一 seed 的各 arm 必须相同 |
| HBAT box | 每条双曲线一个框（instance-level），由 HBAT 标注生成；basic_ds2 全部用它 |
| pooled threshold | 5 个 fold 的验证集预测合在一起选一个 score threshold，再在合并的测试集（131 张图、222 个目标）上算 F1 |
| 工作点 | 检测器在 precision-recall 曲线上实际用的那个点，由 score threshold 决定。阈值低 → recall 高、FP 多 |
| maxR | 所有预测（conf ≥ 0.001）都算上时的 recall：检测器到底有没有把这个目标提出来，与阈值无关 |
| R@P0.8 | precision 不低于 0.8 时能达到的最高 recall。两个 arm 在同一 precision 下比 recall，排除工作点的影响 |
| band IoU | 预测 band（阈值 0.5）和 HBAT band 的像素 IoU，测试集像素合计（同 basic_ds2 Round 16） |
| near miss / grouped | 最佳 IoU 在 0.3–0.5 的漏检；official box 里套着 ≥2 条双曲线的目标（同 basic_ds2） |

## 实验设计

| 项 | 设置 | 来自哪条经验 |
|---|---|---|
| 数据 | 数据集 2，`ds2_f{0-4}_s100_hbatbox`，目标 = HBAT box | basic_ds2 Round 2–3：official box 是 scene-level，和方法要学的东西不一致 |
| 划分 | 5-fold，按来源分组；每张图恰好在一个 fold 的测试集里 | basic_ds2 附录 A |
| seed | 先导 3 个，正式 10 个；按 seed 配对。seed 同时决定检测头初始化、数据顺序和增广 | 投稿版只有 3 seeds、初始化没对齐，+0.06 是噪声（basic_ds2 Round 4） |
| 评估 | pooled threshold → test F1@IoU0.5；另报 AP50、F1@0.7、分层 recall。预测框先裁到图内（`evaluate.py`，11-21 起） | 同 basic_ds2，可直接用 `aggregate_cv.py`、`diag_strata.py` |
| 模型选择 | 最后一个 epoch（`last.pt`） | 验证集每折只有 13 张图；ultralytics 8.4.70 的 best.pt 只看 mAP50-95，噪声大，而且两个 arm 可能挑到不同 epoch |
| 训练 | 100 epoch，640 输入（224 原图放大），batch 16，ultralytics 默认超参和增广；`--workers 2` | 和预训练尺度一致。640 下 P3 是 80×80，相当于原图 stride 2.8，比 basic_ds2 的 56×56 还细 |
| band 损失 | BCE + Dice，λ = 1，前 3 epoch 线性热身，在 160×160 上算（soft target） | basic_ds2 Round 10：λ 没有影响，所以不扫 |
| 机器 | IWAC1 + IWAC2（各一块 RTX 5090）；同一 (fold, seed) 的 3 个 arm 在同一块卡上 | 冒烟时两台机器逐位相同（Round 1），所以按 (fold, seed) 分组分到两台机器是安全的 |

## Round 0：代码审查和改动（2026-11-21）

### 为什么做

插件代码（`../detector_comparison/band_yolo.py`）是 11-14 写的，比 basic_ds2 的大部分发现早。用这些经验检查一遍再跑。

### 发现的问题和改动

| # | 问题 | 后果 | 改动 |
|---|---|---|---|
| 1 | 目标框默认用 official box | 和 basic_ds2 结论不可比，也重复了 Round 2–3 的问题 | 只用 `_hbatbox` 数据目录 |
| 2 | `run_matrix.py` 只跑 fold 0 | 测试集 27 张图，一个 seed 的 F1 波动很大 | 新写 `run_cv.py`：5 fold × seeds，目录格式和 basic_ds2 相同 |
| 3 | 用 best.pt（按 val mAP50-95 选 epoch） | 13 张图上选 epoch 是噪声；两个 arm 选到不同 epoch 会混进 Δ | `--weights last` |
| 4 | 没有检查配对 | 两个 arm 的初始化 / 数据顺序不一样时，Δ 里混着噪声（投稿版的问题） | 每个 run 记 fingerprint；`run_cv.py --check` 对比 |
| 5 | 没有噪声底 | 不知道多大的 Δ 算真的 | A/A control（`y8s_pre_g0`） |
| 6 | 测试时 aux head 的输出直接丢掉 | 不知道 band 学得好不好。basic_ds2 Round 16 的结论是 band 质量受数据量限制，预训练 backbone 可能改善这一点，需要量 | `--save-band`：导出 val/test 的 band 图（PNG）和 band IoU |
| 7 | Windows 上 4 个 DataLoader worker 报 MemoryError | 每个 worker 都加载一遍 torch + CUDA DLL，页面文件不够 | `--workers 1`（Windows）；Linux 上用 2 |
| 8 | **ultralytics 8.4.70 的训练 DataLoader 用写死的种子**（`build.py`：`6148914691236517205 + RANK`） | 数据顺序和 mosaic / 仿射 / 翻转对所有 `--seed` 都一样，seed 只改变检测头的初始化。多个 seed 共用同一条增广序列，seed 间的差异被低估。这是看 fingerprint 时发现的：fold 0 的 seed 0、1、2 第一个 batch 的 hash 都是 `52efd014…` | `train_ultralytics.py` 在建好 DataLoader 后用 `--seed` 重设它的 generator（`--fixed-data-order` 可以恢复原行为）。同一 seed 的各 arm 仍然相同 |

没改、但要知道的：

- **approach A 只能测"band 监督有没有改善特征"**。basic_ds2 里有效的是推理时用 band 修框（Round 12），在 YOLO 上要另做（见下一步）。
- **6 个 unparam 目标**（每折的 manifest 都记着）有 HBAT box、没有 band：band arm 把它们当背景，boxmask arm 把它们当前景。影响很小，但是两个 arm 的一个不对称。对齐检查图左上角的点状目标就是这种。
- 导出预测时 NMS IoU = 0.7（ultralytics 默认）。grouped targets 互相重叠，NMS 会影响它们的 recall，三个 arm 用同一个值。
- DataLoader 的每个 worker 有自己的随机种子，**worker 数不同，增广就不同**。所有 run 都用 `--workers 2`。

### 冒烟测试（fold 0、2 epoch，RTX 5060 Ti）

| 检查 | 结果 |
|---|---|
| fingerprint：none / band / boxmask / g0 | 同一 seed 下全部相同；修复第 8 条后，seed 0 和 seed 1 的第一个 batch 不同（`c2450bfd…` vs `6b25adb7…`） |
| A/A：g0 vs none | **逐位相同**（seed 0：test F1 0.4211 = 0.4211；seed 1：0.2410 = 0.2410） |
| band arm 自己重跑一次 | `preds_test.json` **逐位相同**。band 损失里的 bilinear upsample 没有带来非确定性 |
| 增广后 band 和图像对齐（`train_batch0_band.jpg`） | mosaic、缩放、翻转之后 band 仍贴在回波上，也在框内 |
| band 导出 | `band_val/`、`band_test/` PNG 正常；2 epoch 时 band IoU 0.06（还没学） |
| run_info | `box_source = hbat`、`fold`、`model_selection = last epoch`、`data_order = seeded by --seed`、`fingerprint`、`band_quality` 都写进去了 |
| 统计脚本 | `../basic_ds2/aggregate_cv.py --runs ...` 能直接读这里的目录 |

### 结论

代码可以跑正式实验。同一台机器上，同一 fold、同一 seed 的各 arm 起点完全一样，训练也是确定性的，所以 Δ 只来自监督本身。

## Round 1：先导，3 seeds（2026-11-21）

### 为什么做

先看方向和量级，再决定要不要花 10 seeds 的时间。

### 做了什么

1. **两台 Linux 机器上重新冒烟**（fold 0、seeds 0–1、2 epoch、none / band / g0，`--jobs 3 --workers 2`），IWAC1 和 IWAC2 同时跑：

   | 检查 | 结果 |
   |---|---|
   | 两台机器之间 | 6 个 run 的 `preds_val.json`、`preds_test.json` **逐位相同** |
   | A/A：g0 vs none | 逐位相同（3 个 run 并行跑在同一块卡上也一样） |
   | 配对 | 同一 seed 下三个 arm 的 fingerprint 相同；seed 0 / 1 的第一个 batch 不同（`ed62c5a0…` / `95dbf060…`） |
   | 对齐图 | band 贴在回波上、在框内 |

   两台机器逐位相同，所以按 (fold, seed) 分组把 run 分到两台机器上，不会给 Δ 加噪声。Linux 和 Windows 的第一个 batch hash 不同（seed 0：`ed62c5a0…` vs `c2450bfd…`），所以 Windows 上的 13 个 run 不能和这里的混用。
2. **正式先导**：3 arm × 5 fold × seeds 0–2 = 45 个 run。15 个 (fold, seed) 组，IWAC1 跑 8 组（folds 0–1 全部、fold 2 seeds 0–1），IWAC2 跑 7 组（folds 3–4 全部、fold 2 seed 2）；同一组的 3 个 arm 在同一块卡上。每台机器 3 个 run 并行，每个 run 约 95 秒，共约 13 分钟。

### 结果

`run_cv.py --check`：15 组，0 组 fingerprint 不一致。

| arm | F1（5 折合并，3 seeds） | AP50 |
|---|---|---|
| none | 0.709 ± 0.024 | 0.731 |
| band | 0.698 ± 0.019 | 0.722 |
| boxmask | 0.707 ± 0.029 | 0.737 |

| A → B | ΔF1 [95% CI] | 胜/平/负 |
|---|---|---|
| none → band | −0.010 [−0.059, +0.019] | 2/0/1 |
| none → boxmask | −0.002 [−0.018, +0.011] | 2/0/1 |
| boxmask → band | −0.008 [−0.059, +0.027] | 2/0/1 |

（修复数据顺序前 Windows 上 fold 0 的 13 个 run：none 0.794、band 0.792、boxmask 0.775，单折，只作参考。）

### 结论

3 seeds 看不出方向，CI 宽 ±0.04。两台 5090 上扩到 10 seeds 只要约 30 分钟，所以直接扩（Round 2）。

## Round 2：10 seeds 和诊断（2026-11-21）

### 为什么做

1. Round 1 的 CI 太宽；严格协议要求 10 seeds。
2. 旧"下一步"的第 1 条：YOLO baseline 的瓶颈在哪。basic_ds2 里 band 的增益来自 near miss，如果 YOLO 的 near miss 很少，修框（approach C）就没有空间。

### 做了什么

- seeds 3–9，3 arm × 5 fold = 105 个 run。IWAC1 跑 seeds 3–5 全部和 seed 6 的 folds 0–2，IWAC2 跑 seeds 7–9 全部和 seed 6 的 folds 3–4。约 30 分钟。
- `aggregate_cv.py`（主结果）、`diag_strata.py`（分层 recall、定位质量）。
- 新写 `diag_operating_point.py`：
  - 工作点：maxR、R@P0.7/0.8/0.9、验证集选出的阈值；
  - 框内 band 图：每个预测框里 band 图的均值，能不能区分 TP 和 FP（AUC）。

### 结果

`run_cv.py --check`：50 组，0 组 fingerprint 不一致。150 个 run 全部 rc = 0。

**主结果**（`runs_cv/cv_summary.md`）：

| arm | F1@0.5 | AP50 | 测试集最优阈值 F1 | F1@0.7 |
|---|---|---|---|---|
| none | 0.697 ± 0.019 | 0.726 | 0.711 | 0.496 |
| band | 0.711 ± 0.021 | 0.733 | 0.722 | 0.481 |
| boxmask | 0.703 ± 0.022 | 0.733 | 0.718 | 0.501 |

| A → B | ΔF1 [95% CI] | p | 胜/平/负 | ΔAP50 [95% CI] | ΔF1@0.7 [95% CI] |
|---|---|---|---|---|---|
| none → band | +0.014 [−0.008, +0.034] | 0.16 | 8/0/2 | +0.007 [−0.009, +0.024] | −0.015 [−0.033, +0.002] |
| none → boxmask | +0.006 [−0.006, +0.018] | 0.19 | 8/0/2 | +0.007 [+0.000, +0.014] | +0.005 [−0.013, +0.024] |
| boxmask → band | +0.008 [−0.016, +0.031] | 0.43 | 6/0/4 | +0.000 [−0.020, +0.017] | −0.020 [−0.043, +0.004] |

Round 1 的 3 个 seed 恰好是 band arm 较差的几个：3 seeds 时 Δ 是 −0.010，10 seeds 时是 +0.014，**3 seeds 连符号都定不下来**。

**band arm 多检出了目标，FP 也一起多了**（`runs_cv/strata/summary.md`）：

| 指标 | none | band | boxmask | none → band | boxmask → band |
|---|---|---|---|---|---|
| recall@0.5 | 0.623 | 0.677 | 0.636 | +0.054 [+0.028, +0.081]，9/0/1 | +0.041 [+0.000, +0.074]，8/0/2 |
| 完全没检到（最佳 IoU < 0.3） | 0.31 | 0.25 | 0.29 | −0.058，9 个 seed 更少 | −0.042，9 个 seed 更少 |
| near miss（0.3–0.5） | 0.064 | 0.069 | 0.067 | +0.005 | +0.002 |
| FP/图：单目标图 / 多目标图 | 0.24 / 0.33 | 0.33 / 0.46 | 0.25 / 0.35 | +0.09 / +0.13，9/10、8/10 | +0.08 / +0.11 |

recall 的增加在每个分层里都差不多（visibility、宽度、单 / 多目标、grouped / alone 都是 +0.03 到 +0.08），没有集中在某一类目标上。

**工作点**（`runs_cv/diag/summary.md`）：

| 指标 | none | band | boxmask | none → band | boxmask → band |
|---|---|---|---|---|---|
| maxR（所有预测都算） | 0.907 | 0.905 | 0.901 | −0.001 [−0.010, +0.008] | +0.004 [−0.006, +0.015] |
| R@P0.7 | 0.695 | 0.705 | 0.701 | +0.011 [−0.019, +0.040] | +0.004 [−0.025, +0.032] |
| R@P0.8 | 0.620 | 0.637 | 0.636 | +0.018 [−0.018, +0.051] | +0.002 [−0.035, +0.036] |
| R@P0.9 | 0.458 | 0.513 | 0.495 | +0.055 [−0.030, +0.110]，9/0/1 | +0.018 [−0.044, +0.072] |
| 验证集选出的阈值 | 0.483 | 0.411 | 0.496 | −0.072 [−0.133, −0.011] | −0.085 [−0.148, −0.007]，9 个 seed 更低 |

**换阈值会怎样**（`runs_cv/diag/summary.md` 第 3 部分；11-21 用户问"改动阈值会有什么影响"后补做）：

| 阈值 | 怎么换 | band − baseline | band − box mask |
|---|---|---|---|
| score 阈值，三个 arm 用同一个固定值 | 0.05 … 0.8，12 个值 | +0.002 到 +0.024；0.05、0.10、0.60–0.70 处 CI 不含 0 | −0.003 到 +0.011，全部不显著 |
| score 阈值的选法 | 验证集各自选（协议）/ 都用 baseline 的 / 固定 0.25 / 固定 0.5 / 测试集最优（不合规）/ AP50 | +0.005 到 +0.015，全部不显著 | 0.000 到 +0.008，全部不显著 |
| 判定命中的 IoU | 0.3 / 0.4 / 0.5 / 0.6 / 0.7 | +0.006 到 +0.014；**0.7 时 −0.020**（1/0/9） | 0.3 时 −0.009（2/0/8）；**0.7 时 −0.020** |
| band 图二值化阈值（截断目标的 slope 误差，3 seeds） | t = 0.3 … 0.7 | — | 都是 ×1.09（框的几何 ×1.20） |

同一阈值下，band arm 的 recall 比 baseline 高约 0.02、precision 一样（如 0.5：recall 0.618 → 0.639，precision 0.802 → 0.802），box mask 也一样（0.711 = 0.711）。所以：
- 验证集上的 recall 增益 +0.054 ≈ 同一阈值下的 +0.02（任何掩码监督都有，不是 band 形状的）+ 阈值更低带来的 +0.03；
- band − baseline 在 12 个固定阈值里有 5 个 CI 不含 0，但这是多次比较里挑出来的，而且 box mask 在同样的阈值上也一样高；
- IoU 要求越严，band 越不利：band arm 的框没有更贴，IoU 0.7 时显著更差。

**band 质量**（`run_info.json` 的 `band_quality`，5 折像素合计）：

| arm | 测试集 band IoU（10 seeds 均值 [最小, 最大]） | precision | recall |
|---|---|---|---|
| band | 0.470 [0.464, 0.474] | 0.62 | 0.66 |
| boxmask（和 HBAT band 比） | 0.220 [0.214, 0.228] | 0.23 | 0.85 |

basic_ds2 的 CenterNet：attention 图 0.38–0.46，Round 16 全分辨率 band head 0.42（训练集 0.76）。

**框内 band 图能否区分 TP / FP**（AUC，0.5 = 没有信息；模糊区 = 0.2 ≤ score < 0.6，每个 seed 约 120 个预测，31% 是 TP）：

| arm 的图 | score 本身（全部） | 框内均值（全部） | score 本身（模糊区） | 框内均值（模糊区） |
|---|---|---|---|---|
| band | 0.885 | 0.619 | 0.651 | **0.496** [0.420, 0.563] |
| boxmask | 0.887 | 0.668 | 0.660 | 0.488 [0.412, 0.568] |

**YOLO baseline 的瓶颈**（`diag_strata.py`，最佳 IoU 命中 / near miss / 没检到）：

| 检测器 | 命中 ≥0.5 | near miss | 没检到 <0.3 |
|---|---|---|---|
| YOLOv8s 预训练（none） | 0.63 | 0.06 | 0.31 |
| basic_ds2 CenterNet baseline | 0.46 | 0.16 | 0.38 |

### 结论

1. **approach A 在 YOLOv8s 上没有增益。** F1、AP50、测试集最优阈值 F1 都没有显著变化。和 box mask 对照比，band 形状特有的部分：F1 +0.008 [−0.016, +0.031]，AP50 +0.000。
2. **band arm 多出来的 recall 不是 band 形状的作用。** 约 0.03 是工作点移动，约 0.02 是任何掩码监督都有的：
   - 所有预测都算上时，三个 arm 的 recall 一样（0.90–0.91），band 没有提出 baseline 提不出的目标；
   - 同一 precision 下，band 和 box mask 的 recall 一样（R@P0.7、R@P0.8 相差 ≤ 0.004）；同一 score 阈值下，band 和 box mask 的 F1 也一样（换阈值会怎样）；
   - band arm 在验证集上选出的阈值更低（9/10 seeds 低于 box mask），所以 recall 和 FP 一起升；
   - 增益平均分布在所有分层上，不像某种机制在起作用。

   R@P0.9 上 none → band 是 9/0/1，但 CI 跨 0，box mask 也有 +0.037，不是 band 形状特有的。
3. **换阈值不改变结论。** score 阈值怎么选、IoU 阈值取多少，band − box mask 都不显著；IoU 0.7 时 band 显著更差。
4. **预训练 backbone 没有突破 band 质量的上限。** 测试集 band IoU 0.47，CenterNet 是 0.42。和 basic_ds2 Round 16 的结论一致：限制来自数据量（训练集 0.76）。
5. **框内 band 图不包含 score 以外的信息。** 在 score 模糊的那一段，用 band 图区分 TP / FP 是随机水平（AUC 0.50，box mask 0.49）。所以"用 band 重新打分 / 验证检测"这条路，用现在的预测 band 也走不通。
6. **YOLO 的漏检不是框画错。** near miss 只占 6%，CenterNet 是 16%。basic_ds2 里 band 唯一起作用的是修框（Round 12 / 15），在 YOLO 上最多能救 0.064 × 222 ≈ 14 个目标。YOLO 的差距在于：31% 的目标在阈值上没检到（其中 9% 任何阈值都检不到），以及 TP / FP 的排序。

   （Round 3 更正：每个 near miss 同时也是一个 FP，所以修好全部 near miss 的 F1 上限按算术约 +0.07，不是"只有 14 个目标"那么小。实际测出来，真实 band + fit 只有 +0.016。）

**对论文的意义**：这组实验不支持"band 监督能改善一个现代检测器"。可以如实报告：成熟检测器（COCO 预训练 YOLOv8s）+ band 辅助监督，10 seeds × 5 folds、配对、有 box-mask 对照，F1 和 AP50 都没有显著变化。这回答了 R1-3 / R2-3，结论是否定的。

## Round 3：band 在哪里有用（2026-11-21）

### 为什么做

用户的问题：根据 Round 1–2，band 在什么条件下会有用？要回答三件事：
1. YOLO 错在哪，哪些错误 band 有可能修；
2. basic_ds2 里 band 唯一不可替代的证据（真实 band + hyperbola fit 在 grouped targets 上比框回归好），换到强检测器上还成立吗；
3. band 里有哪些信息是框没有的。

都不重训，只用 Round 2 的 150 个 run 和导出的 band 图。

### 做了什么

- `diag_fit.py`：basic_ds2 Round 15 的 hyperbola fit 规则（`../basic_ds2/fit_hyperbola.py`）和参数网格，用在 YOLO 的检测上。参数在 10 个 seed 的合并验证 F1 上全局选一组。四个变体：

  | 变体 | 检测 | band | 作用 |
  |---|---|---|---|
  | `y8s_pre_none_nms` | baseline | 不用 | 对照：只多一道 NMS |
  | `y8s_pre_none_gtfit` | baseline | 真实 band | 上限：band 完全准确时 |
  | `y8s_pre_band_fit` | band arm | 它自己预测的 band | 方法 |
  | `y8s_pre_boxmask_fit` | boxmask arm | 它预测的框掩码 | 对照 |

- `diag_band_value.py`：错误构成、框和 band 的信息冗余、曲线参数（slope、顶点）误差、速度一致性。

### 结果

**1. 错误构成**（baseline，每个 seed 的平均个数，`runs_cv/band_value/summary.md`）：

| 漏检（共 83.7 / 222） | 个数 | 占比 |
|---|---|---|
| 有 IoU ≥ 0.5 的框，但分数低于阈值 | 51.3 | 61% |
| 任何分数都没有 IoU ≥ 0.5 的框 | 17.3 | 21% |
| near miss（最佳 IoU 0.3–0.5） | 14.2 | 17% |
| 有好框，但被相邻 GT 占了 | 0.9 | 1% |

| FP（共 36.5） | 个数 | 占比 |
|---|---|---|
| 在某个 GT 旁边（IoU 0.3–0.5） | 12.6 | 35% |
| 背景 | 11.7 | 32% |
| 重复（贴着已匹配的 GT） | 5.8 | 16% |
| 和 GT 部分重叠（0.1–0.3） | 5.0 | 14% |
| 没有 HBAT GT，但在 official box 里 | 1.4 | 4% |

grouped 的漏检率 0.44，alone 0.33。

**2. 用 band 修框**（`runs_cv/cv_summary.md`、`runs_cv/strata_fit/summary.md`、参数网格 `runs_cv/fit/sweep.md`）：

| 变体 | F1@0.5 | AP50 | F1@0.7 | grouped recall |
|---|---|---|---|---|
| baseline | 0.697 | 0.726 | 0.496 | 0.563 |
| + NMS（对照） | 0.697 | 0.726 | — | — |
| + **真实** band + fit（上限） | 0.713 | 0.729 | **0.694** | 0.525 |
| band arm | 0.711 | 0.733 | 0.481 | 0.622 |
| band arm + 预测 band + fit | 0.681 | 0.694 | 0.412 | — |
| boxmask arm + 预测框掩码 + fit | 0.611 | 0.590 | — | — |

| A → B | ΔF1 [95% CI] | 胜/平/负 |
|---|---|---|
| baseline → 真实 band + fit | +0.016 [+0.010, +0.023] | 10/0/0 |
| band arm → 预测 band + fit | −0.029 [−0.043, −0.017] | 1/0/9 |
| boxmask + fit → band + fit | +0.070 [+0.047, +0.093] | 10/0/0 |

- 真实 band + fit：grouped recall 0.563 → 0.525（−0.039 [−0.072, −0.003]），alone 0.666 → 0.791。
- 在测试集上挑最好的参数（不合规，只看上限），真实 band + fit 也只到 0.732（+0.035）。
- basic_ds2 Round 15 里，真实 band + fit 让 CenterNet 的 grouped recall 提高（相对 Voronoi 切分 +0.047）；在 YOLO 上 grouped 反而变差。

**3. 信息冗余**（只给一个 GT 框、厚度取 16 px，反推 slope，和 HBAT 标注比）：

| GT 框 | 目标数 | slope 误差（中位数） | ±15% 以内 |
|---|---|---|---|
| 没碰到图像边缘 | 151（70%） | ×1.03 | 0.96 |
| 碰到图像边缘（clipped） | 65（30%） | ×1.22 | 0.38 |

同一张图里各目标的 slope（slope = 2/v，同一介质应该接近）：param_status = ok 的目标，同图两两差 ×1.11，不同图 ×1.35；slope 方差只有 14% 在图内。

**4. 曲线参数**（TP 检测上，每个 seed 的中位误差，10 seeds 均值 [最小, 最大]）：

| 目标 | 估计方法 | slope 误差 | slope ±15% 以内 | 顶点误差 | 顶点 5 px 以内 |
|---|---|---|---|---|---|
| clipped（约 50 / seed） | 数据集中位数 2.99 | ×1.20 | 0.43 | — | — |
| | 框的几何 | ×1.24 [1.16, 1.30] | 0.35 | 7.2 px [5.7, 8.0] | 0.34 |
| | 借同图未截断框的 slope | ×1.18 | 0.45 | — | — |
| | **预测 band + fit** | **×1.09 [1.08, 1.11]** | **0.65** | **4.5 px [3.7, 5.1]** | **0.57** |
| | 预测框掩码 + fit（boxmask arm） | ×2.15 | 0.05 | 28.8 px | 0.03 |
| | 真实 band + fit（上限） | ×1.05 | 0.89 | 1.9 px | 0.83 |
| unclipped（约 99 / seed） | 框的几何 | ×1.07 | 0.78 | 3.6 px | 0.68 |
| | 预测 band + fit | ×1.10 | 0.68 | 4.0 px | 0.59 |
| | 真实 band + fit（上限） | ×1.05 | 0.96 | 1.6 px | 0.88 |

**5. 速度一致性能否区分 TP / FP**（baseline 检测，模糊区 0.2 ≤ score < 0.6）：框反推的 slope 和同图高分检测一致的程度，AUC 0.549；换成同图标注的 slope（上限）也只有 0.566。score 本身是 0.642。

### 结论

1. **在检测 F1 上，band 没有可争取的空间。** 即使 band 完全准确，用它修 YOLO 的框 F1 也只 +0.016（上限 +0.035）。原因是 YOLO 61% 的漏检是"有好框但分数低"，这是排序问题，修框修不到。basic_ds2 里 grouped targets 上的优势，在 YOLO 上消失了。
2. **band 和 HBAT box 的信息大部分重复。** HBAT box 是 band 的外接框，70% 的目标可以从框还原 band（slope ±3%）。所以无论 attention、fusion、辅助监督还是修框，band 和 box mask 都打平，这是信息量决定的，不是结构没设计好。
3. **band 独有的是被截断双曲线的形状。** 碰到图像边缘的目标（30%），框推不出 slope 和顶点（比猜中位数还差），借同图目标的 slope 也不行；YOLO 自己预测的 band 做 fit，slope 误差 ×1.24 → ×1.09，顶点 7.2 → 4.5 px，10 个 seed 都如此。box-mask 图做同样的 fit 完全不行（×2.15），所以是 band 形状的作用。
4. 速度一致性（同一张图 slope 应该接近）是真的（图内方差只占 14%），但用来重新打分几乎没有信息（AUC 0.55）。
5. **band 有没有用，取决于检测器有多强。** 轻量模型（basic_ds2 的 CenterNet）上，band attention + hyperbola fit 比 box-mask + fit 高 +0.086、比 box-mask 最好的后处理高 +0.044（basic_ds2 Round 18，修好翻转后的正式结果，都是 10/10）；COCO 预训练的 YOLOv8s 上，连真实 band 都只有 +0.016。（YOLO 不受翻转 bug 影响：ultralytics 自己做翻转。）

**对论文的意义**（不改论文主张的前提下）：
- 检测 F1 的提升可以主张在论文的轻量模型上（basic_ds2 Round 18 已确认：band + fit 0.472 → 0.585；Round 19：每个训练比例都成立）；对 R1-3 / R2-3 要如实说明：换成预训练的 YOLOv8s，band 监督对 F1 没有影响（不增也不减）。
- 方法上的小改动：在 band attention 后面接 hyperbola fit，让检测器**额外输出每条双曲线的顶点和速度（slope = 2/v）**。对被图像边缘截断的双曲线，这是框给不了的（上面结论 3），可以作为 band attention 的新增价值写进论文，对两种检测器都成立。

## 下一步

按对"band 有没有用"这个问题的信息量排列。都还没跑，等用户决定。

1. **从零训练的 YOLO + fit**（`y8s_scr_none / band / boxmask`，再跑 `diag_fit.py`）：直接检验结论 5"band 的作用取决于检测器强弱"。同一个 YOLO 结构，去掉预训练就变成弱检测器；如果 band + fit 在从零训练时比 box-mask 好、在预训练时打平，论文就能说清楚 band attention 在什么条件下有效。两台 5090 上 150 个 run 约 45 分钟，fit 只用 CPU。
2. **轻量模型上的参数输出**：用 basic_ds2 flipfix 重跑的 band 图（Round 18，已完成），按 `diag_band_value.py` 第 3 部分算截断双曲线的 slope / 顶点误差，让"band 额外输出物理参数"在论文的模型上也有数字。只用 CPU。
3. **把 band 预测好**：真实 band + fit 在截断目标上是 ×1.05 / 1.9 px，预测 band 是 ×1.09 / 4.5 px，差距来自 band 质量（测试集 band IoU 0.47），受数据量限制（basic_ds2 Round 16）。数据集 1 补齐 band 标注后再看。
4. （可选，防审稿人问）不用 band mask、直接用 HBAT 参数回归顶点和 slope 的头。如果它在截断目标上也能做到 ×1.09，审稿人可能会说起作用的是参数标注而不是 band mask。这是另一种方法、不是本文的对象，做不做由用户定。
5. **检测器对比矩阵**（`../detector_comparison/`）：R1-3 也要求，和 band 无关。

不建议再做的：调 fusion / attention / 辅助监督的结构（basic_ds2 和这里共 6 种结构都打平，原因是信息冗余）；用 band 修框提高 F1（上限 +0.016）；用速度一致性重新打分（AUC 0.55）。

## 附录 A：文件

| 文件 | 作用 |
|---|---|
| `run_cv.py` | 实验矩阵（arm 定义在 `EXPS` 里）、并行、跳过已完成的 run、`--check` 配对检查 |
| `diag_operating_point.py` | Round 2 诊断：maxR、R@P、验证集阈值；框内 band 图的 TP / FP AUC。输出 `runs_cv/diag/` |
| `diag_fit.py` | Round 3：hyperbola fit 修 YOLO 的框（真实 / 预测 band、boxmask 对照、只加 NMS 对照）。写 `runs_cv/ds2/y8s_pre_{none_nms,none_gtfit,band_fit,boxmask_fit}/`，参数在 `runs_cv/fit/` |
| `diag_band_value.py` | Round 3：错误构成、框和 band 的信息冗余、曲线参数误差、速度一致性。输出 `runs_cv/band_value/`（约 5 分钟） |
| `../detector_comparison/train_ultralytics.py` | 单个 run：训练 → 导出预测 → 评估。Round 0 新增 `--run-name`、`--weights`、`--save-band`、`--fixed-data-order`、fingerprint |
| `../detector_comparison/band_yolo.py` | 插件本体。Round 0 新增 `predict_band`、`band_counts` |
| `../basic_ds2/aggregate_cv.py` | 5-fold 合并 + 按 seed 配对统计（`--runs runs_cv --pairs ...`）。**会把缺折的 seed 也合并进去**，只在全部 run 完成后跑 |
| `../basic_ds2/diag_strata.py` | 分层 recall、near miss、F1@0.7。输出的 `summary.md` 标题写着"第 11 轮"，是 basic_ds2 的文字，内容是这里的 |
| `runs_cv/ds2/<exp>/f<fold>_seed<k>/` | 结果：`run_info.json`、`metrics.json`、`preds_{val,test}.json`、`train/results.csv`、`band_{val,test}/*.png`（不入库）、权重（不入库） |
| `runs_cv/cv_summary.{md,json}` | `aggregate_cv.py` 的输出（10 seeds） |
| `runs_cv/strata/summary.{md,json}` | `diag_strata.py` 的输出（10 seeds） |
| `runs_cv/diag/summary.{md,json}` | `diag_operating_point.py` 的输出（10 seeds） |
| `runs_cv/strata_fit/summary.{md,json}` | fit 变体的分层分析（`diag_strata.py --out runs_cv/strata_fit`） |
| `runs_cv/band_value/summary.{md,json}` | `diag_band_value.py` 的输出（10 seeds） |
| `runs_cv/run_cv.log`、`runs_cv/run_cv_iwac2.log` | 两台机器各自的调度日志 |
| `runs_cv_pilot_win5060ti/` | Windows 上修复数据顺序前的 13 个 run，只作参考 |
| `runs_smoke/` | 冒烟测试（不入库） |

arm 一览（`run_cv.py` 的 `EXPS`）：`y8s_pre_none`、`y8s_pre_band`、`y8s_pre_boxmask`、`y8s_pre_g0`（A/A）、`y8s_pre_none_rep`（同 seed 重跑）、`y8s_scr_none`、`y8s_scr_band`、`y8s_scr_boxmask`（从零训练）。

## 附录 B：复现步骤

```bash
# 1. 环境：Python 3.10；torch 按 CUDA 版本装（requirements.txt）；ultralytics 必须是 8.4.70
#    （band_yolo.py 覆盖了 ultralytics 的 load_image / _predict_once，别的版本不要跑）
pip install -r requirements.txt && pip install ultralytics==8.4.70

# 2. 数据目录（datasets/ 不入库；需要 tkinter）。重建结果和 11-15 的目录逐位相同（manifest 里只有仓库路径不同）
cd experiments/revision/detector_comparison
for f in 0 1 2 3 4; do python prepare_data.py --dataset ds2 --fold $f --subset 100 --box-source hbat; done

# 3. 冒烟：看 OK 行、g0 = none、runs_smoke/ds2/y8s_pre_band/f0_seed0/train/train_batch0_band.jpg
cd ../yolo_band_ds2
python run_cv.py --exps y8s_pre_none y8s_pre_band y8s_pre_g0 --folds 0 --seeds 0 1 --epochs 2 --runs runs_smoke --plots --jobs 3 --workers 2

# 4. 主比较（中途断了重跑同一命令，已完成的自动跳过）。分到两台机器时按 --folds / --seeds 分组，
#    各自写自己的 runs_cv/，再 rsync 合并
python run_cv.py --exps y8s_pre_none y8s_pre_band y8s_pre_boxmask --seeds 0 1 2 3 4 5 6 7 8 9 --jobs 3 --workers 2

# 5. 检查和汇总
python run_cv.py --check
python ../basic_ds2/diag_strata.py --runs runs_cv --exps y8s_pre_none y8s_pre_band y8s_pre_boxmask --seeds 0 1 2 3 4 5 6 7 8 9 \
    --pairs y8s_pre_none:y8s_pre_band y8s_pre_none:y8s_pre_boxmask y8s_pre_boxmask:y8s_pre_band

# 6. Round 3（只用 CPU）：hyperbola fit 修框、band 的信息价值
python diag_fit.py --jobs 8
python ../basic_ds2/aggregate_cv.py --runs runs_cv --pairs y8s_pre_none:y8s_pre_band y8s_pre_none:y8s_pre_boxmask y8s_pre_boxmask:y8s_pre_band \
    y8s_pre_none:y8s_pre_none_nms y8s_pre_none:y8s_pre_none_gtfit y8s_pre_none:y8s_pre_band_fit y8s_pre_band:y8s_pre_band_fit y8s_pre_boxmask_fit:y8s_pre_band_fit
python diag_operating_point.py --runs runs_cv      # 读 cv_summary.json，要在 aggregate_cv 之后
python ../basic_ds2/diag_strata.py --runs runs_cv --exps y8s_pre_none y8s_pre_none_gtfit y8s_pre_band_fit --seeds 0 1 2 3 4 5 6 7 8 9 \
    --pairs y8s_pre_none:y8s_pre_none_gtfit y8s_pre_band:y8s_pre_band_fit --out runs_cv/strata_fit
python diag_band_value.py --runs runs_cv
```

## 附录 C：运行记录

| 项 | 内容 |
|---|---|
| 机器 | IWAC1、IWAC2，各一块 RTX 5090（32 GB）；conda env `gpr`（Python 3.10、torch 2.11.0+cu128、ultralytics 8.4.70） |
| IWAC2 的代码 | 新 clone 到 `~/yolo_r1/shuangquxian`（commit `13f99ba7`），没有用那台机器上旧的 checkout；数据目录在那里重建，和 IWAC1 逐位相同 |
| 时间 | 2026-11-21 14:40–15:23，150 个 run；每台机器 3 个 run 并行，每个 run 约 95 秒 |
| 权重 | 留在各自机器的 `runs_cv/ds2/*/*/train/weights/`，不入库，也没有从 IWAC2 拷回来 |
| 预测框裁剪 | `evaluate.py` 在 11-21 改成评估前把预测框裁到图内。YOLO 的 35,914 个预测框本来就都在 224×224 内（ultralytics 自己会裁），改前改后 `cv_summary.json` 逐位相同。IWAC2 上的 `metrics.json` 用的是改之前的 `evaluate.py`，结果一样 |
