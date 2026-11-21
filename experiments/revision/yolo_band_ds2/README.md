# YOLOv8 + band 辅助监督（数据集 2）

**要回答的问题**：band 监督插进一个成熟检测器（COCO 预训练的 YOLOv8s）还有没有用？

**当前状态（2026-11-21）**：代码审查和冒烟测试完成（Round 0）。Round 1（3 seeds 先导）**换到另一台电脑重跑**，步骤见 [交接](#交接换电脑重跑-round-12026-11-21)。本机跑了一部分，用的是修复前的数据顺序，只作参考。

和 [basic_ds2](../basic_ds2/README.md) 的关系：basic_ds2 用三层卷积的 CenterNet 做了 16 轮，结论是：
- fusion 没有增益；
- band 的增益主要来自"把框画对"；
- 这件事一个普通的 stage 2 框回归（不用 band）做得更好（basic_ds2 Round 14）。

弱 baseline 会放大增益，所以这里换成强 baseline 再测一次。审稿意见 R1-3 / R2-3 也要求这个实验。

## 目录

- [交接：换电脑重跑 Round 1](#交接换电脑重跑-round-12026-11-21)
- [术语表](#术语表)
- [实验设计](#实验设计)
- [Round 0：代码审查和改动](#round-0代码审查和改动2026-11-21)
- [Round 1：先导，3 seeds](#round-1先导3-seeds)
- [下一步](#下一步)
- [附录：文件和命令](#附录文件和命令)

## 交接：换电脑重跑 Round 1（2026-11-21）

### 现在到哪了

- 代码改完并测过（Round 0）：配对检查、A/A、band arm 确定性、增广对齐、band 导出都通过。
- 本机（Windows，RTX 5060 Ti）跑了 13 / 45 个 run 后停了。这些 run 用的是**修复前的数据顺序**（所有 seed 共用同一条增广序列，见 Round 0 第 8 条），已改名为 `runs_cv_pilot_win5060ti/`，**不要合并进正式结果**。
- 新电脑从空的 `runs_cv/` 开始跑全部 45 个 run。

### 新电脑上的步骤

```bash
# 0. 同步代码（数据集 2 的原图、HBAT 标注、划分文件都在 git 里）
git pull                                   # 或 git clone https://github.com/youryugi/shuangquxian.git

# 1. 环境：Python 3.10；torch 按 CUDA 版本装（见 requirements.txt）；ultralytics 必须是 8.4.70
pip install -r requirements.txt
pip install ultralytics==8.4.70
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
#    -> 期望 ... True 8.4.70。版本不同不要跑：band_yolo.py 覆盖了 ultralytics 的 load_image / _predict_once

# 2. 重建数据目录（datasets/ 不入库，约 1 分钟；需要 tkinter，Linux 上没有就 apt install python3-tk）
cd experiments/revision/detector_comparison
for f in 0 1 2 3 4; do python prepare_data.py --dataset ds2 --fold $f --subset 100 --box-source hbat; done
#    -> 每折 train 91–92 张、val 13、test 26–27；"items without band 0"

# 3. 冒烟（约 2 分钟；第一次会下载 yolov8s.pt 到 detector_comparison/）
cd ../yolo_band_ds2
python run_cv.py --exps y8s_pre_none y8s_pre_band y8s_pre_g0 --folds 0 --seeds 0 1 --epochs 2 --runs runs_smoke --plots
```

冒烟必须看三件事，任何一件不对就先停：

| 检查 | 怎么看 | 应该是 |
|---|---|---|
| 配对 | 输出最后的 `OK fold 0 seed 0 / seed 1` | 两行都是 OK；同一 seed 下三个 arm 的 hash 相同，seed 0 和 seed 1 的第二段 hash 不同 |
| A/A | `[..] y8s_pre_g0` 和 `y8s_pre_none` 的 test F1 | 同一 seed 下完全相同。不同说明这台机器不是确定性的，在 README 里记下来，继续跑也可以，但 Δ 里会多一份噪声 |
| 对齐 | `runs_smoke/ds2/y8s_pre_band/f0_seed0/train/train_batch0_band.jpg` | 绿色 band 贴在回波上、在红框里 |

```bash
# 4. 正式先导：3 arm × 5 fold × 3 seed = 45 个 run（中途断了重跑同一命令，已完成的自动跳过）
python run_cv.py --exps y8s_pre_none y8s_pre_band y8s_pre_boxmask --seeds 0 1 2 --jobs <N> --workers <W>
#    Windows：--jobs 1 --workers 1（worker 多了会页面文件不足 MemoryError）。5060 Ti 上约 3 分钟/run，共约 2.5 小时
#    Linux + RTX 5090：先试 --jobs 3 --workers 2，看 nvidia-smi 显存（每个 run 约 4 GB）

# 5. 检查和汇总
python run_cv.py --check                   # 必须 "0 with mismatching fingerprints"
python ../basic_ds2/aggregate_cv.py --runs runs_cv \
    --pairs y8s_pre_none:y8s_pre_band y8s_pre_none:y8s_pre_boxmask y8s_pre_boxmask:y8s_pre_band
python ../basic_ds2/diag_strata.py --runs runs_cv --exps y8s_pre_none y8s_pre_band y8s_pre_boxmask --seeds 0 1 2 \
    --pairs y8s_pre_none:y8s_pre_band y8s_pre_boxmask:y8s_pre_band
```

### 跑完要同步回来的东西

`git add` 这些（权重、PNG、JPG 已在 .gitignore 里，不会进来）：
- `runs_cv/`：每个 run 的 `run_info.json`、`metrics.json`、`preds_{val,test}.json`、`train/results.csv`、日志；
- `runs_cv/cv_summary.md`、`cv_summary.json`（aggregate_cv 的输出）；
- `runs_cv/strata/summary.md`、`summary.json`（diag_strata 的输出）。

不要提交 `runs_smoke/`（已忽略）。在 README 里写一句是哪台机器、哪块卡、跑了多久。

### 注意

- **3 个 arm 必须在同一台机器上跑完**。不能一部分在这台、一部分在那台：换卡会改变结果（basic_ds2 Round 4：±0.013），`run_cv.py` 又会跳过已有 `metrics.json` 的 run，混进去也看不出来。
- band 图（`band_test/*.png`）不入库。下一步"推理时用 band 修框"要用它们，所以**新电脑上的 `runs_cv/` 先别删**。
- 本机的 13 个 run 留在 `runs_cv_pilot_win5060ti/`，可以用来对比修复数据顺序前后的差别，不要拿来算正式结果。

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
| band IoU | 预测 band（阈值 0.5）和 HBAT band 的像素 IoU，测试集像素合计（同 basic_ds2 Round 16） |
| near miss / grouped | 最佳 IoU 在 0.3–0.5 的漏检；official box 里套着 ≥2 条双曲线的目标（同 basic_ds2） |

## 实验设计

| 项 | 设置 | 来自哪条经验 |
|---|---|---|
| 数据 | 数据集 2，`ds2_f{0-4}_s100_hbatbox`，目标 = HBAT box | basic_ds2 Round 2–3：official box 是 scene-level，和方法要学的东西不一致 |
| 划分 | 5-fold，按来源分组；每张图恰好在一个 fold 的测试集里 | basic_ds2 附录 A |
| seed | 先导 3 个，正式 10 个；按 seed 配对。seed 同时决定检测头初始化、数据顺序和增广 | 投稿版只有 3 seeds、初始化没对齐，+0.06 是噪声（basic_ds2 Round 4） |
| 评估 | pooled threshold → test F1@IoU0.5；另报 AP50、F1@0.7、分层 recall | 同 basic_ds2，可直接用 `aggregate_cv.py`、`diag_strata.py` |
| 模型选择 | 最后一个 epoch（`last.pt`） | 验证集每折只有 13 张图；ultralytics 8.4.70 的 best.pt 只看 mAP50-95，噪声大，而且两个 arm 可能挑到不同 epoch |
| 训练 | 100 epoch，640 输入（224 原图放大），batch 16，ultralytics 默认超参和增广 | 和预训练尺度一致。640 下 P3 是 80×80，相当于原图 stride 2.8，比 basic_ds2 的 56×56 还细 |
| band 损失 | BCE + Dice，λ = 1，前 3 epoch 线性热身，在 160×160 上算（soft target） | basic_ds2 Round 10：λ 没有影响，所以不扫 |

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
| 7 | Windows 上 4 个 DataLoader worker 报 MemoryError | 每个 worker 都加载一遍 torch + CUDA DLL，页面文件不够 | `--workers 1` |
| 8 | **ultralytics 8.4.70 的训练 DataLoader 用写死的种子**（`build.py`：`6148914691236517205 + RANK`） | 数据顺序和 mosaic / 仿射 / 翻转对所有 `--seed` 都一样，seed 只改变检测头的初始化。多个 seed 共用同一条增广序列，seed 间的差异被低估。这是看 fingerprint 时发现的：fold 0 的 seed 0、1、2 第一个 batch 的 hash 都是 `52efd014…` | `train_ultralytics.py` 在建好 DataLoader 后用 `--seed` 重设它的 generator（`--fixed-data-order` 可以恢复原行为）。同一 seed 的各 arm 仍然相同 |

没改、但要知道的：

- **approach A 只能测"band 监督有没有改善特征"**。basic_ds2 里有效的是推理时用 band 修框（Round 12），在 YOLO 上要另做（见下一步）。
- **6 个 unparam 目标**（每折的 manifest 都记着）有 HBAT box、没有 band：band arm 把它们当背景，boxmask arm 把它们当前景。影响很小，但是两个 arm 的一个不对称。对齐检查图右下角的点状目标就是这种。
- 导出预测时 NMS IoU = 0.7（ultralytics 默认）。grouped targets 互相重叠，NMS 会影响它们的 recall，三个 arm 用同一个值。

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

## Round 1：先导，3 seeds

### 为什么做

先看方向和量级，再决定要不要花 10 seeds 的时间。

### 做了什么

`y8s_pre_none` / `y8s_pre_band` / `y8s_pre_boxmask` × 5 folds × seeds 0–2 = 45 个 run。**在另一台电脑上跑**（见交接）。

### 本机的部分结果（修复数据顺序前，只作参考）

13 个 run，fold 0 三个 seed 齐全，fold 1 不全。fold 0 上（不是 5-fold 合并）：

| arm | test F1（3 seeds） | AP50 |
|---|---|---|
| none | 0.794 ± 0.039 | 0.854 |
| band | 0.792 | 0.842 |
| boxmask | 0.775 | 0.848 |

- 看不出方向：none → band 是 −0.002，CI [−0.091, +0.054]。
- **YOLOv8s baseline 在 fold 0 上约 0.79**。basic_ds2 的 CenterNet baseline 5-fold 合并是 0.473，stage 2 是 0.632。两者不是同一个口径（单折 vs 合并），但差距很大，说明这个 baseline 强得多。

### 结果 / 结论

（等新电脑跑完）

## 下一步

1. **诊断 YOLO baseline 的瓶颈**（`diag_strata.py`）：near miss 占漏检的多少、grouped recall 多少。basic_ds2 里 band 的增益来自 near miss；如果 YOLO 的 near miss 很少，修框这条路在这里就没有空间。
2. **推理时用 band 修框**（approach C）：用导出的 band 图做 basic_ds2 Round 12 / 15 的 box refinement / hyperbola fit。只在第 1 步显示有空间时做。
3. **从零训练**（`y8s_scr_*`）：没有预训练特征时 band 监督是否更有用。
4. 方向明确后扩到 10 seeds，再做数据集 1（等 band 标注补齐）。

## 附录：文件和命令

| 文件 | 作用 |
|---|---|
| `run_cv.py` | 实验矩阵（arm 定义在 `EXPS` 里）、并行、跳过已完成的 run、`--check` 配对检查 |
| `../detector_comparison/train_ultralytics.py` | 单个 run：训练 → 导出预测 → 评估。本轮新增 `--run-name`、`--weights`、`--save-band`、`--fixed-data-order`、fingerprint |
| `../detector_comparison/band_yolo.py` | 插件本体。本轮新增 `predict_band`、`band_counts` |
| `../basic_ds2/aggregate_cv.py` | 5-fold 合并 + 按 seed 配对统计（`--runs runs_cv --pairs ...`） |
| `../basic_ds2/diag_strata.py` | 分层 recall、near miss、F1@0.7 |
| `runs_cv/ds2/<exp>/f<fold>_seed<k>/` | 结果：`run_info.json`、`metrics.json`、`preds_{val,test}.json`、`band_{val,test}/*.png`（不入库）、`train/`（ultralytics 目录，权重不入库） |
| `runs_cv_pilot_win5060ti/` | 本机修复数据顺序前的 13 个 run，只作参考 |
| `runs_smoke/` | 冒烟测试（不入库） |

arm 一览（`run_cv.py` 的 `EXPS`）：`y8s_pre_none`、`y8s_pre_band`、`y8s_pre_boxmask`、`y8s_pre_g0`（A/A）、`y8s_pre_none_rep`（同 seed 重跑）、`y8s_scr_none`、`y8s_scr_band`、`y8s_scr_boxmask`（从零训练）。
