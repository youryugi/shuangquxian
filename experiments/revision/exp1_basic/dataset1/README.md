# 实验 1 · 数据集 1（Roboflow gpr-hhalx，pipe + rebar）

实验 1 在数据集 1 上的过程记录：数据检查、为数据集 1 改的代码、各轮的设置和结果。实验 1 的总结论仍只写在 [../README.md](../README.md) 开头，这里放过程和数字。

**定位（2026-11-24 定）**：数据集 2 是主数据集，数据集 1 只做三件事：主结论复现（本目录的主对比，分 pipe / rebar / 全部）、跨数据集迁移（R2-7）、适用边界（钢筋排上 band 近似框、fit 不适用）。递增训练集、形状消融、标注敏感性、检测器和注意力模块对比、困难样本分层都不在数据集 1 上做。完整范围见 [revision_map.md 第 3 节“数据集 1 的范围”](../../../../paper/revision/revision_map.md#3-两个数据集能撑什么)。

**答案（截至 2026-11-23，第 2 轮完成）**：**pipe 在主 config（s224）上复现了数据集 2 的结论。band + fit 比 baseline 高 0.066，比 box-mask 走同样的后处理高 0.049–0.058，10 个 seed 全胜。rebar 被 fit 拉低 0.078，全部类别合起来不显著（+0.012）。在稳健性 config（s224d4gb）上 pipe 没有复现。** 具体：

1. **数据集 1 和数据集 2 很不一样，不能原样搬管线。** rebar（25 张图、303 个目标）目标密、小，band 基本是框状；pipe（145 张图、297 个目标）几何和数据集 2 接近，但几乎都是 clear、两翼完整，数据集 2 上 band + fit 增益最大的 weak / single-wing 分层在这里几乎没有样本（第 0 轮）。
2. **实验 1 的代码已经按数据集参数化，数据集 2 逐位不变**（第 0.5 轮的回归检查）。
3. **baseline 用 224 输入最好，和数据集 2 相同**（第 1 轮）。输入加大到 320–448 时 pipe F1 从 0.39 掉到 0.17–0.24（双曲线超出感受野），rebar 从 0.14 涨到 0.29。合并 F1 约 0.30，比数据集 2（0.47）难得多，rebar 召回率不到 0.15。224 + 检测头步长 4 + 按框 sigma（s224d4gb）在验证集上和 224 打平，把 pipe 换成 rebar。

4. **只加 band 监督（不改后处理）没有作用**，和数据集 2 一样；band 和 box-mask 打平（第 2 轮）。
5. **band + fit 只在 pipe 上有效。** s224 上 pipe 0.375 → 0.441，box-mask + 同样后处理最高 0.392，只放大框 0.407。rebar 上所有后处理都变差，fit 最差（0.124 → 0.046）：检测器本来就把相邻钢筋合成一个宽框，相邻 band 又连成一条带，fit 把一整排当成一条双曲线。用真实 band 拟合也一样，所以是 fit 不适用于成排的团状目标（第 2 轮）。
6. **s224d4gb 上 band + fit 的 pipe 和 baseline 打平、没赢 box-mask**，原因未查（第 2 轮）。

**下一步**：给 fit 加一个不分类别的框宽门槛（fit 框比检测框宽太多就保留原检测框，门槛在验证集上选），看 rebar 能否回到不拉低的水平、pipe 增益是否保持，再看 s224d4gb。标注已是最终版（只有 hyperbola46、hyperbola48 各缺一个目标，按用户决定不补）。

---

## 术语表

| 术语 | 含义 |
|---|---|
| pipe / rebar | 数据集 1 的两类图：管线（`hyperbolaN`、`line*`、部分数字编号）/ 钢筋排（数字编号的一部分）。split 文件里每个 item 的 `category` |
| arm | 同一协议下比较的一种设置：none（无 attention，baseline）、abs（band 监督，本文方法）、boxmask（填实的 HBAT 框监督，对照） |
| config | 输入尺寸 × 热图 sigma 的组合（`s224`、`s224gb`、`s448`、`s448gb`，定义在 [d1_exps.py](d1_exps.py)） |
| fit / fitbox | 沿预测 band 拟合双曲线定框（[../fit_hyperbola.py](../fit_hyperbola.py)，数据集 2 Round 15、18） |
| 合并验证 F1 | 5 折验证集预测合在一起选阈值时的 F1；选 config、选后处理参数只看它 |
| HBAT 框 | 每条 band 的外接框（`prepare_data.py --box-source hbat`），检测目标 |

## 文件

| 文件 | 作用 |
|---|---|
| [d1_data_stats.py](d1_data_stats.py) → [data_stats.md](data_stats.md) | 第 0 轮：数据集 1 / 2 的标注分布对比 |
| [d1_exps.py](d1_exps.py) | 数据集 1 的实验名、参数、后处理变体、默认配对（`run_seeds.py` / `infer_bandbox.py` / `aggregate_cv.py` 都从这里读） |
| [d1_eval.py](d1_eval.py) → `results/` | 分类别（全部 / pipe / rebar）评估和按 seed 配对比较 |
| `../runs_cv/ds1/<exp>/f<fold>_seed<k>/` | 各 run（和数据集 2 的 `runs_cv/ds2/` 平行） |
| `../runs_cv/bandbox_ds1/`、`../runs_cv/bandbox_cache_ds1/` | 数据集 1 的 fit / Voronoi 选参结果和缓存（缓存不入库） |

---

## 第 0 轮（2026-11-23）：数据检查

**为什么做**：数据集 1 的 band 标注已完成（hyperbola46、hyperbola48 各缺一个目标，不补），要先确定数据集 2 上调好的管线能不能原样用。

**做了什么**：[d1_data_stats.py](d1_data_stats.py) 统计两个数据集的标注（全表见 [data_stats.md](data_stats.md)）。

**结果**（框 = band 外接框；@224 = 换算到 224 输入）：

| | ds1 rebar | ds1 pipe | ds2 |
|---|---|---|---|
| 图 / 目标 | 25 / 303 | 145 / 297 | 131 / 222 |
| 每图目标 中位 / 最大 | **8 / 33** | 1 / 8 | 1 / 6 |
| 框宽 / 图宽 中位 | **0.035** | 0.314 | 0.279 |
| 图边截断 | 2% | 28% | 28% |
| 与邻框重叠（IoU > 0） | 23% | 38% | 46% |
| span / thickness @224 中位 | **7.9 / 11.3** | 72.5 / 14.0 | 66.2 / 16.3 |
| slope 5 / 50 / 95% | 0.95 / 7.7 / 11.7 | 0.63 / 3.0 / 9.6 | 1.9 / 3.0 / 4.1 |
| 最近顶点间距 @224 中位 | **10.2 px** | 46.8 px | 53.6 px |
| 224 输入、步长 8 时中心落进同一热图格子 | **19 / 301** | 0 | 0 |
| weak / single-wing 目标 | 0 / 1 | 1 / 7 | 29 / 50 |
| param_status = rough | 43% | 25% | 25% |

**结论**：

- **rebar 的 band 是一团，不是曲线**：span 比 thickness 还小，band 和框几乎没有区别。预期 rebar 上 band 和 box-mask 打平，band 形状的论点只能落在 pipe 上。
- **224 输入 + 原样热图对 rebar 不够**：相邻顶点间距中位 10 px，而热图 sigma 固定 6 格 × 8 px = 48 px，相邻钢筋的高斯完全重叠；19 个目标的中心和别的目标落进同一格，训练目标直接被覆盖。448 输入或步长 4 时只剩 1 个。
- **fit 的 slope 网格要放宽**：数据集 2 的网格是 1–6，ds1 pipe 有 15%、rebar 有 55% 超过 6。
- **pipe 几乎没有 weak / single-wing**：数据集 2 上 band + fit 提升最大的两层（单翼 +0.142）在这里没有样本，pipe 的提升只能来自截断和重叠目标。
- **评估要分类别**：按目标计数时 rebar（15% 的图）占一半目标。

**标注状态**：本地 `todo.txt` 当时列了 5 张 partial，但它是旧的：`3254`（33 / 33）、`hyperbola45`（5 / 5）、`hyperbola6`（5 / 5）在 `annotations.json` 里已经标完（远端 f51ec26 的 todo.txt 也已删掉这三行）。剩下 `hyperbola46`、`hyperbola48` 各缺一个目标（band 1 / 官方框 2，都在 fold 0 的测试集），用户 2026-11-23 决定不补。目标用 HBAT 框，这两个没标的双曲线在测试里检测到会算 FP，影响 2 / 600 个目标。所以第 1、2 轮用的导出就是最终数据，不需要重跑。

## 第 0.5 轮（2026-11-23）：为数据集 1 改代码

新参数的默认值都等于数据集 2 的原设置：

| 文件 | 改动 |
|---|---|
| [../train_basic.py](../train_basic.py) | `--input-size`（默认 224）；`--hm-sigma`（不给 = 原样 6 格，数字 = 固定格数，`box` = CenterNet 按框大小算半径）；`--copy-sampling one`（训练集 3 份亮度副本每轮每个 item 随机取一份，每轮样本数 = item 数）；`run_info.json` 记 `imgsz`、`hm_sigma`、`copy_sampling`、`n_train_per_epoch` |
| [../fit_hyperbola.py](../fit_hyperbola.py) | 像素常数（GAP、MIN_SUPPORT、DV_CAP、默认厚度、顶点搜索范围）按 输入尺寸 / 224 缩放；slope 网格按数据集（ds2 1–6 共 13 个，ds1 0.5–12 共 23 个，步长相同） |
| [../refine_with_band.py](../refine_with_band.py) | `set_dataset`；band 放大到各 run 自己的输入分辨率（从缓存的 meta 算）；Voronoi 的最少像素数按面积缩放 |
| [../infer_bandbox.py](../infer_bandbox.py) | `--dataset ds1`：run、缓存、选参结果放到 `runs_cv/ds1/`、`bandbox_cache_ds1/`、`bandbox_ds1/`；推理按 run 的 `imgsz` 设输入尺寸；d1 的后处理变体（网格同数据集 2） |
| [../run_seeds.py](../run_seeds.py) | `d1_*` 实验读 `ds1_f<k>_s100_hbatbox`，写 `runs_cv/ds1/` |
| [../aggregate_cv.py](../aggregate_cv.py) | `--dataset ds1`：读 `runs_cv/ds1/`，默认配对来自 [d1_exps.py](d1_exps.py)，输出 `cv_summary_ds1.{json,md}`（不覆盖数据集 2 的） |

**回归检查（数据集 2 必须逐位不变）**：

- 训练：`hbatbox_flipfix_none`、`hbatbox_flipfix_abs` 的 f0 seed0 用改后的代码重训（`--workers 1`），`preds_val.json`、`preds_test.json` 的 md5 和原 run 完全相同。
- fit：`hbatbox_flipfix_abs` f0 seed0 的拟合缓存（t = 0.7 和真实 band，val / test）重算，逐字节相同。
- 后处理：`hbatbox_flipfix_{abs_fitbox, boxmask_bandbox_nms, none_scale}` 在选中参数下重算 5 折 val / test 预测，和已有 run 逐字节相同。

**数据导出**：`prepare_data.py --dataset ds1 --fold 0..4 --box-source hbat` → `shared/datasets/ds1_f{0-4}_s100_hbatbox/`。每折训练 108–124 个 item（280–332 张图，含副本）、验证 17 个 item、测试 29–45 个 item；5 折测试集合起来 600 个目标，每个目标恰好在一折的测试集里。叠图抽查一张 rebar、一张 pipe，框、band 和回波对齐。

## 第 1 轮（2026-11-23）：选输入尺寸和热图 sigma

**为什么做**：第 0 轮表明 224 输入 + 固定 sigma 对 rebar 不够，但改输入尺寸和 sigma 会同时改 baseline。按协议，这两个只在 baseline 上、按合并验证 F1 选一次，选定后 none / abs / boxmask 共用，不为 band 单独调。

**做了什么**：只跑 none arm，每个 config 5 折 × 3 seeds（`--augment --copy-sampling one --workers 1`，其余与数据集 2 相同：120 轮、Adam 5e-4、batch 8）。先跑 224 / 448 × 两种 sigma；看到 pipe 和 rebar 方向相反后补了 320、384 两个中间尺寸，再补 224 输入 + 检测头步长 4（数据集 2 第 8 轮的结构）。评估：[d1_eval.py](d1_eval.py)，结果 [results/r1_config.md](results/r1_config.md)。

**结果**（3 seeds 均值；阈值在 5 折合并验证集上、全部类别一起选）：

| config | 合并验证 F1（选参用） | 测试 F1 全部 | pipe | rebar | rebar recall |
|---|---|---|---|---|---|
| s224（= 数据集 2 的设置） | 0.282 ± 0.013 | 0.301 | **0.391** | 0.138 | 0.084 |
| s224gb | 0.267 ± 0.017 | 0.274 | 0.362 | 0.126 | 0.081 |
| s224d4（步长 4） | 0.282 ± 0.028 | 0.283 | 0.365 | 0.133 | 0.084 |
| s224d4gb（步长 4 + 按框 sigma） | **0.288 ± 0.038** | 0.282 | 0.322 | 0.211 | 0.139 |
| s320 | 0.205 ± 0.023 | 0.192 | 0.238 | 0.107 | 0.069 |
| s384 | 0.221 ± 0.008 | 0.191 | 0.206 | 0.170 | 0.116 |
| s448 | 0.209 ± 0.021 | 0.208 | 0.165 | **0.286** | 0.205 |
| s448gb | 0.209 ± 0.005 | 0.216 | 0.190 | 0.252 | 0.179 |

s224 按图像来源拆开（测试集，3 seeds 合计）：`line`（灰度单目标）F1 0.58，`hyperbolaN`（蓝色、多目标）0.32，数字编号的 pipe 0.33，rebar 0.14（recall 0.08）。448 时 `line` 掉到 0.08。

**结论**：

- **输入一变大，pipe 就掉。** 224 → 320 → 448，pipe F1 0.39 → 0.24 → 0.17；rebar 在 448 涨到 0.29。这个网络三次下采样后感受野大约 70–85 px，pipe 在 224 上宽约 70 px，刚好装得下；输入变大后一条双曲线超出感受野。
- **数据集 1 的 baseline 明显比数据集 2 难**：最好的 config 合并 F1 约 0.30（数据集 2 是 0.47），rebar 召回率不到 0.15。
- **选 config**：合并验证 F1 前三名 s224d4gb 0.288、s224 0.282、s224d4 0.282，差距在 seed 噪声以内（std 0.013–0.038），按验证集分不出来。s224d4gb 是拿 pipe（−0.07）换 rebar（+0.07）。所以第 2 轮两个都跑：
  - **s224**：和数据集 2 完全相同的网络和输入（只多了副本抽样），两个数据集用同一个模型，作为主表；
  - **s224d4gb**：验证集排第一的 config，作为稳健性对照（rebar 检测得更好时结论是否不变）。

## 第 2 轮（2026-11-23）：三个 arm + fit 定框

**为什么做**：数据集 2 Round 18 的主比较在数据集 1 上重做：none / abs / boxmask 三个 arm，之后 band + fit 定框、box-mask 走同样的后处理，按类别报告。后处理方法（band + fit）沿用数据集 2 定下的，不在数据集 1 上重新挑方法，只在合并验证集上选它的参数。

**做了什么**：s224 和 s224d4gb 两个 config × 三个 arm × 5 折 × 10 seeds（none 的 seed 0–2 复用第 1 轮），IWAC1，`--workers 1`，训练共 55 分钟。之后 `infer_bandbox.py --dataset ds1` 推理、拟合、在合并验证集上全局选后处理参数（网格与数据集 2 相同，阈值和参数都按全部类别一起选）。评估：[d1_eval.py](d1_eval.py)，结果 [results/r2_fusion.md](results/r2_fusion.md)（只改训练）、[results/r2_fit.md](results/r2_fit.md)（加后处理）。

**结果**（测试 F1，10 seeds 均值；Δ 是按 seed 配对的差，括号里是 95% bootstrap CI 和 胜/负 seed 数）：

s224（主表，和数据集 2 同一个网络和输入）：

| 方法 | 全部 | pipe | rebar |
|---|---|---|---|
| none（baseline） | 0.286 | 0.375 | **0.124** |
| none + scale（只放大框） | 0.294 | 0.407 | 0.091 |
| abs（只加 band 监督，不改后处理） | 0.283 | 0.380 | 0.101 |
| boxmask | 0.286 | 0.383 | 0.107 |
| boxmask + Voronoi bandbox | 0.283 | 0.392 | 0.078 |
| boxmask + fit | 0.265 | 0.382 | 0.048 |
| band + Voronoi bandbox | 0.294 | 0.419 | 0.073 |
| **band + fit（本文方法）** | **0.297** | **0.441** | 0.046 |
| band + fit，用真实 band 拟合（上限） | 0.421 | 0.631 | 0.016 |

band + fit 在 pipe 上的配对差：

| 对照 | ΔF1 pipe | 胜/负 |
|---|---|---|
| none | **+0.066** [+0.037, +0.098] | 10/0 |
| none + scale | +0.034 [+0.013, +0.053] | 8/2 |
| boxmask + fit | **+0.058** [+0.031, +0.089] | 10/0 |
| boxmask + Voronoi bandbox | **+0.049** [+0.037, +0.062] | 10/0 |
| band + Voronoi bandbox | +0.022 [+0.015, +0.030] | 10/0 |

在 rebar 上 band + fit 比 none 低 0.078 [−0.088, −0.069]（0/10），全部类别合起来 +0.012 [−0.006, +0.031]（6/4，不显著）。

s224d4gb（稳健性对照：步长 4 + 按框 sigma）：

| 方法 | 全部 | pipe | rebar |
|---|---|---|---|
| none | 0.284 | 0.328 | 0.205 |
| none + scale | 0.284 | 0.344 | 0.180 |
| abs | 0.277 | 0.314 | 0.212 |
| boxmask + Voronoi bandbox | 0.276 | 0.342 | 0.161 |
| boxmask + fit | 0.261 | 0.337 | 0.125 |
| band + Voronoi bandbox | 0.266 | 0.340 | 0.135 |
| band + fit | 0.254 | 0.323 | 0.127 |
| band + fit，真实 band | 0.358 | 0.528 | 0.017 |

在 s224d4gb 上，band + fit 的 pipe 和 none 打平（−0.005，5/5），比 boxmask + fit 低 0.014（3/7，不显著），全部类别比 none 低 0.029（1/9）。

**rebar 为什么被后处理拉低**（s224，seed 0，5 折测试；每个真实框取得分前 2N 个预测里 IoU 最高的那个）：

| | 最佳 IoU 中位 | IoU ≥ 0.5 的比例 | 预测框宽 / 真实框宽 中位 | 得分 > 0.3 的预测数 / 真实目标数 |
|---|---|---|---|---|
| abs（无后处理） | 0.04 | 0.11 | 3.4 | 96 / 303 |
| band + Voronoi bandbox | 0.10 | 0.08 | 4.3 | 93 / 303 |
| band + fit | 0.05 | 0.06 | 6.1 | 74 / 303 |
| 真实 band + fit | 0.05 | 0.04 | 7.9 | 49 / 303 |
| （对比：pipe 上 band + fit） | 0.55 | 0.57 | 1.03 | 376 / 297 |

**结论**：

- **只改训练（band 监督）在两个 config 上都没有作用**，和数据集 2 相同：none → abs 在 s224 上全部 −0.003，pipe +0.005（不显著）；band 和 boxmask 打平。
- **在 s224 上，pipe 复现了数据集 2 Round 18 的结论**：band + fit 比 baseline 高 0.066，比 box-mask 走同样两种后处理高 0.049–0.058，10 个 seed 全胜；比只放大框（+0.034，8/2）和 band + Voronoi（+0.022，10/0）也高。
- **在 rebar 上，所有后处理都变差，fit 最差**（−0.078）。根本原因在检测器：baseline 本来就把相邻的几根钢筋合成一个框（预测框宽是真实的 3.4 倍，303 个目标只出 96 个预测框）。相邻钢筋的 band 间距约等于厚度（第 0 轮：10 px 对 11 px），连成一条带，fit 把一整排当成一条双曲线，框宽变成 6–8 倍。用真实 band 拟合也一样（0.016），所以这是 fit 不适用于成排的团状目标，不是 band 预测得不好。rebar 的 band 本身就是框状，不适合用来论证 band 形状。
- **全部类别合起来不显著**（+0.012）：rebar 的损失抵掉了 pipe 的增益，两类目标数各占一半。
- **稳健性对照 s224d4gb 上 pipe 没有复现**：band + fit 和 none 打平，没赢 box-mask。原因没有查，一个可能是：阈值和后处理参数按全部类别选，s224d4gb 的 rebar 检测更好、在验证集里占的分量更大，选出来的参数对 pipe 不利（s224 选的是 seed 阈值 0.1，s224d4gb 选的是 0.5）。
- **pipe 的上限还很远**：用真实 band 拟合，pipe 是 0.631（s224），预测 band 只到 0.441。和数据集 2 一样，瓶颈是 band 预测。

**还没做、可以做的**：

1. **按 fit 框宽设门槛**（不分类别）：fit 框比检测框宽太多（例如超过 2 倍）时保留原检测框，门槛在合并验证集上选。这样 rebar 应回到 abs 的水平，pipe 的增益不受影响。只用 CPU，约 1 小时。
2. 上面做完后，在 s224d4gb 上再看 pipe 是否复现；如果仍不复现，再按类别分开选参数，确认是不是选参偏向 rebar。
3. rebar 的检测本身（召回率不到 0.15）是检测器的问题，和 band 无关，不在本文论点里处理。
