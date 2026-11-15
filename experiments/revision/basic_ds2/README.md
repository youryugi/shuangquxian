# 基础实验重跑（数据集 2，新标注）：attention supervision 到底有没有用

**要回答的问题**：论文说"用 hyperbola band mask 监督 attention，检测 F1 从 0.67 升到 0.73"（Table IV）。换上重新标注的数据，用更严格的实验方法再跑一遍，这个提升还在吗？

**答案（截至 2026-11-15）**：**不在。** 严格方法下 ΔF1 = +0.004，95% CI [−0.021, +0.028]，和 0 没有区别。原来的提升是实验方法有漏洞造成的。不过 attention 分支并不是什么都没学到：它学到了"目标有多大"，只是这个信息没有用到 box regression 上（见 Round 6）。

**最重要的待办**：在数据集 1（论文主数据）上用同样的严格方法再跑一次，然后决定论文的主张怎么改。

---

## 术语表

本文中的专业词统一用英文，含义如下。

| 术语 | 含义 |
|---|---|
| **baseline** | 只有检测头的模型，不加 attention |
| **+attention** | 检测头 + attention 分支，attention 用 band mask 监督（即论文方法） |
| **arm** | 对比实验中的一组。本文只有两组：baseline arm 和 +attention arm |
| **band / band mask** | 沿双曲线画出的一条带状区域，用作 attention 的监督目标 |
| **official box** | 数据集自带的标注框。经常把相邻的 2–4 条双曲线框成一个大框（scene-level） |
| **HBAT box** | 由 HBAT 标注的 band 几何算出的框，一条双曲线一个框（instance-level） |
| **box granularity mismatch** | 标注框和模型输出对"一个目标"的定义不一致：标注是一个大框套多条，模型是一条一个小框。小框和大框的 IoU 不到 0.5，于是被判成 false positive |
| **fold / 5-fold CV** | 5-fold cross-validation：把 131 张图分成 5 份，轮流用其中 1 份做测试。**single fold** 指只跑了其中一份（fold 0），测试集只有 27 张 |
| **seed** | 随机种子。同一配置换不同 seed 重跑，用来估计随机波动 |
| **paired initialization** | 两个 arm 用同一个 seed 时，检测部分的初始权重逐个参数完全相同，这样两者的差值只反映 attention 本身的作用 |
| **data leakage** | 测试图的"近亲"（同一张原图的增强版本）出现在训练集里，模型等于提前见过答案 |
| **TP / FP / FN** | true positive（检对）/ false positive（误检，报了不存在的目标）/ false negative（漏检，没检出真实目标） |
| **precision / recall** | precision = TP /（TP + FP）；recall = TP /（TP + FN） |
| **score threshold** | 模型输出分数高于这个值才算检出。在 validation set 上选，然后用在 test set 上 |
| **box regression** | 网络预测框的位置和宽高的那一部分 |
| **post-processing** | 网络输出之后、不需要重新训练的处理步骤（例如 NMS） |
| **upper bound (oracle)** | 用人工标注的真值代替网络预测，得到的理论最好成绩 |
| **receptive field** | 网络某一层的一个输出格子能"看到"的输入图像范围（像素），由卷积核大小和 pooling 次数决定 |
| **stride** | 特征图相对输入图的缩小倍数。stride 8 = 224 的图变成 28×28；stride 4 = 56×56 |
| **fusion** | 把 attention map 和检测特征合并的那一步。concat = 拼接成多一个通道；gate = 特征乘以 (1 + attention) |
| **extent features** | Round 8 新加的 4 个通道：在 attention map 上用水平 / 竖直的 box filter 算"band 在这附近延伸了多宽"，直接喂给检测头 |
| **λ (lam_att)** | attention loss 的权重，总 loss = 检测 loss + λ × attention loss，论文默认 0.7 |

---

## 目录

- [一页总览](#一页总览)
- [Round 1：标注文件用错，作废](#round-1标注文件用错作废)
- [Round 2：新标注 + official box，single fold](#round-2新标注--official-boxsingle-fold)
- [Round 3：换 HBAT box + 水平翻转增强，3 seeds](#round-3换-hbat-box--水平翻转增强3-seeds)
- [Round 4：paired initialization + 5-fold + 10 seeds（正式结果）](#round-4paired-initialization--5-fold--10-seeds正式结果)
- [Round 5：诊断为什么没有提升](#round-5诊断为什么没有提升)
- [Round 6：用 band 修正框，验证诊断](#round-6用-band-修正框验证诊断)
- [Round 7–10 之前：三个假设的分析](#round-710-之前三个假设的分析)
- [Round 7：upper bound 诊断，真实 band 直接做 fusion](#round-7upper-bound-诊断真实-band-直接做-fusion)
- [Round 8：把尺寸信息做进网络](#round-8把尺寸信息做进网络)
- [Round 9：数据量曲线](#round-9数据量曲线)
- [Round 10：λ 扫描 + gate fusion](#round-10λ-扫描--gate-fusion)
- [下一步](#下一步)
- [附录 A：实验设置（正式协议）](#附录-a实验设置正式协议)
- [附录 B：文件与数据位置](#附录-b文件与数据位置)
- [附录 C：潜在问题清单](#附录-c潜在问题清单)

---

## 一页总览

| Round | 日期 | 做了什么 | baseline F1 | +attention F1 | 结论 |
|---|---|---|---|---|---|
| 1 | 11-14 下午 | 误用了旧标注（HBAT 启动时没加 `--out`） | 0.42 | 0.41 | **作废** |
| 2 | 11-14 晚 | 新标注、official box、single fold、1–3 seeds | 0.44 | 0.44 | 两个 arm 分数一样；比论文低 0.3；不同 seed 之间波动很大 |
| 3 | 11-14 晚 | 改用 HBAT box（一条双曲线一个框）、开水平翻转增强，3 seeds | 0.52 | 0.57 | 整体分数上来了；看起来 +0.04，但只有 3 个 seed |
| **4** | 11-14 深夜 | **paired initialization + 5-fold + 10 seeds** | **0.475** | **0.479** | **ΔF1 = +0.004 [−0.021, +0.028]，没有提升。正式结果** |
| 5 | 11-15 | 诊断：分辨率问题？代码 bug？瓶颈在哪？ | — | — | 瓶颈是 FN（漏检）；约 4 成 FN 是框画小了；attention 解决的不是这个问题 |
| 6 | 11-15 | 用 attention map 里的 band 把框扩大（不重新训练） | 0.475 | **0.541** | band 的信息有用，但没进入 box regression。只是 post-processing，不进论文 |
| 7 | 11-15 | upper bound：真实 band 直接送进 fusion（训练和 inference 都给） | 0.475 | 36/50 run，初步 ≈ baseline | **初步：真实 band 走现有 fusion 也没有提升（Δ +0.003，18 胜 18 负）→ 瓶颈是 fusion 通路，不是 band 质量**。待补 fold 4 |
| 8 | 11-15 | 尺寸信息进网络：a) extent features fusion；b) 检测头搬到 stride 4 | — | 待跑（换机器） | 把 Round 6 的 +0.063 变成网络自己的能力 |
| 9 | 11-15 | 数据量曲线：训练集 25% / 50% | — | 待跑 | 提升是否在数据更少时出现 |
| 10 | 11-15 | λ = 0.1 / 0.3 / 1.5 + gate fusion | — | 待跑 | 审稿人会问的对照 |

---

## Round 1：标注文件用错，作废

**日期**：11-14 下午

**做了什么**：用 `hbat/annotations.json` 生成数据，baseline 和 +attention 各跑 1 个 seed。

**结果**：0.419 vs 0.411，远低于论文。

**出了什么问题**：用户发现有几张图明明标过了，数据里却是空的。查下来，用户重新标注的文件保存在图片目录 `original/original_Utilities/annotations.json`（HBAT 不加 `--out` 时的默认保存位置），而训练读的是另一份旧的初始标注。对比两份文件：131 张图全部有改动，是一次完整的重新标注。

**处理**：Round 1 的结果全部作废，归档到 `runs_legacy_init_1114/`；重新标注的文件移到 `hbat/`，旧的初始标注改名为 `annotations_legacy_init_20261114.json`。

> ⚠️ 以后启动 HBAT 请加 `--out data/additional_utilities/hbat`。标注改过之后要重新运行 `prepare_data.py`。

---

## Round 2：新标注 + official box，single fold

**日期**：11-14 晚

**做了什么**：用重新标注的 band 作 attention 的监督目标，检测目标用 official box。只跑 fold 0（train 91 / val 13 / test 27 张），先跑 1 个 seed，再补到 3 个。

**为什么这样做**：当时的决定是"检测框一律用 official box，HBAT 只提供 mask"。

**结果**（test set F1，score threshold 在 validation set 上选）

| seed | baseline | +attention |
|---|---|---|
| 0 | 0.444 | 0.442 |
| 1 | 0.222 | 未跑 |
| 2 | 0.430 | 0.411 |

另做了一个对照：把 attention 的监督目标从 band 换成"把 official box 整个涂满"，F1 降到 0.308，比用 band 差很多。说明 band 的形状确实起作用。

**学到什么**

1. 两个 arm 分数基本一样，论文里的 +0.06 没有出现。
2. 整体分数从论文的 0.67–0.73 掉到 0.44。查了原因：
   - **旧实验有 data leakage**：旧实验把 247 张增强图随机切分成训练集和测试集，84% 的测试图在训练集里有同一张原图的增强版本。新的切分按来源分组，同一来源的图只会在同一边，这部分"靠见过而得的分数"就没了。
   - **box granularity mismatch**：official box 经常把 2–4 条双曲线框成一个大框，而模型是一条双曲线出一个小框。小框和大框的 IoU 达不到 0.5，就被判成 FP。test set 27 张图里有 12 张是这种情况；FP 中 80–90% 都是"框住了 official box 的一部分"。
3. 不同 seed 之间波动极大：baseline 的 seed 1 直接掉到 0.222。波动这么大，3 个 seed 得不出任何结论。

---

## Round 3：换 HBAT box + 水平翻转增强，3 seeds

**日期**：11-14 晚

**做了什么**：针对 Round 2 的两个问题各改一项：检测目标改成 HBAT box（一条双曲线一个框）；打开原脚本里自带的水平翻转 data augmentation。2 × 2 共四种组合，各跑 3 个 seed。

**结果**（fold 0，3 seeds 平均）

| 检测框 | augmentation | baseline | +attention | ΔF1 |
|---|---|---|---|---|
| official box | 无 | 0.37 | 0.43 | （seed 1 未跑） |
| official box | 水平翻转 | 0.38 | 0.43 | +0.049（3 个 seed 都为正） |
| HBAT box | 无 | 0.56 | 0.52 | −0.032 |
| HBAT box | 水平翻转 | 0.52 | 0.57 | +0.044（3 个 seed 中 2 个为正） |

**学到什么**

1. **box granularity mismatch 是整体分数低的主要原因**：同样的模型，检测目标换成 HBAT box，F1 从 0.4 升到 0.55。
2. **水平翻转让结果更稳定**：不同 seed 之间的 std 从 0.08–0.12 降到 0.03–0.04。
3. 开了翻转之后看到 +0.04–0.05，当时以为提升回来了。**后来证明是随机波动**（见 Round 4）。

---

## Round 4：paired initialization + 5-fold + 10 seeds（正式结果）

**日期**：11-14 深夜

**为什么还要做**：Round 3 的 +0.04 有三个问题：
- **两个 arm 的初始化其实没对齐**：两个 arm 用同一个 seed 号，但 +attention 模型要多建几层，多消耗了一段随机数，导致后面检测头的初始权重和数据顺序都跟 baseline 不一样了。所以两者的差值里混进了初始化带来的随机差异。
- 只有 3 个 seed，而且 single fold 的 test set 只有 47 个目标，多检对一个目标 F1 就变化 0.02。
- 要检测的效应是 0.04，随机波动也是 0.04，要可靠地测出来大约需要 15 个 seed。

**做了什么**
- **paired initialization**：检测部分用 seed 初始化，两个 arm 的参数逐个完全相同（可用 `init_fingerprint` 核对）；attention 分支用另一个 seed 单独初始化；数据顺序用独立的随机生成器固定。
- **5-fold CV**：按来源分组切成 5 份，5 个 fold 的 test set 合起来正好是全部 131 张图。
- **10 seeds** × 5 folds × 4 组配置 = 200 个 run，共 74 分钟。
- **评估**：每个 seed 把 5 个 fold 的 validation 预测合在一起选一个 score threshold，再把 5 个 fold 的 test 预测合在一起算一个 F1。这样每个 seed 得到一个数，10 个数做 paired 统计检验。

**结果**（HBAT box + 水平翻转，10 seeds 的 mean ± std）

| arm | F1 | AP50 | precision / recall |
|---|---|---|---|
| baseline | 0.475 ± 0.028 | 0.360 | 0.511 / 0.450 |
| +attention | 0.479 ± 0.021 | 0.364 | 0.525 / 0.445 |

**paired 差值**（+attention − baseline）：**ΔF1 = +0.004，95% CI [−0.021, +0.028]，Wilcoxon p = 0.56，10 个 seed 中 7 个为正、3 个为负**。AP50 的差值同样在 0 附近。official box 那一组也一样（+0.009 [−0.012, +0.031]）。

每个 seed 的 ΔF1：+0.028, +0.003, −0.035, +0.019, +0.080, +0.007, −0.078, +0.003, +0.026, −0.013。

**学到什么**

1. **提升不存在。** 95% CI 的半宽是 0.025，意思是：如果真实提升超过 0.03，这次实验就应该能测出来。
2. Round 3 的 +0.044 是随机波动：在 fold 0 上用正确对齐的 10 个 seed 重测，ΔF1 是 −0.005。
3. 论文 Table IV 的 0.67 → 0.73 无法复现。问题不在整体分数（那是 data leakage 和 box granularity mismatch 造成的），而在**提升本身**。
4. 两个正面的发现：加 attention 不会让检测变差；attention map 确实落在 band 上（和 band 的 IoU 为 0.38–0.46）。

完整数字见 `runs_cv/cv_summary.md`。

---

## Round 5：诊断为什么没有提升

**日期**：11-15

不重新训练，只分析 Round 4 的预测结果。按顺序问了四个问题。

### 问题 1：是不是图太小，band 经过卷积之后变模糊了？

量了一下：原图 224×224，band 约 9 像素宽。attention map 是 56×56，band 只剩约 2 格宽，57% 的格子一半是 band、一半是背景。attention map 送进检测头之前还要 pooling 到 28×28，band 只剩约 1 格宽，弧形基本变成一团亮斑。

**答**：attention 本身学到了 band（IoU 0.38–0.46），所以不是"教不会"；但 band 的**形状**在送到检测头之前就丢了。检测头拿到的只是"这里有个东西"，而这一点检测任务自己就能学到。

### 问题 2：是不是代码有 bug？

| attention 要起作用需要满足 | 检查结果 | 满足？ |
|---|---|---|
| attention 学到了 band | IoU 0.38–0.46 | ✅ |
| 检测头用上了 attention | fusion 层里 attention 通道的权重和普通特征通道一样大 | ✅ |
| attention 提供了新信息 | 训练集上检测 loss 更低，但 test 没有变好；attention 质量好坏和 ΔF1 无关 | ❓ 大概率没有 |
| 新信息能解决瓶颈 | 见问题 3 | ❌ |

**答**：没有 bug。信息传进去了，但补充的不是网络缺的东西。

### 问题 3：瓶颈在哪？为什么 attention 帮不上？

两个 arm 的 recall 都只有 0.45，一半目标没检出来。把 FN（漏检）分了类（10 个 seed 累计 2220 个目标）：

| FN 类型 | 占比 | 说明 |
|---|---|---|
| B：位置对，但 score 低于 threshold | 36% | 找到了，但置信度不够 |
| **C：位置大致对，但框的大小不对**（和真值的 IoU 在 0.1–0.5） | **37–41%** | 找到了，框画错了 |
| D：基本没检测到 | 22–24% | 目标太弱或太小 |

关键事实：**两个 arm 漏掉的是同一批目标**（相关系数 0.89）。难检的目标有一半来自信号偏弱、被图像边缘截断、或多条双曲线重叠的图。

**答**：attention 的作用像荧光笔，标出"看这里"。但漏检时网络其实已经在看那里了，它缺的是"敢不敢确定"（B 类）和"框该画多大"（C 类）。attention 是从同一份特征算出来的，网络看不清的地方它也画不准，提供不了新证据。原设计针对的是 FP（不要看背景），而现在的问题是 FN。

### 问题 4：C 类（框画错）具体错在哪？

| 指标 | 数值 |
|---|---|
| 预测框宽 / 真值框宽（中位数） | 0.75–0.79 → **大多画小了** |
| 只把预测框的宽高换成真值，IoU 能超过 0.5 的比例 | 52–58% |
| 只把预测框的中心换成真值，IoU 能超过 0.5 的比例 | 36% |
| 预测框其实落在了相邻那条双曲线上的比例 | 35–42% |

**答**：主要是大小错，不是位置错。双曲线的两翼越往外越淡，网络还没到边就把框收住了。

### 这一轮的结论

> **band 的信息有用，但用错了地方。** 把它当"往哪看"的提示没用，因为网络本来就知道往哪看；网络真正缺的是"**目标有多大**"，而这正是 band 能提供的。

脚本：`diag_resolution.py`（问题 1）、`diag_misses.py`（问题 3）、`diag_box_size.py`（问题 4）。

---

## Round 6：用 band 修正框，验证诊断

**日期**：11-15

**做了什么**：不重新训练，拿 Round 4 的 50 个 +attention 模型，在 inference 时多加一步 post-processing：把 attention map 二值化得到 band，沿着 band 把预测框往外扩，直到包住整条弧（**只扩大、不缩小**）。相邻双曲线的 band 按"离哪个检测中心更近"分开。参数只在 validation set 上选。另外用**人工标注的真实 band** 走同一流程，作为 upper bound。

```
真值框（绿）          网络预测框（红）        沿 band 扩大后（蓝）
┌─────────┐           ┌───┐                 ┌─────────┐
│  /   \  │           │/ \│  ← 只框住顶部    │  /   \  │
│ /     \ │           └───┘                 │ /     \ │
└─────────┘                                 └─────────┘
```

**结果**（5 个 fold 合并、10 seeds）

| | F1 | recall | AP50 |
|---|---|---|---|
| baseline | 0.475 | 0.450 | 0.360 |
| +attention（Round 4 原样） | 0.479 | 0.445 | 0.364 |
| **+attention + 沿 band 扩框** | **0.541** | 0.483 | **0.470** |
| 用真实 band 扩框（upper bound） | 0.587 | 0.545 | 0.529 |

paired 比较：+attention → 扩框 **ΔF1 = +0.063 [+0.054, +0.071]，p = 0.002，10 个 seed 全部为正**；baseline → 扩框 +0.067，9 正 1 负。

**三个发现**

1. **"只扩不缩"的 20 组参数全部变好，"直接用 band 外接框替换"的 20 组全部变差。** 意思是预测的 band 画不全（两翼太淡的地方没涂到），但只要 band 伸出了原来的框，真值框几乎总是在那里。
2. FN 中的 C 类（框画错）从 500 个降到 363 个，B 类和 D 类不变。改善的正是诊断指出的那一类。
3. 离 upper bound 还差 0.046，差距来自"预测的 band 画得不够全"。

**定位（2026-11-15 决定）**：这只是一个 **post-processing** 步骤（性质类似 NMS），**不作为论文的主张，论文不因此改写**。它的价值在于证明：attention 分支学到了框缺少的尺寸信息，只是这个信息没有进入 box regression。正确的做法是把尺寸信息做进网络里，而不是靠 post-processing。

两点备注：用全局统一参数的那一组是在看过参数敏感性表之后才加的（参数只在 validation set 上选，而且 20 组参数全部变好，结论不依赖这一组）；baseline 没有 band 可用，所以"baseline → 扩框"比较的是"论文方法 + post-processing"和"什么都不加"。

脚本：`refine_with_band.py`、`vis_refine.py`；结果 `runs_cv/refine/summary.md`；对比图 `runs_cv/refine/vis/seed0/`。

---

## Round 7–10 之前：三个假设的分析

**日期**：11-15

Round 6 之后提出了三个问题：图片尺寸是否适合当前 backbone、λ 的取值是否有影响、数据量是不是硬伤。先按代码算了一遍，再决定跑什么。

**1. 图片尺寸与 backbone：确实不匹配，但病根是 receptive field，改图片尺寸解决不了。** 三个 DownBlock（每块两个 3×3 卷积 + 一次 pooling）的理论 receptive field：

| 位置 | stride | receptive field | 对比 |
|---|---|---|---|
| f3（attention head 的输入） | 4 | 32 px，attention head 再加一个 3×3 后约 40 px | 一条双曲线的框约 65 px 宽，attention head 只看得到六成 |
| 检测头的输入 | 8 | 约 68 px，head 内的 3×3 后约 84 px | 刚好盖住一条双曲线，几乎没有上下文 |
| band | — | 约 9 px 厚 → 56×56 上 2 格，28×28 上 1 格 | 形状在 fusion 之前就丢了 |

attention head 看不全一条双曲线，所以它"不知道两翼在哪里结束"，这就是 Round 6 里"预测的 band 画不全"的来源。把图放大，band 变粗了，但 receptive field 不够的问题加倍；把图缩小，band 就只剩几个像素被 stride 4 吃掉。**调输入尺寸是零和的**，该改的是网络：更大的 receptive field，以及在更高分辨率上做 fusion。另外数据集 2 的原图本来就是 224×224，放大不会带来新信息；分辨率假设只能在数据集 1（原图 640）上真正验证。

**2. λ：值得扫一遍，但不是主因。** 训练末期检测 loss 约 0.15、attention loss 约 0.39，乘上 λ = 0.7 后 attention 项是检测项的 1.8 倍。但 Round 5 已经证明 attention 学到了 band、信息也传进了检测头，瓶颈是"传进去的不是检测头缺的那种信息"。λ 只决定学多少 band，不改变 band 的信息用在哪里。

**3. 数据量：对整体精度是硬伤，对"证明 attention 有效"恰恰相反。** 每个 fold 约 91 张训练图、约 150 个训练目标，2M 参数从零训练，训练 loss 0.15 而 test F1 0.48，overfitting 明显。但辅助监督在小数据上理应起正则化作用，小数据下没有提升，说明现在的设计没有把 band 用成正则项，不能拿数据量当理由。数据量真正限制的是统计功效：10 个 seed 的 95% CI 半宽 0.025，只能测出 ≥ 0.03 的效应，而小于 0.05 的提升发出去也没有说服力。

**结论**：接下来要证明的不是"加 attention 有用"，而是"**用对地方的 band 监督有用**"。先用 upper bound 实验区分"band 预测得不够好"还是"fusion 本身用不上 band"（Round 7），再把 band 的尺寸信息直接接到 box regression（Round 8）；数据量曲线、λ、gate 作为补充（Round 9、10）。

**数据集 1 复跑（原优先级 1）暂时阻塞**：`data/primary_roboflow/hbat/annotations.json` 的 band 标注不完整，`ds1_f0_s100/manifest.json` 显示 60 多个训练项没有 band（数据是用 `--missing-mask skip` 生成的）。两个 arm 都丢掉这些图会让比较失真，要等标注补齐再跑。

**代码**：`train_basic.py` 新增 `--mode oracle`、`--fuse extent`、`--det-stride 4`、`--train-frac`。旧的 none / abs 两个 arm 在新脚本下逐位复现（3 个 epoch 的 train log 与预测完全一致，见 `runs_smoke/ds2/regress_*`），所以 Round 4 的结果可以直接拿来配对。

---

## Round 7：upper bound 诊断，真实 band 直接做 fusion

**日期**：11-15

**做了什么**：新模式 `--mode oracle`：去掉 attention head，把**人工标注的真实 band**（56×56 的 mask）直接当 attention map 送进原来的 concat fusion 层，训练和 inference 都用真实 band。其余与 Round 4 完全一致（paired initialization、5-fold × 10 seeds、HBAT box + 水平翻转）。实验名 `hbatbox_aug_oracle`，50 个 run。

**为什么**：这是"band 完全正确时，现在这条 fusion 通路能换来多少 F1"的 upper bound。
- 如果 F1 大涨：fusion 通路能利用 band，瓶颈是 band 预测质量，下一步该提升 band 的完整度。
- 如果 F1 基本不动：1 通道 pooling 后 concat 这条通路本身就传不了尺寸信息，再在它上面调 λ 没有意义。

**结果（初步，36/50 run；fold 0–2 各 10 seeds，fold 3 有 6 seeds，fold 4 没跑；11-15 停机换新机器）**

| fold | seeds | baseline | +attention | oracle（真实 band） | Δ oracle − baseline（paired） | 正/负 |
|---|---|---|---|---|---|---|
| 0 | 10 | 0.514 | 0.509 | 0.538 | +0.024 | 6/4 |
| 1 | 10 | 0.461 | 0.472 | 0.455 | −0.006 | 5/5 |
| 2 | 10 | 0.499 | 0.523 | 0.511 | +0.012 | 6/4 |
| 3 | 6 | 0.368 | 0.356 | 0.339 | −0.029 | 1/5 |
| 合计 | 36 | | | | **+0.003**（std 0.092） | **18/18** |

**初步判断**：**把人工标注的真实 band 送进现在的 fusion 通路，F1 也不涨。** 这是第二种情形：1 通道 pooling 后 concat 这条通路本身传不了尺寸信息，再调 λ、再提升 band 预测质量都没有意义。对比 Round 6：同一份真实 band 用在 post-processing 上能到 0.587。信息在，通路不行。这直接支持 Round 8 的方向（换 fusion 方式 / 在 stride 4 做 fusion）。

正式结论要等 fold 4 的 10 个 run 和 fold 3 剩下的 4 个补齐后用 `aggregate_cv.py` 做 5-fold 合并的 paired 统计（现在每个 seed 都缺 fold 4，合并统计算不出来）。

---

## Round 8：把尺寸信息做进网络

**日期**：11-15

**做了什么**：两条路，都保持 paired initialization 协议。

a) **extent features fusion**（`--fuse extent`）：fusion 的输入不再是 1 通道平均 pooling，而是从 56×56 的 attention map 算 4 个通道：原图、水平 17 格 box filter（约 68 px）、竖直 17 格、水平 33 格（约 132 px），再 pooling 到 28×28 和检测特征 concat。box filter 没有参数，只是把"band 在这附近延伸了多宽"显式算出来给 wh head 用。实验 `hbatbox_aug_abs_extent`，和 Round 4 的 baseline / +attention 配对。另跑 `hbatbox_aug_oracle_extent`（真实 band + extent features）作为这条路的 upper bound。

b) **检测头搬到 stride 4**（`--det-stride 4`）：bottleneck 特征最近邻上采样回 56×56，和 f3 concat 后过一个 ConvBlock，再出三个 head；attention 在 56×56 上直接 concat，不再 pooling。heatmap 的 sigma 从 6 格改成 12 格、NMS 核从 3 改成 5，物理尺度不变。两个 arm 都换这个 decoder，所以配对比较的是 `hbatbox_aug_none_s4 → hbatbox_aug_abs_s4`（新 baseline 3.39 M 参数，比原来多 1.5 M）。

**为什么**：Round 5、6 说明 band 能提供"目标有多大"，但 1 格宽的 attention map 到检测头时只剩"这里有东西"。a 直接把尺寸算成特征；b 让 fusion 发生在 band 还有 2 格厚的时候。

**结果**：跑中。

---

## Round 9：数据量曲线

**日期**：11-15

**做了什么**：`--train-frac 0.5 / 0.25`：训练子集由 fold 决定的固定 seed 抽取，两个 arm、所有 seed 看到完全相同的子集；validation set 和 test set 不变。四个实验各 50 个 run：`hbatbox_aug_{none,abs}_frac{50,25}`。

**为什么**：直接回答"数据量是不是硬伤"。如果 ΔF1 随数据减少而变大，可以主张"band 监督是小样本下的有效先验"；如果不变，就彻底排除"数据量掩盖了效应"的解释。

**结果**：待跑。

---

## Round 10：λ 扫描 + gate fusion

**日期**：11-15

**做了什么**：`hbatbox_aug_abs_lam0.1 / 0.3 / 1.5`（默认 λ = 0.7）和 `hbatbox_aug_abs_gate`（fusion 改成 feat × (1 + attention)，早期探索里对弱检测器有 +0.03 的那种），各 50 个 run，都和 `hbatbox_aug_none` 配对。

**为什么**：审稿人一定会问；排除"attention loss 权重太大"和"fusion 方式太弱"两种解释。

**结果**：待跑。

---

## 在新机器上继续

**2026-11-15 状态**：本机停跑，换到新 GPU 机器。仓库历史已重置为单个提交并推到 GitHub（旧的 12 GB 历史删除）。每个 run 的轻量结果（`preds_*.json`、`metrics*.json`、`run_info.json`、`train_log.csv`、控制台日志）已入库；权重 `*.pth`、可视化图和 `*.npz` 没有入库。

```bash
git clone https://github.com/youryugi/shuangquxian.git
cd shuangquxian
pip install -r requirements.txt        # torch 按新机器的 CUDA 版本装，见文件内注释

# 1. 重建 prepare_data.py 的数据目录（datasets/ 不入库，约 1 分钟）
cd experiments/revision/detector_comparison
for f in 0 1 2 3 4; do
  python prepare_data.py --dataset ds2 --fold $f --subset 100 --box-source hbat
  python prepare_data.py --dataset ds2 --fold $f --subset 100 --box-source official
done

# 2. 续跑。run_seeds.py 会跳过已有 metrics.json 的 run
cd ../basic_ds2
python run_seeds.py --exps hbatbox_aug_oracle --jobs 8                              # Round 7 补齐 14 个 run
python run_seeds.py --exps hbatbox_aug_abs_extent hbatbox_aug_oracle_extent hbatbox_aug_none_s4 hbatbox_aug_abs_s4 --jobs 8   # Round 8
python run_seeds.py --exps hbatbox_aug_none_frac50 hbatbox_aug_abs_frac50 hbatbox_aug_none_frac25 hbatbox_aug_abs_frac25 --jobs 8   # Round 9
python run_seeds.py --exps hbatbox_aug_abs_lam0.1 hbatbox_aug_abs_lam0.3 hbatbox_aug_abs_lam1.5 hbatbox_aug_abs_gate --jobs 8        # Round 10

# 3. 汇总（含所有配对）
python aggregate_cv.py
```

两点注意：
- **换 GPU 后结果不再和本机逐位一致**（kernel 不同），但 paired initialization 由 CPU 的 seed 决定，跨机器相同，配对比较仍然成立。更干净的做法是把 Round 4 的两个主 arm（`hbatbox_aug_none`、`hbatbox_aug_abs`）在新机器上也重跑一遍：先把 `runs_cv/ds2/hbatbox_aug_none` 和 `hbatbox_aug_abs` 改名备份，再跑，这样所有配对都在同一张卡上。5090 上 100 个 run 约 20–30 分钟。
- `--jobs` 在 5090 上可开 8–12，`--workers 1`；本机 5060 Ti 是 `--jobs 4 --workers 2`。
- Round 6 的 `refine_with_band.py infer` 需要 `final.pth`，新机器上要先重跑 `hbatbox_aug_abs` 才有权重；`refine` 阶段只读 npz，也要重新 infer。

## 下一步

| 优先级 | 做什么 | 为了回答什么 | 成本 |
|---|---|---|---|
| **1** | **数据集 1 用 Round 4 的方法重跑** | 论文主张能不能保住 | 100 个 run，约 1 小时 |
| 2 | 把尺寸信息做进网络：用 band 的外接框监督宽高 regression，或把 loss 改成 GIoU | 把 +0.063 变成网络自己的能力 | 改几行代码 + 100 个 run |
| 3 | 让 band 画得更全：attention head 移到 112×112、监督目标不做二值化 | 逼近 upper bound 0.587 | 每组 100 个 run |
| 4 | 扫描 attention loss 权重 λ = 0.1 / 0.3 / 1.5 | 排除"attention loss 权重太大"的可能 | 每个 λ 100 个 run |
| 5 | fusion 方式从 concat 换成 gate | 排除"fusion 方式太弱"的可能 | 100 个 run |
| 6 | upper bound 诊断：把真实 band 直接送进 fusion | 区分"信息本身没用"和"信息没传到" | 100 个 run |

先做 1；2、3 可以同时准备。4–6 等前面的结果出来再决定。

**写作**：如果数据集 1 也没有提升，论文主张从"提升 F1"缩小为"attention 可解释、不降低精度、能为 band 级输出提供中间表示"，主表如实报告 +0.004 [−0.021, +0.028]。数据描述里要写清楚：official box 是 scene-level、HBAT box 是 instance-level，以及两者的匹配率；test set 27 张图中有 12 张存在一个框包含多条双曲线的情况。

---

## 附录 A：实验设置（正式协议）

Round 4 起所有正式数字都按这套设置。

| 项 | 设置 |
|---|---|
| 数据 | 数据集 2 共 131 张原图（224×224），全部使用；按来源分组做 5-fold CV |
| 检测目标 | HBAT box（主）；official box（对照） |
| attention 监督目标 | HBAT 导出的 band mask PNG，224→56 用面积插值缩小，再以 0.5 二值化 |
| 网络 | 三层 backbone（base_ch=32）+ CenterNet heatmap head；attention head 接在 f3（56×56）上，pooling 到 28×28 后用 concat 做 fusion |
| 训练 | Adam，lr 5e-4，batch size 8，120 epochs，bf16，水平翻转增强，用最后一个 epoch 的权重；attention loss = BCE + Dice，权重 λ = 0.7 |
| 初始化 | paired initialization：检测部分两个 arm 逐参数相同；attention 分支用单独 seed；数据顺序固定 |
| 评估 | 5 个 fold 的 validation 预测合并选 score threshold → 5 个 fold 的 test 预测合并算 F1、AP50（IoU 0.5，greedy matching） |
| 统计 | 按 seed 配对，n = 10：Δ 的均值、bootstrap 95% CI、Wilcoxon signed-rank p、正/零/负个数 |

**和论文原脚本的差别**：图像（247 张增强图 → 131 张原图）、数据切分（随机 50/50 → 按来源分组 5-fold）、检测框（自己标的框 → HBAT box）、mask（自己画的抛物线 → HBAT 双曲线 band）、初始化（未对齐 → paired initialization）、评估（固定 threshold 0.375 → 在 validation set 上选 threshold）。网络结构、loss、超参数没改。原脚本：[centernet_band_attention.py](../../submission/additional_centernet/centernet_band_attention.py)。

## 附录 B：文件与数据位置

**代码**

| 文件 | 用途 | Round |
|---|---|---|
| `train_basic.py` | 训练 + 导出预测 + 评估。`--mode none` 为 baseline，`--mode abs` 为 +attention，`--mode oracle` 为真实 band fusion；`--fuse concat/gate/extent`；`--det-stride 4/8`；`--train-frac` | 2–4, 7–10 |
| `run_seeds.py` | 并行跑 5 folds × 10 seeds，已完成的自动跳过；Round 7–10 的实验名都登记在 `EXPS` 里 | 4, 7–10 |
| `aggregate_cv.py` | 合并 5 个 fold + paired 统计 → `runs_cv/cv_summary.md` | 4 |
| `check_data.py` | 核对数据目录和 mask | 2 |
| `diag_resolution.py` / `diag_misses.py` / `diag_box_size.py` | 诊断脚本，只读预测结果 | 5 |
| `refine_with_band.py` / `vis_refine.py` | 用 band 修正框 + 可视化 | 6 |

**结果**

| 位置 | 内容 | 能否引用 |
|---|---|---|
| `runs_cv/ds2/<exp>/f<fold>_seed<k>/` | Round 4 的 200 个 run 和 Round 7–10 的各实验：`final.pth`、`preds_*.json`、`metrics*.json`、`run_info.json`、`train_log.csv` | ✅ 正式 |
| `runs_cv/cv_summary.md/.json` | Round 4 汇总 | ✅ 正式 |
| `runs_cv/refine/` | Round 6：attention map（npz）、修正后的预测、`summary.md`、`vis/` | ✅（仅作 post-processing 证据） |
| `runs/` | Round 2、3（single fold、初始化未对齐） | ❌ 只作对照 |
| `runs_legacy_init_1114/` | Round 1 | ❌ 作废 |
| `runs_smoke/` | smoke test（只验证代码能跑通） | ❌ |

**数据**（`../detector_comparison/datasets/`）：`ds2_f<0-4>_s100_hbatbox/`（HBAT box，主）、`ds2_f<0-4>_s100/`（official box）。每个目录里两套框的 JSON 都有，可以交叉评估。标注的唯一来源是 `data/additional_utilities/hbat/annotations.json`。

**细节备查**：训练图 246 里唯一的目标没有双曲线参数（unparam），它的 mask 全黑，但仍参与训练；全部 6 个 unparam 目标在 HBAT box 模式下改用 official box。

## 附录 C：潜在问题清单

1. 数据集 1 还没用严格协议跑过（最大的空白）。
2. "不降低检测精度"是一个很弱的主张。
3. 整体精度低（F1 0.48）：模型从零训练，训练集只有约 91 张图。
4. λ = 0.7 时 attention loss 是检测 loss 的 2.3 倍，没有扫过其他取值。
5. fusion 只试了 concat 一种。
6. 56×56 的 attention map 上 band 只有 2 格宽（Round 5 问题 1）。
7. validation set 合计只有约 70 个目标，选参数有 overfitting 风险（Round 6 中每个 seed 单独选参比全局统一选参差 0.023）。
8. 用的是最后一个 epoch 的权重，训练末尾 loss 还在波动。
9. official box 和 HBAT box 的 granularity 不同，论文里要写清楚。
10. 本目录还没提交到 git；`runs_cv/` 有 2 GB，提交前要确认 .gitignore。
