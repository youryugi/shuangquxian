# 实验 1：基础实验（数据集 2，新标注）：band 监督到底有没有用

**问题**：论文 Table IV 说"用 hyperbola band mask 监督 attention，检测 F1 从 0.67 升到 0.73"。换上重新标注的数据、用严格的实验方法重跑，这个提升还在吗？如果不在，band 标注还有什么价值？

**答案（截至 2026-11-22，Round 22；新口径：预测框裁到图内，翻转增强 bug 已修）**

1. **论文的 +0.06 不在。** 严格协议下 +attention（band fusion）是 +0.023（8/2），而且 fusion 这一步 band 和填实的框没有区别（−0.008，n.s.）。原来的提升来自 data leakage 和没对齐的初始化。
2. **把 band 用来定框，主张成立。** 沿预测的 band 拟合双曲线定框（hyperbola fit），F1 0.472 → **0.585**（+0.112，10/10 seeds）。band 形状的贡献由 box-mask control（同结构，只把监督换成填实的框）分离：
   - 同一后处理：box-mask + fit 0.499，band 高 **+0.086**（10/10）；sweep 里逐组比较相同的参数，band 赢 145/180（U-shaped decoder 180/180）。
   - 对 box-mask 臂最好的后处理（Voronoi + NMS，0.541）：band 高 **+0.044**（10/10）。这是最保守的数字。
   - band 的作用在"按形状定框"：Voronoi 定框只用外接范围，band 对 box-mask 不稳健；fit 用到形状，band 才拉开差距。
3. **在每个训练比例下都成立（Round 19，固定测试集 + 训练数据 25 / 50 / 75 / 100%）。** band + fit 在每个比例都赢过 baseline（+0.170 / +0.138 / +0.098 / +0.112，全部 10/10）和该比例下 box-mask 臂最好的后处理（+0.046 / +0.055 / +0.055 / +0.044，8/2 到 10/0）。数据越少，相对 baseline 的提升越大：用一半训练数据的 band + fit（0.528）超过用全部数据的 baseline（0.472）。
4. **剩下的空间在 grouped targets（相邻双曲线成簇）。** 这里 band 对 box-mask 的优势最大（recall +0.105，10/10），但对 baseline 还没有提升（+0.011，n.s.）；换成真实 band 能多拿 +0.141。瓶颈是重叠目标中约三成有一条 band 整条没画出来，以及翼尖画不全。
5. **Round 20 试的四种具体补法都没有让 band + fit 超过 0.585（只否定这四种做法，不否定方向）。** 补翼尖（对称补翼、沿曲线续走）在 union 模式下被检测框覆盖；曲线归属能把 grouped recall 提到 0.489，但 alone 上误报增加、F1 不变；attention 损失按目标均衡 + 翼尖加权只把覆盖率从顶点挪到翼尖，整条漏画没有减少。grouped 的具体瓶颈定位到了 fit 之后的配对和去重：没画出 band 的目标只能拟合到邻居的曲线上，变成重复被删掉；曲线归属说明这里有可拿的 recall，但需要更好的区分规则。
6. **Round 22 换了四种配对规则，都在 +0.02 以内。** 降分保留、几何判重主要涨 precision；全图分配只对 box-mask 对照有用。"解释掉再拟合"（抹掉邻居的曲线、低阈值从原框再拟合）是第一个只对 band 成立、并直接涨 grouped 召回率的规则：conv 0.585 → 0.594（8/10，grouped 0.439 → 0.459），unet 0.585 → 0.597（8/10，grouped 0.426 → 0.462），box-mask 上无效。但在全部做法里统一按验证集选时选不中它，单独报告要说明。
7. **对标注参数稳健（Round 21，已单独整理为[实验 2](../exp2_annotation_sensitivity/README.md)，回应 R1-4、R2-4）。** 只换 band 监督掩码、每档训练 50 个 run：厚度 ×0.5 到 ×1.25 和"全部目标统一厚度"与标准标注差 −0.002 到 −0.006（n.s.），对 box-mask 最好的后处理仍 +0.037 到 +0.042；在实测的重复标注误差下（同一人标两次，训练掩码 IoU 只有 0.58）掉 −0.009（n.s.），仍对 baseline +0.103、对 box-mask 最好的后处理 +0.034。跨度最敏感（±25% 掉 −0.047）；厚度太细（×0.25）或太粗（×1.5、×2）也会下降，粗到一定程度就退化成 box-mask。15 个档位 band + fit 全部赢 baseline（最差 +0.049，10/10）。
8. **Round 23（2026-11-23）：只改去重 / 定框，fit 和 grouped 两头保不住。** curve-aware NMS、拥挤处不做 fit、soft-NMS 及其组合，在全网格上都是 grouped 和 alone 此消彼长（没有一组同时 grouped ≥ 0.47、alone ≥ 0.61）。被丢的 grouped 目标恰好是 band 没画出来的那一条（多为 medium / weak），框的几何分不清它和重复检测；要两头都保住需要不依赖这条 band 的新证据，目前最近的是 Round 22 的"解释掉再拟合"。soft-NMS 单独 +0.013（8/2），对 box-mask 也 +0.021，不涉及 grouped。
9. **Round 24（2026-11-23）：按 visibility 加权 attention 损失（weak ×4、medium ×2）没有让 weak band 画出来**（weak 覆盖率 0.110 → 0.104，重叠对整条漏画 7.3 → 7.6）；band + fit 按协议 −0.040。加上 Round 20，两种损失加权都失败：在 131 张图上 weak band 学不出来，是数据问题。推理时 band 在 weak 候选上也没有信号（AUC 0.38），grouped 撞曲线的检测 85% 是重复、band 分不开（AUC 0.57）。

**主线**（11-21 决定）：论文证明的是 band attention 的作用，对照是 box-mask control（必要时加 scale control）。stage 2（Round 14，不用 band 的第二阶段框回归）和论文对象无关，不作为要超过的对照，见[附录 C](#附录-c潜在问题清单)。

**下一步**：① 用 Round 19 的表替换论文 Table II；② 数据集 1 用同一协议重跑（等 band 标注补齐）；③ grouped：在 Round 22 的"解释掉再拟合"上继续（多轮解释、分配和再拟合交替）；④ 给 band 更多数据。详见[下一步](#下一步)。

---

## 目录

- [术语表](#术语表)
- [结论是怎么一步步变的](#结论是怎么一步步变的)
- [一页总览](#一页总览)
- **逐轮详细记录 → [rounds.md](rounds.md)**
  - 阶段 A：复现
    - [Round 1：标注文件用错，作废](rounds.md#round-1标注文件用错作废)
    - [Round 2：新标注 + official box，single fold](rounds.md#round-2新标注--official-boxsingle-fold)
    - [Round 3：HBAT box + 水平翻转，3 seeds](rounds.md#round-3hbat-box--水平翻转3-seeds)
    - [Round 4：严格协议（正式结果）](rounds.md#round-4严格协议正式结果)
  - 阶段 B：找原因
    - [Round 5：诊断为什么没有提升](rounds.md#round-5诊断为什么没有提升)
    - [Round 6：用 band 修正框，验证诊断](rounds.md#round-6用-band-修正框验证诊断)
    - [Round 7–10 之前：三个假设](rounds.md#round-710-之前三个假设)
    - [Round 7：真实 band 直接做 fusion（upper bound）](rounds.md#round-7真实-band-直接做-fusionupper-bound)
    - [Round 8：把尺寸信息做进网络](rounds.md#round-8把尺寸信息做进网络)
    - [Round 9：数据量曲线](rounds.md#round-9数据量曲线)
    - [Round 10：λ 扫描 + gate fusion](rounds.md#round-10λ-扫描--gate-fusion)
  - 阶段 C：换一种用法
    - [Round 11：分层分析，attention 帮了哪类目标](rounds.md#round-11分层分析attention-帮了哪类目标)
    - [Round 12：box refinement 做进推理](rounds.md#round-12box-refinement-做进推理)
    - [Round 13：box-mask control，有多少是 band 形状的功劳](rounds.md#round-13box-mask-control有多少是-band-形状的功劳)
  - 阶段 D：找 band 不可替代的地方
    - [Round 14：stage 2，按每条双曲线做 attention](rounds.md#round-14stage-2按每条双曲线做-attention)
    - [Round 15：hyperbola fit，用形状分开 grouped targets](rounds.md#round-15hyperbola-fit用形状分开-grouped-targets)
    - [Round 16：U-shaped decoder，把 band 预测好](rounds.md#round-16u-shaped-decoder把-band-预测好)
  - 阶段 E：从输出端用 band 标注（新口径：预测框裁到图内）
    - [Round 17：参数化检测头，网络直接输出双曲线参数](rounds.md#round-17参数化检测头网络直接输出双曲线参数)（含翻转增强 bug）
  - **阶段 F：修好翻转、新口径下的正式结果（论文引用这一轮）**
    - [Round 18：修好翻转后重跑，band + hyperbola fit 的正式结论](rounds.md#round-18修好翻转后重跑band--hyperbola-fit-的正式结论)
    - [Round 19：固定测试集 + 递增训练数据（数据效率，回应 R2-6）](rounds.md#round-19固定测试集--递增训练数据数据效率回应-r2-6)
    - [Round 21：标注敏感性（厚度、跨度、顶点、slope，回应 R1-4、R2-4）](../exp2_annotation_sensitivity/README.md)
- [下一步](#下一步)
- [附录 A：实验设置（正式协议）](#附录-a实验设置正式协议)
- [附录 B：文件与数据位置](#附录-b文件与数据位置)
- [附录 C：潜在问题清单](#附录-c潜在问题清单)
- [附录 D：在新机器上复现 / 续跑](#附录-d在新机器上复现--续跑)

---

## 术语表

专业词统一用英文，含义如下。

**实验设计**

| 术语 | 含义 |
|---|---|
| **baseline** | 只有检测头的模型，不加 attention |
| **+attention** | 检测头 + attention 分支，attention 用 band mask 监督（论文方法） |
| **arm** | 对比实验中的一组配置，例如 baseline arm、+attention arm |
| **fold / 5-fold CV** | 131 张图按来源分成 5 份，轮流用 1 份做测试。**single fold** = 只跑 fold 0，测试集只有 27 张 |
| **seed** | 随机种子。同一配置换 seed 重跑，用来估计随机波动 |
| **paired initialization** | 两个 arm 用同一个 seed 时，检测部分的初始权重逐个参数相同，差值只反映 attention 本身的作用 |
| **paired Δ / 95% CI / 胜/平/负** | 按 seed 配对的差值（B − A）的均值、bootstrap 置信区间、10 个 seed 里 B 赢 / 平 / 输的次数 |
| **data leakage** | 测试图的"近亲"（同一张原图的增强版本）出现在训练集里 |
| **hardware replication** | 同一代码、同一 seed 在两块显卡（5060 Ti / 5090）上各跑一次；浮点 kernel 不同，结果会有小差别 |
| **upper bound / oracle** | 用人工标注的真值代替网络预测，得到的理论最好成绩 |
| **control（对照）** | 为了排除另一种解释而设计的 arm，只改一处 |
| **新口径（clip）** | 11-21 起：预测框先裁到图内 [0, 224]² 再和 GT 匹配，只比较图内的重叠（GT 本来就裁过） |
| **flipfix** | 11-21 修好水平翻转增强后重跑的 run（`hbatbox_flipfix_*`）。Round 3–16 的 `hbatbox_aug_*` 带翻转 bug（翻转样本的目标偏左 8 px），配对结论不受影响，但正式数字以 flipfix 为准 |

**数据与标注**

| 术语 | 含义 |
|---|---|
| **band / band mask** | 沿双曲线画出的带状区域，attention 的监督目标 |
| **official box** | 数据集自带的标注框，经常把相邻 2–4 条双曲线框成一个（scene-level） |
| **HBAT box** | 由 HBAT 标注的 band 几何算出的框，一条双曲线一个（instance-level）。按定义就是 band 的外接矩形 |
| **box granularity mismatch** | 标注框和模型对"一个目标"的定义不一致：一个大框套多条 vs 一条一个小框，IoU 不到 0.5，被判成 FP |
| **grouped target** | official box 里还套着别的双曲线的目标，即相邻双曲线成簇（222 个里 93 个）。反义 **isolated**：official box 只套这一条 |
| **narrow / medium / wide** | 按 GT 框宽度三分位：< 53 px / 53–80 px / ≥ 80 px |
| **visibility / completeness / distortion** | HBAT 的目标标签：清晰度（clear / medium / weak）、完整度（both 两翼 / single 单翼 / apex 只有顶部）、形状（fit 符合双曲线 / deviates 偏离） |

**指标**

| 术语 | 含义 |
|---|---|
| **TP / FP / FN** | 检对 / 误检 / 漏检 |
| **precision / recall / F1** | TP/(TP+FP) / TP/(TP+FN) / 两者调和平均。默认 IoU 0.5 匹配 |
| **F1@IoU0.7** | 匹配要求 IoU ≥ 0.7 的 F1，衡量定位是否精确 |
| **AP50** | IoU 0.5 下的 average precision，不依赖 score threshold |
| **score threshold** | 分数高于它才算检出。在 validation set 上选，用在 test set 上 |
| **near miss** | GT 的最佳 IoU 在 0.3–0.5：找到了但框画错（多数是画小了） |
| **band IoU** | 预测 band 与真实 band 的 IoU，衡量 band 画得好不好 |

**网络与方法**

| 术语 | 含义 |
|---|---|
| **stride** | 特征图相对输入图的缩小倍数。stride 8 = 224 → 28×28；stride 4 = 56×56 |
| **receptive field** | 某层一个输出格子能"看到"的输入范围（像素） |
| **fusion** | 把 attention map 和检测特征合并的一步。**concat** = 拼成多一个通道；**gate** = 特征 × (1 + attention)；**extent** = 额外算 4 个"band 延伸多宽"的通道（Round 8） |
| **λ (lam_att)** | 总 loss = 检测 loss + λ × attention loss，默认 0.7 |
| **dense head** | 逐像素输出的头。attention head 就是一个 dense head |
| **box regression** | 网络预测框位置和宽高的部分 |
| **post-processing** | 网络输出之后、不需要重新训练的步骤（例如 NMS） |
| **box refinement（band-to-box）** | 推理时用 attention map 里的 band 修正检测框的范围（Round 6、12） |
| **union / replace / blend** | box refinement 的三种合并方式：新框 = 原框 ∪ band 外接框（只扩不缩）/ 直接用 band 外接框 / 两者坐标平均 |
| **Voronoi split** | 把 band 像素分给离它最近的检测中心，用来切开相邻双曲线（Round 6、12） |
| **hyperbola fit** | 沿 band 给每个检测拟合一条双曲线（顶点 + slope），取 band 支持的那一段作框，代替 Voronoi split（Round 15） |
| **scale control** | 对照：把 baseline 的框绕中心统一放大 k 倍（k 在验证集上选） |
| **box-mask control** | 对照：attention head 的监督换成**填实的 HBAT 框**。只需框标注，没有 band 形状（Round 13） |
| **stage 2** | 第二阶段：每个候选框裁一个窗口、缩放到 96×96，在窗口上再回归一次框（Round 14）。五个 arm：**plain**（只回归框）/ **unsup**（有 attention 分支但不监督）/ **boxmask** / **band** / **oracle**（直接给真实 band） |
| **U-shaped decoder（unet head）** | Round 16 的 attention head：逐级上采样、接 skip，在 224×224 上输出 |

---

## 结论是怎么一步步变的

| 阶段 | Rounds | 要回答的问题 | 得到的答案 | 引出的下一个问题 |
|---|---|---|---|---|
| **A 复现** | 1–4 | 论文的 +0.06 还在吗？ | 不在。严格协议下 +0.004 / +0.019，和换显卡同量级 | 为什么没有？ |
| **B 找原因** | 5–10 | 是分辨率、bug、λ、fusion 方式还是数据量的问题？ | 都不是主因。attention 学到了 band，但 fusion 通路只传过去一点（真实 band 走 fusion 也只 +0.03）；瓶颈是框画小了（near miss），而 fusion 解决不了这个 | band 直接用来定框行不行？ |
| **C 换用法** | 11–13 | box refinement 有效吗？是不是 band 形状的功劳？ | 有效（+0.072，10/10）。但只用框标注的 box-mask control 拿到其中 +0.046，band 形状特有 +0.025 | band 形状有没有别的东西替代不了的价值？ |
| **D 找不可替代处** | 14–16 | 按目标做 attention、按形状分开 grouped targets、把 band 预测好，能不能让 band 超过不用 band 的做法？ | 不用 band 的 stage 2 是 +0.159，超过所有 band 方案。band 形状唯一优势：band 准确时 grouped targets 的 recall +0.10。预测 band 的测试 IoU 停在 0.38–0.48，受数据量限制 | 更多带 band 标注的数据（数据集 1、合成数据）；论文主张怎么写 |
| **E 从输出端用 band** | 17 | 让网络直接输出双曲线参数，框由参数算出，行不行？ | 参数化 +0.08，但只用框反推参数的 pseudo 拿到几乎全部；真实参数只在单翼目标上有用。顺带发现翻转增强 bug | bug 修好后，前面的结论还成立吗？ |
| **F 正式结果** | 18–19 | 修好翻转、新口径下，band 定框相对 box-mask control 还有没有优势？换训练数据量还成立吗？ | **有**：band + fit 0.585，对 box-mask + fit +0.086、对 box-mask 最好的后处理 +0.044，都是 10/10，换参数也成立。优势来自按形状定框，不来自 fusion。训练数据 25–100% 下都成立，数据越少相对 baseline 提升越大 | grouped targets 对 baseline 还没提升；预测 band 漏画、翼尖画不全 → 形状外推 |
| **G 补 grouped** | 20 | 补翼尖、曲线归属、加权 attention 损失，能不能缩小预测 band 和真实 band 的差距？ | 这四种做法不能，0.585 不变。翼尖的宽度在 union 模式下已由检测框补上；grouped 丢在 fit 之后的去重（丢 11.1 个、新增 6.9 个），根子是整条漏画的 band；这一种损失加权画不出更多 band | 检测和曲线怎么配对、重复和不同目标怎么区分 |
| **G′ 配对规则** | 22 | 降分保留、几何判重、全图分配、解释掉再拟合，能不能拿回 fit 后丢的 grouped？ | 都在 +0.02 以内。只有"解释掉再拟合"只对 band 成立、涨 grouped 召回率（+0.020 / +0.036），F1 +0.010 / +0.012（8/10）；全图分配只帮 box-mask 对照 | 弱 band 本身没信号的目标怎么找回 |
| **H 标注敏感性** | 21 | 标注参数标偏了，band + fit 的提升还在吗？（R1-4、R2-4） | 在。厚度 ×0.5–×1.25、统一厚度、实测的重复标注误差下都不显著变化，仍对 box-mask 最好的后处理 +0.03–0.04；跨度 ±25% 掉 −0.047；厚度太细或太粗都下降。15 档全部赢 baseline | 两位标注员之间的差异（等第二位标注员） |

---

## 一页总览

两次 baseline 数字：Round 1–6 在 5060 Ti 上（0.475），Round 4 起在 5090 上重跑（0.473）。

| Round | 日期 | 做了什么 | 关键数字（F1） | 结论 |
|---|---|---|---|---|
| [1](rounds.md#round-1标注文件用错作废) | 11-14 下午 | 误用旧标注 | 0.42 vs 0.41 | **作废** |
| [2](rounds.md#round-2新标注--official-boxsingle-fold) | 11-14 晚 | 新标注、official box、single fold | 0.44 vs 0.44 | 没有提升；比论文低 0.3 |
| [3](rounds.md#round-3hbat-box--水平翻转3-seeds) | 11-14 晚 | HBAT box + 水平翻转，3 seeds | 0.52 vs 0.57 | 看似 +0.04，后证明是随机波动 |
| **[4](rounds.md#round-4严格协议正式结果)** | 11-14 深夜；11-15 5090 重跑 | **paired initialization + 5-fold + 10 seeds** | 0.473 → 0.492 | **Δ +0.004 / +0.019，和换显卡同量级。正式结果** |
| [5](rounds.md#round-5诊断为什么没有提升) | 11-15 | 诊断（只读预测） | — | 瓶颈是 FN；约 4 成 FN 是框画小了 |
| [6](rounds.md#round-6用-band-修正框验证诊断) | 11-15 | 用 band 把框扩大（post-processing） | 0.479 → 0.541 | band 知道框该多大（Round 12 正式化） |
| [7](rounds.md#round-7真实-band-直接做-fusionupper-bound) | 11-15 | 真实 band 直接喂 fusion | 0.473 → 0.505 | fusion 通路上限只有 +0.03 |
| [8](rounds.md#round-8把尺寸信息做进网络) | 11-15 | extent fusion；检测头 stride 4 | 0.484；0.434 | 都没有用 |
| [9](rounds.md#round-9数据量曲线) | 11-15 | 训练集 50% / 25% | Δ +0.005 / −0.021 | 数据越少 attention 越没用 |
| [10](rounds.md#round-10λ-扫描--gate-fusion) | 11-15 | λ 0.1 / 0.3 / 1.5；gate | Δ 全在 ±0.02 内 | λ、fusion 方式都不是原因 |
| [11](rounds.md#round-11分层分析attention-帮了哪类目标) | 11-15 晚 | 分层分析（只读预测） | — | fusion 的 +0.02 = 多检出约 5 个孤立目标，框大小没变 |
| **[12](rounds.md#round-12box-refinement-做进推理)** | 11-15 晚 | **box refinement 做进推理** | 0.473 → **0.545** | **+0.072，10/10**；其中 scale control 能拿 +0.029 |
| **[13](rounds.md#round-13box-mask-control有多少是-band-形状的功劳)** | 11-15 深夜 | **box-mask control** | 0.519 vs 0.545 | **band 形状特有 +0.025，7/3，p = 0.11** |
| **[14](rounds.md#round-14stage-2按每条双曲线做-attention)** | 11-16 凌晨 | **stage 2**（按每条双曲线回归） | plain 0.632 / band 0.644 | **只用框标注 +0.159**；band 不比 plain 好 |
| **[15](rounds.md#round-15hyperbola-fit用形状分开-grouped-targets)** | 11-16 凌晨 | **hyperbola fit** 代替 Voronoi split | 真实 band 0.658；预测 band 0.530 | band 准确时 grouped recall +0.10；预测 band 不够准 |
| **[16](rounds.md#round-16u-shaped-decoder把-band-预测好)** | 11-16 凌晨 | **U-shaped decoder** attention head | fit 0.549 | band IoU 训练 0.76 / 测试 0.42：**受数据量限制** |
| **[17](rounds.md#round-17参数化检测头网络直接输出双曲线参数)** | 11-21 | **参数化头**：热图峰 = 顶点，回归 slope / span / thickness，框闭式算出；修好翻转增强 bug（新口径） | 修好翻转 0.471；pseudo 0.545；param 0.550 | 参数化 +0.08（10/10），但只用框反推参数的 pseudo 拿到几乎全部；真实参数只在单翼目标上 recall +0.066；翻转 bug 修好后 F1@0.5 不变 |
| **[18](rounds.md#round-18修好翻转后重跑band--hyperbola-fit-的正式结论)** | 11-21 | **修好翻转后重跑 300 个第一阶段 run + 18 个后处理变体**（新口径） | baseline 0.472；box-mask + fit 0.499；**band + fit 0.585** | **对 box-mask + fit +0.086，对 box-mask 最好的后处理 +0.044，都是 10/10，换参数仍成立。正式结果**；Voronoi 上的 band 特有效应不稳健 |
| **[19](rounds.md#round-19固定测试集--递增训练数据数据效率回应-r2-6)** | 11-21 | **固定测试集 + 训练数据 25 / 50 / 75 / 100%**（450 个 run + 24 个后处理变体） | band + fit 0.443 / 0.528 / 0.564 / 0.585；baseline 0.273 / 0.390 / 0.465 / 0.472 | **每个比例都赢 baseline（10/10）和 box-mask 最好的后处理（+0.044–0.055）**；fusion 在任何比例都不显著 |
| **[21](../exp2_annotation_sensitivity/README.md)** | 11-22 | **标注敏感性**：15 套扰动后的 band 监督掩码（厚度、跨度、顶点、slope、重复标注差值），750 个 run | 标准 0.585；厚度 ×0.5–×1.25 0.578–0.579；统一厚度 0.583；重复标注 ×1 0.576；跨度 ±25% 0.537 | **厚度、实测标注误差下稳健；跨度最敏感；15 档全部赢 baseline** |

---

---

## 下一步

（2026-11-21 Round 18 之后。）

| 优先级 | 做什么 | 为了回答什么 | 成本 |
|---|---|---|---|
| **0** | **用 Round 19 的表替换论文 Table II**：固定测试集 + 递增训练集，列 baseline、+attention（fusion）、box-mask 最好的后处理、band + fit、真实 band + fit（上限），均值 ± std + 配对 Δ | 回应 R2-6、R2-8；原论文"各训练比例下都提升"的主张换成有对照支持的版本 | 只需写作（表、图的数字都在 `runs_cv/data_efficiency/`） |
| **2** | **数据集 1 用严格协议重跑**（等 band 标注补齐）：baseline / +attention / box-mask control / band + fit / box-mask + fit / box-mask + Voronoi + NMS，`--workers 1`，用修好的翻转 | 主结论能不能在另一份数据上复现。数据集 1 有 170 张原图（每折训练约 124 张） | 约 150 个 run；`infer_bandbox.py`、`diag_band_quality.py` 里写死的 `ds2` 路径要先参数化 |
| 3 | **给 band 更多数据**：两个数据集合并训练 attention head；或用合成双曲线（HBAT 参数模型随机采样、叠到无目标背景上）预训练 attention head 再微调 | 预测 band 的测试 IoU 停在 0.38–0.48，训练 0.76：受数据量限制 | 合成数据生成 + 每组 100 run |
| 4 | 形状输出的价值：hyperbola fit 给出每个目标的顶点和 slope（对应波速），和 HBAT 标注比误差；对照 = 从框反推 | band 在"检测 F1"之外的用途；Round 17 已看到单翼目标上的差别 | 只需评估代码 |
| 5 | 两次硬件复现怎么写进论文 | fusion 效应和换卡噪声（+0.013）同量级 | 只需写作 |

已完成：~~box refinement 做进推理~~（R12）；~~换掉 Voronoi split~~（R15）；~~λ、gate、oracle 上限~~（R7–10）；~~论文怎么面对 stage 2~~（11-21 决定：不作为主线对照）；~~修好翻转后重跑~~（R18）；~~固定测试集 + 递增训练集~~（R19）；~~形状外推、曲线归属、加权 attention 损失~~（R20，都没有提升）；~~标注敏感性~~（R21）。不再优先：hyperbola fit 接在 stage 2 之后（stage 2 不在主线上）；scale control 宽高分开（band 对 scale control 已是 +0.093，10/10）；YOLOv8s + band（结果为零，见 `../exp3_comparison/yolo_band/`）。

**写作要点**

- **主结论**：band 监督的 attention 画出双曲线，沿它拟合双曲线定框，F1 0.472 → 0.585（+0.112，10/10）。band 形状的贡献：同一后处理对 box-mask control +0.086，对 box-mask 臂最好的后处理 +0.044（10/10）；sweep 里逐组比较相同的参数，band 赢 145/180、180/180。定位更准：F1@IoU0.7 0.225 → 0.322。
- **数据效率**（Round 19，替换 Table II）：固定测试集、训练数据 25 / 50 / 75 / 100%，band + fit 在每个比例都赢 baseline（+0.10 到 +0.17，10/10）和 box-mask 最好的后处理（+0.044 到 +0.055）；数据越少相对 baseline 提升越大，一半数据的 band + fit 超过全部数据的 baseline。小数据下 band 对 box-mask 的优势依赖于在验证集上选好参数（25% 时全网格只赢 104/180，union 模式 60/60）。
- **fusion 不要作为卖点**：+0.023（8/2），band 和 box-mask 没区别（−0.008）。可写成"attention 本身不降低精度、可解释"，价值体现在后面的定框步骤。
- **Voronoi 定框不要用来论证 band 形状**：+0.061 换参数就不成立（24/60）。
- **grouped targets 要写清前提**：band 让 refinement 在 grouped 上不帮倒忙（box-mask + fit 在多目标图上 recall 0.442 → 0.329，band + fit 0.476），但对 baseline 还没有提升；真实 band 上 +0.141 是上限。
- **标注敏感性**（Round 21 = [实验 2](../exp2_annotation_sensitivity/README.md)，回应 R1-4、R2-4、R1-2）：画一张曲线（横轴扰动档位，纵轴 band + fit F1，参考线 baseline 和 box-mask 最好的后处理）。厚度 ×0.5–×1.25 是平台，统一厚度和逐个标一样（HBAT 可以省掉调厚度，也回应"固定宽度条带"）；跨度最敏感，论文写清楚要标到可见翼尖；实测重复标注误差下提升不变。误差量级来自同一人的重复标注，两位标注员之间的差异另测。
- 统计：paired initialization，5 折 × 10 seeds，所有参数在合并验证集上选，报告 Δ、bootstrap CI、Wilcoxon p、胜/平/负。
- 数据描述：official box 是 scene-level、HBAT box 是 instance-level 及两者匹配率；27 张测试图中 12 张一框多条。

---

## 附录 A：实验设置（正式协议）

Round 4 起所有正式数字都按这套设置。

| 项 | 设置 |
|---|---|
| 数据 | 数据集 2 共 131 张原图（224×224），全部使用；按来源分组 5-fold CV |
| 检测目标 | HBAT box（主）；official box（对照） |
| attention 监督目标 | HBAT 导出的 band mask PNG，224 → 56 面积插值后 0.5 二值化（Round 16 的 U-shaped decoder 直接用 224） |
| 网络 | 三层 backbone（base_ch = 32）+ CenterNet heatmap head；attention head 接 f3（56×56），pooling 到 28×28 后 concat fusion |
| 训练 | Adam，lr 5e-4，batch 8，120 epochs，bf16，水平翻转（Round 18 起为修好的几何翻转；Round 3–16 的 run 带 8 px bug，`--flip-legacy` 可复现），`--workers 1`（worker 数会改变翻转的随机数），用最后一个 epoch 的权重；attention loss = BCE + Dice，λ = 0.7 |
| 初始化 | paired initialization：检测部分两个 arm 逐参数相同；attention 分支单独 seed；数据顺序固定 |
| 评估 | 预测框先裁到图内（11-21 起）；5 个 fold 的 validation 预测合并选 threshold → 5 个 fold 的 test 预测合并算 F1、AP50（IoU 0.5，greedy matching） |
| 统计 | 按 seed 配对，n = 10：Δ 均值、bootstrap 95% CI、Wilcoxon signed-rank p、胜/平/负 |
| 推理步骤的参数 | box refinement、stage 2、hyperbola fit 的参数都在 10 seeds 的合并 validation F1 均值上全局选一组，测试集不参与 |

**和论文原脚本的差别**：图像（247 张增强图 → 131 张原图）、切分（随机 50/50 → 按来源分组 5-fold）、检测框（自标 → HBAT box）、mask（自画抛物线 → HBAT 双曲线 band）、初始化（未对齐 → paired）、评估（固定 threshold 0.375 → validation 选 threshold）。网络、loss、超参没改。原脚本：[centernet_band_attention.py](../../submission/additional_centernet/centernet_band_attention.py)。

## 附录 B：文件与数据位置

**代码**

| 文件 | 用途 | Round |
|---|---|---|
| `train_basic.py` | 训练 + 导出预测 + 评估。`--mode none/abs/oracle`；`--fuse concat/gate/extent`；`--det-stride 4/8`；`--train-frac`；`--attn-head conv/unet`；`--mask-dir masks_band/masks_box`；`--flip-legacy`（复现旧的翻转 bug）；`--attn-weight-dir`（加权 attention 损失） | 2–4, 7–10, 13, 16, 18, 20 |
| `run_seeds.py` | 并行跑 5 folds × 10 seeds，已完成的跳过；实验名登记在 `EXPS` | 4, 7–10, 13, 16 |
| `aggregate_cv.py` | 合并 5 个 fold + paired 统计 → `runs_cv/cv_summary.md` | 4 起 |
| `check_data.py` | 核对数据目录和 mask | 2 |
| `diag_resolution.py` / `diag_misses.py` / `diag_box_size.py` | 诊断，只读预测 | 5 |
| `refine_with_band.py` / `vis_refine.py` | box refinement 首版 + 可视化 | 6 |
| `diag_strata.py` | 分层 recall + 定位质量 → `runs_cv/strata*/` | 11–16 |
| `infer_bandbox.py` | box refinement 正式版：`infer` 重推理缓存 attention map；`prefit` 预计算 hyperbola fit；`build` 全局选参、生成 `*_bandbox` / `*_gtbandbox` / `*_scale` / `*_fitbox` / `*_unet_*` 实验目录 | 12, 13, 15, 16 |
| `stage2_instance.py` | stage 2：`train`（五个 arm）/ `build`（选参、生成 `*_s2*` 目录）/ `diag`（窗口级诊断）；`--src` 换第一阶段 | 14 |
| `fit_hyperbola.py` | hyperbola fit 规则 | 15 |
| `diag_band_quality.py` | 预测 band 质量（IoU、按宽度 / grouped 分层覆盖率）→ `runs_cv/band_quality/` | 16 |
| `train_param.py` / `r17_analysis.py` | 参数化检测头（`augfix_{none, pseudo, param}`）+ 分析 → `runs_cv/r17/` | 17 |
| `diag_config_robust.py` | 同一组参数下 band 对 box-mask 的测试 F1（读 `sweep.md`），另按合并方式、验证前 20% 分开 → `runs_cv/bandbox/config_robust.md` | 18、19 |
| `data_efficiency.py` | 固定测试集 + 递增训练集的汇总表（读 `cv_summary.json`）→ `runs_cv/data_efficiency/summary.md` | 19 |
| `diag_attn_gaps.py` | attention 画不出来的部分：沿曲线覆盖率、分层覆盖率、阈值、重叠目标漏画 / 粘连（读 `bandbox_cache`）→ `runs_cv/attn_gaps*/` | 18 |
| `vis_errors.py` | 逐图画 TP / FP / FN，可叠加预测 band → `runs_cv/vis_errors/` | 诊断 |
| `extend_fit.py` | 补翼尖：沿拟合曲线低阈值续走、对称补翼（`infer_bandbox.py` 的 `*_fit{mirror, track, trackmirror, ext, wscale}`） | 20 |
| `make_weight_maps.py` | attention 损失的像素权重图（按目标均衡 + 翼尖加权）→ `datasets/*/wmaps_{band, box}_b<β>/`，`train_basic.py --attn-weight-dir` 读取 | 20 |
| `r20_compare.py` / `r20_lost_by_fit.py` / `r20_curve_nms.py` / `r20_ownership.py` | 第 20 轮配对比较（写 `runs_cv/r20/`）；fit 后丢掉的 grouped 目标分类；曲线去重、曲线归属探查 | 20 |
| `r22_assign.py` | 第 22 轮：fit 之后检测和曲线的配对规则（降分保留、几何判重、全图分配、解释掉再拟合），验证集选参、按 seed 配对（写 `runs_cv/r22/`） | 22 |
| `vis_band_resolution.py` | band 缩到 56 / 28 后保留多少（统计）；原 head vs U-shaped decoder 的预测并排图 + 翼尖放大 → `runs_cv/band_resolution/`。画图要两个 `final.pth`（`--conv-run` / `--unet-run`） | 5、16 补充 |

**结果**

| 位置 | 内容 | 能否引用 |
|---|---|---|
| `runs_cv/ds2/<exp>/f<fold>_seed<k>/` | 所有训练和推理生成的实验，每个 run：`preds_*.json`、`metrics*.json`、`run_info.json`、`train_log.csv`（入库）；`final.pth`、`vis/` 只在跑它的机器上 | ✅ 正式 |
| `runs_cv/ds2/*_win5060ti/` | Round 4 两个主 arm 和 Round 7 前 36 个 oracle run 的 5060 Ti 原始结果；`aug_none` / `aug_abs`（official box）只有这一版 | ✅ 硬件复现对照 |
| `runs_cv/cv_summary.md/.json` | 全部配对汇总 | ✅ 正式 |
| `runs_cv/bandbox/`、`s2/`、`band_quality/`、`band_resolution/`、`strata*/`、`attn_gaps*/`、`r17/` | 选参记录（`chosen.json`、`sweep.md`）、窗口级诊断、band 质量、分层分析、日志 | ✅ |
| `runs_cv/refine/` | Round 6 首版 box refinement | ✅（被 Round 12 取代） |
| `runs_cv/bandbox_cache/`、`s2_cache/` | 中间缓存 | 不入库，可重算 |
| `runs/` | Round 2、3（single fold、初始化未对齐） | ❌ 只作对照 |
| `runs_legacy_init_1114/` | Round 1 | ❌ 作废 |
| `runs_smoke/` | smoke test | ❌ |

**生成的汇总文件**（脚本自动写，不要手改；`strata*/summary.md` 标题都写着"第 11 轮"，是脚本模板，实际轮次看下表）

| 文件 | 生成脚本 | 对应 Round |
|---|---|---|
| `runs_cv/cv_summary.md` | `aggregate_cv.py`（不带参数即可，`DEFAULT_PAIRS` 包含全部 202 对） | 4 起，全部配对汇总 |
| `runs_cv/refine/summary.md`、`sweep.md` | `refine_with_band.py` | 6 |
| `runs_cv/strata/summary.md` | `diag_strata.py` | 11 |
| `runs_cv/strata_r12/` … `strata_r16/summary.md` | `diag_strata.py` | 12 … 16（`strata_r15_fair/` = Round 15 的 stage 2 plain 对 真实 band + fit） |
| `runs_cv/strata_r17/`、`strata_flipfix/summary.md` | `diag_strata.py` | 17、18 |
| `runs_cv/bandbox/config_robust.md` | `diag_config_robust.py` | 18、19 |
| `runs_cv/data_efficiency/summary.md` | `data_efficiency.py` | 19 |
| `runs_cv/attn_gaps/`、`attn_gaps_flipfix/summary.md` | `diag_attn_gaps.py` | 18（`attn_gaps/` 是修翻转前的 run） |
| `runs_cv/r20/`（`summary.md`、`stage1.md`、`strata/`、`attn_gaps/`、`*.txt`） | `r20_*.py`、`diag_strata.py`、`diag_attn_gaps.py` | 20 |
| `runs_cv/r22/`（`summary.md`、`summary_stage1.md`、`grid*_*.json`） | `r22_assign.py` | 22 |
| `runs_cv/bandbox/sweep.md` | `infer_bandbox.py build` | 12、13、15、16 选参 |
| `runs_cv/s2/sweep.md`、`diag.md` | `stage2_instance.py build` / `diag` | 14 |
| `runs_cv/band_quality/summary.md` | `diag_band_quality.py` | 16 |
| `runs_cv/vis_errors/seed0/index.md` | `vis_errors.py` | 诊断图索引 |

**权重在哪台机器**：`hbatbox_flipfix_*`（Round 18、19，含 `*_frac*`）全部在 IWAC1（IWAC2 跑的那一半已合并过来，IWAC2 上另有一份在 `~/flipfix_mirror/`）。**注意**：IWAC2 的 `~/flipfix_mirror/` 和 `~/s2_mirror_r14/` 还是 11-22 整理目录之前的布局（`basic_ds2/`、`detector_comparison/`），在那里续跑要先按新布局（`exp1_basic/`、`shared/`）同步脚本和数据。旧 run：IWAC1：none / abs / oracle / extent / lam0.1 / lam0.3 / boxmask / abs_unet；IWAC2：s4 / frac / gate / lam1.5；boxmask_unet 在 IWAC2 的 `~/s2_mirror_r14/`。stage 2 的权重不保存，可用同一命令重算（两台机器逐位一致）。

**数据**（`../shared/datasets/`）：`ds2_f<0-4>_s100_hbatbox/`（HBAT box，主）、`ds2_f<0-4>_s100/`（official box）。每个目录两套框的 JSON 都有。标注唯一来源 `data/additional_utilities/hbat/annotations.json`。训练图 246 的唯一目标没有双曲线参数（unparam），mask 全黑；全部 6 个 unparam 目标在 HBAT box 模式下改用 official box。

## 附录 C：潜在问题清单

1. **数据集 1 还没用严格协议跑过**（最大空白）。Round 14–16 的结论也只在数据集 2 上成立。
2. **baseline 的框回归弱**（Round 14）：不用 band、只加 stage 2 是 0.632，高于 band + fit 的 0.585。11-21 决定 stage 2 不在主线上（它证明的是"再回归一次框有用"，与 band 无关），band 的对照是 box-mask control。审稿人如果问"更强的检测器上 band 还有没有用"，目前的证据只有 YOLOv8s + band（为零）。
3. ~~band 特有效应 CI 下界贴着 0~~：Round 18 用 hyperbola fit 后，对 box-mask control +0.086 / 对 box-mask 最好的后处理 +0.044，都是 10/10，换参数仍成立。仍未做：强检测器上的复现、数据集 1。
4. fusion 效应（+0.004 / +0.019）和换显卡差别（+0.013）同量级，n = 10 分辨不出 0.02 以下。
5. 预测 band 的测试 IoU 在三种设计下都只有 0.38–0.48，训练图上可达 0.76。数据量变大之前，继续改结构预期收益很小。
6. "不降低检测精度"是很弱的主张；整体精度低（F1 0.47，从零训练、每折 91 张）。
7. validation set 只有约 70 个目标，选参有 overfitting 风险（Round 6 逐 seed 选参比全局差 0.023）。Round 18 的 band + fit 验证 / 测试是 0.587 / 0.585（原 head）、0.598 / 0.585（U-shaped decoder），差距已经很小；box-mask 臂 Voronoi 选参在测试上偏低，是 Voronoi 上 band 特有效应不稳健的原因。
8. Round 15 的 fit 规则在验证集上（用真实 band）迭代了三版，NMS 是看了验证集才加的；测试集没参与。
9. Round 14 的 oracle（0.756）和 Round 15 的真实 band + fit（0.658 / 0.719）都把真值的一部分交给了系统，只能当上限读。Round 14 的 stage 2 没有重打分，训练窗口用的是抖动后的 GT 而不是第一阶段的真实输出：它是对照，不是调到最好的系统。
10. 用的是最后一个 epoch 的权重，训练末尾 loss 还在波动。
11. official box 和 HBAT box 的 granularity 不同；HBAT box 按定义是 band 的外接矩形，评估 GT 和方法的归纳偏置一致。
12. **翻转增强 bug**（Round 17 发现、Round 18 修好）：Round 3–16 的 run 每个翻转样本目标偏左 8 px。修好后所有配对结论不变，但正式数字以 `hbatbox_flipfix_*` 为准。
13. ~~λ 没扫~~（R10）；~~只试了 concat~~（R8、10）；~~56×56 上 band 只有 2 格~~（R16 换成 224×224，测试 IoU 只从 0.38 到 0.42）。

## 附录 D：在新机器上复现 / 续跑

仓库历史 11-15 重置为单个提交。每个 run 的轻量结果已入库；`*.pth`、可视化图、`*.npz`、缓存没有入库。

```bash
git clone https://github.com/youryugi/shuangquxian.git
cd shuangquxian
pip install -r requirements.txt        # torch 按机器的 CUDA 版本装，见文件内注释

# 1. 重建数据目录（datasets/ 不入库，约 1 分钟）
cd experiments/revision/exp3_comparison/detectors
for f in 0 1 2 3 4; do
  python prepare_data.py --dataset ds2 --fold $f --subset 100 --box-source hbat
  python prepare_data.py --dataset ds2 --fold $f --subset 100 --box-source official
done

# 2. 跑实验（已有 metrics.json 的 run 自动跳过），例如：
cd ../exp1_basic
python run_seeds.py --exps hbatbox_flipfix_none hbatbox_flipfix_abs hbatbox_flipfix_boxmask --jobs 10 --workers 1   # --workers 1 必须，翻转决定才和已有 run 一致

# 3. box refinement（需要 final.pth）
python infer_bandbox.py infer --exps hbatbox_flipfix_abs hbatbox_flipfix_boxmask
python infer_bandbox.py prefit --exps hbatbox_flipfix_abs hbatbox_flipfix_boxmask
python infer_bandbox.py build --variants hbatbox_flipfix_abs_fitbox hbatbox_flipfix_boxmask_fitbox

# 4. 汇总全部配对
python aggregate_cv.py
```

注意：
- 换 GPU 后结果不和原机器逐位一致，但 paired initialization 由 CPU seed 决定，跨机器相同。最干净的做法是把要配对的 arm 都在同一张卡上跑。
- 需要 `final.pth` 的步骤（`infer_bandbox.py infer`、`refine_with_band.py`、`stage2_instance.py --src`）要在有权重的机器上跑，或先重训对应实验。

**11-15 实际执行记录（IWAC1 + IWAC2）**：两台 Linux 机器各一块 RTX 5090，conda 环境 `gpr`（Python 3.10，torch 2.11.0 + cu128）。Round 4 的两个主 arm 和 Round 7 的 oracle 在 5090 上从头重跑，5060 Ti 的结果改名为 `*_win5060ti/`。同一配置在两台 5090 上逐位一致（`train_log.csv`、`preds_*.json` md5 相同），所以配对的两臂可以分在不同机器上。`--jobs 10 --workers 1` 时单 run 约 75 s（stride 8）；10 → 16 个进程总吞吐不变，10 个就是上限。
