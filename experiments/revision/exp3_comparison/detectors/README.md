# 完整检测器对比 + 带监督插件实验（R1-3 / R2-3）

对应 [revision_map.md](../../../../paper/revision/revision_map.md) 实验 3 和 R1-3 下的"插件实验设计思路"。
两件事共用一套数据、一套评估脚本：

1. **对比表**：YOLOv8 / RT-DETR / Faster R-CNN（预训练与从零各一组），和本文方法放进同一张表，附参数量、FLOPs、速度。
2. **插件实验**：YOLOv8 vs YOLOv8 + 双曲线带辅助监督（接法 A：只加辅助损失，YOLO 结构不动，推理零开销）。

运行环境：`C:/Users/79152/.conda/envs/gpr/python.exe`（torch 2.10 + cu128，ultralytics 8.4.70，torchvision 0.25）。
下文命令都在本目录执行。

> **2026-11-21 起，正式对比以第 0 节为准。** 第 2 节第 3–4 步是 11-14 的旧计划：只跑 fold 0，预训练和从零训练都用全套增广。它只留作冒烟和先导的参考。插件实验的正式结果在 [exp3_comparison/yolo_band/README.md](../yolo_band/README.md)。

---

## 0. 审稿人要求的对比：当前思路（2026-11-21）

> **结果（2026-11-22，数据集 2，5 折合并 × 10 seeds，详见 [0.7](#07-结果2026-11-22)）**
>
> - **条件对齐时（都不预训练、都只左右翻转），本文 + band 的 F1 比 YOLOv8s 高**：0.496 对 0.452，差 +0.044 [+0.027, +0.062]。但 AP50（0.388 对 0.409）和 F1@0.7（0.238 对 0.287）是 YOLOv8s 高：它的框更贴。
> - **band 监督在 YOLOv8s 从零训练上也显著有效**：F1 0.452 → 0.485，+0.034 [+0.013, +0.056]，8 胜 2 负，Wilcoxon p = 0.020；AP50 +0.030 [+0.011, +0.047]。幅度和本文轻量模型上的 +0.023 [+0.006, +0.040] 相当。在预训练加全套增广的 YOLOv8s 上则不显著（+0.014 [−0.008, +0.034]）。
> - **差距主要来自预训练**：YOLOv8s 只翻转时，预训练比从零高 +0.147（10/0/0）；预训练之上全套增广再加 +0.098（10/0/0）。Faster R-CNN 预训练、只翻转时 F1 0.724，是整张表最高的，代价是 43M 参数、451 GFLOPs。
> - 还没做：从零 YOLOv8s 的 box-mask 对照（R1-1），以及预训练、只翻转的 YOLOv8s 加 band。没有后者，就分不清预训练 YOLO 上 band 不起作用，是因为预训练还是因为全套增广。

### 0.1 审稿人要什么

| 来源 | 原话（节选） | 要求 |
|---|---|---|
| 副编辑 | "comparison with representative GPR detection methods is insufficient" | 列在五个关键问题里，必须回应 |
| R1-3 | "include a comparison with representative GPR target detection methods" | 和 GPR 检测方法比较 |
| R2-3 | "clarify the distinction from GPR-Former and geometric-attention approaches … compare against suitable complete detectors rather than only substituted attention modules" | 讲清和这些方法的区别；对比表里要有完整检测器 |

拆开是三件事：

1. **讲清区别**（写作，不需要实验）：做一张定性表，把 GPR-Former、Li 2025（曲率自适应注意力）、Wang 2025（双级联网络）和本文放在一起。比较四项：
   - 先验从哪里进入网络：输入、结构，还是监督目标；
   - 需要什么标注；
   - 推理开销；
   - 能不能插到别的检测器上。
2. **完整检测器对比**：本节主体，见 0.2–0.6。
3. **原 Table III 按新划分重跑**（SE / CBAM / Non-local / CA）。旧数字（F1 0.89）来自有同源泄漏的随机划分，不能再用。这件事和"要不要和完整检测器比"无关，无论如何都要做。这些模块要加进 `exp1_basic/train_basic.py`（旧实现在 `experiments/submission/primary_yolohead/generic_attention_baselines.py`），所以需要先和 exp1_basic 那边协调。

### 0.2 处理原则

- **论文的贡献是监督方式。** 轻量网络是受控的测试平台，主证据是同一检测器里"有 / 无 band attention"的配对比较。完整检测器表的作用是**给出绝对水平作参考**，不用来比输赢。回复信先讲这一点。
- **做一张小表。** 不拒绝：副编辑点了名，拒绝的风险最大。也不全面比拼：工作量大，而且等于接受了"要比精度"的框架。
- **审稿人为什么会要这张表：论文自己把这些检测器摆成了对手。**
  - §II-A 写了 "YOLO-based detectors struggled with complex hyperbolas due to local features and NMS limitations"（2main.tex:214）和 "existing methods mainly rely on axis-aligned bounding boxes and post-hoc fitting"（:218）；
  - 摘要和贡献 2 写了 "improving detection performance"。

  所以措辞要一起改：:214 那句我们没有证据支持，删掉或弱化；摘要和贡献里写明"在受控的轻量检测器上验证"。
- **建议（待定）：Discussion 里主动写一句"在强预训练检测器上，band 监督的增益有限"。** 依据是 exp3_comparison/yolo_band：YOLOv8s 预训练、全套增广下，band 对 none 是 +0.014 [−0.008, +0.034]，不显著。表里一出现 YOLO，审稿人下一句就会问"对 YOLO 有没有用"，主动说比被问到再答稳妥。

### 0.3 公平性：主表只开左右翻转

现有的 YOLOv8s 结果（F1 0.697）和本文方法（0.472 / 0.496）之间，不止一个条件不同：

| | YOLOv8s（exp3_comparison/yolo_band 那组） | 本文方法（exp1_basic flipfix） |
|---|---|---|
| 预训练 | COCO | 无，从零训练 |
| 增广 | mosaic、缩放、平移、HSV、左右翻转；最后 10 个 epoch 关 mosaic | 只有左右翻转 |
| 输入和容量 | 640（224 的原图放大），约 11M 参数 | 224，三层 CNN |
| 训练 | 100 epoch，batch 16，ultralytics 默认超参 | 120 epoch，batch 8，Adam 5e-4 |

- **决定（11-21，用户）：主对比只开左右翻转，和本文方法一样。** 论文针对的是小样本场景，而强增广和大规模预训练本身就是应对小样本的另一类办法。混在一起比，就说不清差距从哪来。
- **保留一行"YOLOv8s 预训练、全套增广"作参考**（已完成），放补充材料或回复信。它用来回应"关掉增广等于削弱 YOLO"的质疑：我们报了它的正常水平，主表用的是对齐的条件。
- **输入分辨率（建议）：保持 640，在论文里写明。** 放大不增加信息，640 也是 YOLO 预训练时的尺度。改成 224 的话，YOLO 最深一层特征只有 7×7，对它不利。

### 0.4 表里放哪些行（数据集 2）

| 行 | 预训练 | 增广 | 状态 |
|---|---|---|---|
| 本文 baseline（`hbatbox_flipfix_none`） | 无 | 左右翻转 | 完成：F1 0.472 ± 0.030 |
| 本文 + band（`hbatbox_flipfix_abs`） | 无 | 左右翻转 | 完成：0.496 ± 0.021；比 baseline +0.023 [+0.006, +0.040]，8/0/2 |
| **YOLOv8s 从零** | 无 | 左右翻转 | 完成（11-22）：0.452 ± 0.021。条件和本文完全对齐，是最关键的一行 |
| YOLOv8s 预训练 | COCO | 左右翻转 | 完成（11-22）：0.599 ± 0.027。和上一行比，单独看出预训练的作用 |
| Faster R-CNN R50-FPN v2 | COCO | 左右翻转（`train_frcnn.py` 本来就只翻转） | 完成（11-22）：0.724 ± 0.016。也是 §II 里 Pham、Lei 用的两阶段路线 |
| YOLOv8s 从零 + band | 无 | 左右翻转 | 完成（11-22）：0.485 ± 0.018。看小样本、无预训练时 band 对 YOLO 有没有用 |
| 参考：YOLOv8s 预训练、全套增广 | COCO | 全套 | 完成：0.697 ± 0.019，AP50 0.726 |

每行都报参数量和 FLOPs，见 0.7。单张推理时延还没测（`model_stats.py`）。

**不进主表，在回复里说明即可：**

- **GPR-Former**：没找到公开代码。11-21 只搜过一次，同名的 umfieldrobotics/GPRFormer 是火星车定位，和这篇无关。建议同时写信向作者要代码（待定）。
- **RT-DETR**：代码已经写好，可选。
- **C3**（经典的 GPR 双曲线检测 + 拟合）：有公开代码 [lweileeds/hyperbola_recognition](https://github.com/lweileeds/hyperbola_recognition)，MATLAB，没有 license，要移植到 Python 或用 Octave 跑。可选，待定。
- **§II 引的 GPR 检测方法**：检测部分都是通用检测器，已被上表覆盖。Pham、Lei 用 Faster R-CNN；Liu 用 SSD / YOLOv3；Sun、Zhu 用 YOLOv5 / v7 加曲线拟合。它们多出来的只是事后拟合参数。
  - 如果要比参数估计：exp3_comparison/yolo_band Round 3 已有 band 拟合和框几何的对比，被边缘截断的目标上 slope 误差 ×1.09 对 ×1.24。
  - 还缺一个估计器："在 YOLO 框里对原图像素做拟合"。可以在 CPU 上做，可选。

### 0.5 协议和实现

- **协议**：和全部 revision 实验相同。
  - 5 折合并 × 10 seeds；
  - HBAT 框作真值，official 框作旁证；
  - 在合并验证集上选阈值；
  - 按 seed 配对，报 bootstrap CI 和 Wilcoxon；
  - `--workers 2`；
  - 同一个 (fold, seed) 的各行放在同一张卡上跑。IWAC1 和 IWAC2 的结果逐位一致，见 exp3_comparison/yolo_band README。
- **只翻转不用改代码**：

  ```
  train_ultralytics.py --extra '{"mosaic":0,"mixup":0,"copy_paste":0,"hsv_h":0,"hsv_s":0,"hsv_v":0,"degrees":0,"translate":0,"scale":0,"shear":0,"perspective":0,"flipud":0,"fliplr":0.5}'
  ```

  冒烟时看 `train_batch0*.jpg`，确认没有拼图、缩放和亮度变化。band 那一路有自己的增广类（`BandRandomHSV` 等），也要确认它们遵守这些参数。
- **从零训练的轮数**：100 epoch 可能不够。先用 fold 0、只看验证集定下来（预计 200–300），再跑满。
- **驱动脚本**：都用 `exp3_comparison/yolo_band/run_cv.py`（11-21 已加），结果放在 `exp3_comparison/yolo_band/runs_cv/ds2/<exp>/f<fold>_seed<k>/`。这样和参考行 `y8s_pre_none` 在同一个目录下，可以直接用 `exp1_basic/aggregate_cv.py` 配对。（`exp3_comparison/detectors/runs*/` 被 gitignore，所以不放这里。）

  | exp | 对应的行 |
  |---|---|
  | `y8s_pre_flip_none` | YOLOv8s 预训练、只翻转，100 epoch |
  | `y8s_scr_flip_none` / `y8s_scr_flip_band` | YOLOv8s 从零、只翻转（± band）；轮数由 `SCR_EPOCHS` 决定 |
  | `y8s_scr_flip_none_e100/e200/e300` | 从零训练的轮数候选，只跑 seed 0，按 5 折合并验证集 F1 选 |
  | `frcnn_pre` | Faster R-CNN，COCO 预训练，只翻转，50 epoch，取最后一个 epoch |

  - `train_frcnn.py` 新加了 `--run-name` 和 `--weights last`，run_info 里写 `fold`、`box_source`，默认行为不变。
  - `check_pairs` 现在按（模型, 增广）分组检查配对：全套增广和只翻转的第一个 batch 本来就不同；Faster R-CNN 没有 fingerprint，单独一组不检查。
- **规模**：每行 50 个 run。YOLO 三行，共 150 个；Faster R-CNN 50 个；轮数候选 15 个。

### 0.6 依赖和待定

- **用户决定（11-21）**：
  - 不给 GPR-Former 作者写信；
  - 不放 C3；
  - "YOLOv8s 从零 + band"要跑；
  - 写作的事（措辞、Discussion 那句话、回复信）先放着，等结果定下来再说。
- **数据集 1**：完整检测器不需要 band，可以先跑；但真值用 official 框还是 HBAT 框要先定。本文方法那一行要等 band 补标完。
- **Table III 重跑**：见 0.1 第 3 条。

### 0.7 结果（2026-11-22）

由 `cv_compare.py` 生成，输出 `cv_compare.md` / `cv_compare.json`。

**口径**：每个 seed 把 5 折的验证集预测合并，选 F1 最高的阈值，再在合并测试集（222 个目标）上算 P、R、F1。真值是 HBAT 框。表中是 10 个 seed 的均值 ± 标准差。F1@0.7 用同一个阈值，按 IoU 0.7 判定命中。参数量只算推理时的，band 辅助头不计入。

| 方法 | 预训练 | 增广 | F1 | P | R | AP50 | F1@0.7 | 参数量 (M) | GFLOPs | 输入 |
|---|---|---|---|---|---|---|---|---|---|---|
| 本文 baseline（轻量 CNN，无 attention） | 无 | 左右翻转 | 0.472 ± 0.030 | 0.511 | 0.451 | 0.371 ± 0.032 | 0.220 | 1.91 | 6.3 | 224 |
| 本文 + band attention | 无 | 左右翻转 | **0.496 ± 0.021** | 0.501 | 0.496 | 0.388 ± 0.030 | 0.238 | 2.05 | 6.8 | 224 |
| YOLOv8s 从零 | 无 | 左右翻转 | 0.452 ± 0.021 | 0.496 | 0.432 | 0.409 ± 0.029 | 0.287 | 11.1 | 28.4 | 640 |
| YOLOv8s 从零 + band 辅助监督 | 无 | 左右翻转 | 0.485 ± 0.018 | 0.521 | 0.466 | 0.439 ± 0.010 | 0.312 | 11.1 | 28.4 | 640 |
| YOLOv8s 预训练 | COCO | 左右翻转 | 0.599 ± 0.027 | 0.684 | 0.538 | 0.568 ± 0.018 | 0.427 | 11.1 | 28.4 | 640 |
| Faster R-CNN R50-FPN v2 预训练 | COCO | 左右翻转 | **0.724 ± 0.016** | 0.775 | 0.682 | 0.747 ± 0.016 | 0.523 | 43.3 | 450.8 | 640 |
| 参考：YOLOv8s 预训练、全套增广 | COCO | 全套 | 0.697 ± 0.019 | 0.792 | 0.623 | 0.726 ± 0.013 | 0.496 | 11.1 | 28.4 | 640 |

**配对比较**（同一 seed，B − A；同一套代码、初始化能对上的两行之间才配对。YOLO 那几组的 fingerprint 都核对过，200 组 0 个不一致）：

| 比较 | ΔF1 [95% CI] | 胜/平/负 | ΔAP50 [95% CI] |
|---|---|---|---|
| 本文：band attention 的作用 | +0.023 [+0.006, +0.040] | 8/0/2 | +0.017 [−0.003, +0.036] |
| **YOLOv8s 从零：band 辅助监督的作用** | **+0.034 [+0.013, +0.056]**（Wilcoxon p = 0.020） | 8/0/2 | **+0.030 [+0.011, +0.047]** |
| YOLOv8s：预训练的作用（都只翻转） | +0.147 [+0.121, +0.167] | 10/0/0 | +0.159 [+0.135, +0.183] |
| YOLOv8s 预训练：全套增广的作用 | +0.098 [+0.085, +0.112] | 10/0/0 | +0.159 [+0.149, +0.169] |
| （已有）YOLOv8s 预训练、全套增广：band 的作用 | +0.014 [−0.008, +0.034] | 8/0/2 | +0.007 [−0.009, +0.024] |

**独立比较**（两组 seed 没有配对关系，用 bootstrap 估两组均值之差）：

| 比较 | ΔF1 [95% CI] | ΔAP50 [95% CI] |
|---|---|---|
| 本文 + band − YOLOv8s 从零 | +0.044 [+0.027, +0.062] | −0.021 [−0.046, +0.003] |
| 本文 baseline − YOLOv8s 从零 | +0.021 [−0.001, +0.042] | −0.038 [−0.065, −0.013] |

**怎么读**：

1. **条件对齐时，本文方法在主指标 F1 上不输 YOLOv8s。** 不加 band 时两者打平（+0.021，CI 碰到 0）；加 band 后高 +0.044。代价是 AP50 和 F1@0.7 都是 YOLOv8s 高，说明它的排序和框的贴合度更好，本文方法的优势在选定阈值处的工作点。
2. **band 监督在两种检测器上方向一致、大小相当。** 本文 +0.023，YOLOv8s 从零 +0.034，前提都是不预训练、只翻转。到了预训练加全套增广的 YOLOv8s 上降到 +0.014，不显著。这和论文"标注少、没有大规模先验时，几何监督最有用"的说法一致。但现在**分不清**是预训练还是全套增广让 band 失效，因为缺"预训练、只翻转 + band"这一组。
3. **按用户定的标准（比不加 attention 好），两种检测器都过了。** R1-1 要的 box-mask 对照在轻量模型上做过。flipfix 结果：band 0.496，boxmask 0.504；band − boxmask = −0.008 [−0.027, +0.011]，3 胜 7 负，打平。从零 YOLOv8s 上还没做这个对照。
4. **绝对水平由预训练决定。** 预训练 +0.147，全套增广再 +0.098。Faster R-CNN 预训练、只翻转是 0.724，比全套增广的 YOLOv8s 还高，代价是 43M 参数和 451 GFLOPs。
5. **计算量**：本文模型的参数最少（2.05M），但 224 输入下是 6.8 GFLOPs，并不比 YOLOv8s 省（YOLOv8s 在 640 输入下 28.4 GFLOPs，按面积换算到 224 约 3.5）。所以如果要强调"轻量"，只能说参数少，不能说 FLOPs 低。

**附带核对**：

- **从零训练的轮数**：按 5 折合并验证集选。100 epoch 的 F1 0.448，200 epoch 0.410，300 epoch 0.418，选 100（`exp3_comparison/yolo_band/runs_cv/scr_epochs.md`）。第一次跑 200 / 300 时被 ultralytics 的早停（patience 100，按验证集 mAP 判断）截在 122–200 轮，学习率还没退火完。之后长于 100 轮的 run 一律 `--patience 0`（`run_cv.py` 的 `EP()`）。被截断的那批移到了会话草稿目录，不入库。
- **阈值网格的下限**：只翻转的 YOLO 选出的阈值很低，10 个 seed 里有 2–4 个落在网格最低点 0.01。把网格往下延到 0.001 重算，测试集 F1 变化 ≤ 0.005，而且是变低（从零 0.452 → 0.449，预训练 0.599 → 0.594）；band 的配对差仍是 +0.035 [+0.014, +0.058]。说明网格下限没有压低 YOLO 的成绩。
- **复现**：`y8s_scr_flip_none` 和 `y8s_scr_flip_none_e100` 在 seed 0 的 5 折 `preds_test.json` 逐位相同，两者只差 `--patience` 参数，而 100 epoch 下这个参数不起作用。
- **运行记录**：2026-11-21 22:49 – 11-22 00:31。YOLO 全部在 IWAC1 跑。Faster R-CNN 的前 8 个在 IWAC2，中途按用户安排让出过 IWAC2，剩下 42 个 23:30 起回到 IWAC2；结果都已拷回（`runs_cv/run_cv_iwac2_frcnn.log`）。

**下一步（可选，每组 50 个 run，单卡约 30 分钟）**：

- `y8s_scr_flip_boxmask`：R1-1 在 YOLO 上的对照；
- `y8s_pre_flip_band`：分清预训练和全套增广，哪个让 band 失效。

exp1_basic 那边的 Round 21 已经排在两台机器上，开跑前要先和那边协调。

---

## 1. 文件

| 文件 | 作用 |
|---|---|
| `../../shared/common.py` | 读划分文件、官方框、HBAT 标注；用 `tools/annotation/hbat.py` 的画图代码生成带掩码（和标注界面逐像素一致） |
| `../../shared/prepare_data.py` | 把一个划分（数据集 × fold × 训练子集）导出成 YOLO 格式目录：图像、官方框标签、带掩码 PNG、对照掩码、`gt_boxes.json` |
| `../../shared/evaluate.py` | **统一评估**。所有模型（包括本文方法）都只需输出同一格式的预测 JSON；给出 F1@IoU0.5、AP50、配对统计 |
| `band_yolo.py` | **插件本体**：掩码作为第 4 通道跟着 ultralytics 自己的增广走；辅助头接在颈部 P3；BCE+Dice 损失 |
| `train_ultralytics.py` | 训练 YOLOv8 / YOLO11 / RT-DETR（加 `--band` 启用插件），导出预测，写指标 |
| `train_frcnn.py` | 训练 torchvision Faster R-CNN（R50-FPN v2），同样的输出 |
| `model_stats.py` | 参数量、FLOPs、单张推理时延 |
| `run_matrix.py` | 按实验名 × 数据集 × seed 顺序跑完整矩阵，自动跳过已完成的 |

输出目录：`../../shared/datasets/`（准备好的数据）、`runs/`（正式结果）、`runs_smoke/`（冒烟测试）。这些目录不要提交到 git。

## 2. 步骤

### 第 1 步：准备数据

```powershell
cd ../../shared
python prepare_data.py --dataset ds2 --fold 0 --subset 100
python prepare_data.py --dataset ds1 --fold 0 --subset 100
```

- 训练集用每个来源组的全部图（数据集 1 的 3 个光度副本都用），验证集和测试集只用 `eval_file`。
- 检测框一律用**官方框**（数据集 1：Roboflow labelTxt；数据集 2：`gt_boxes_official_aug6`），HBAT 只提供掩码。这和 2026-11-14 的决定一致。
- 掩码：`masks_band/`（HBAT 带并集）和 `masks_box/`（官方框填实，R1-1 的对照）。thickness / span 等标注敏感性掩码放成同级文件夹 `masks_ann_<档位>/`（[实验 2](../../exp2_annotation_sensitivity/README.md)），训练时 `--mask-dir` 换目录即可。
- **带标注还没补齐时**，默认会报错并列出缺带的训练项。冒烟测试可加 `--missing-mask skip`（把缺带的项从训练集里去掉，两个臂都去掉，保证公平）。正式实验必须在标注补齐后用默认模式重新准备。
- 递增训练集（R2-6）：`--subset 25/50/75`，各自单独一个目录。

### 第 2 步：冒烟测试（2 个 epoch，几分钟）

```powershell
python run_matrix.py --smoke
```

检查三件事：
1. 两个 run 都生成了 `metrics.json`；
2. **`runs_smoke/ds2/yolov8n_pre_band/seed0/train/train_batch0_band.jpg`**：训练图经过 mosaic、翻转、缩放之后，绿色带是否仍然贴在回波上、红框是否套住目标。这是"增广同步"的直接证据，正式跑之前必须看；
3. 训练日志里多了一列 `band_loss`，且在下降。

### 第 3 步：先导实验（3 个 seed，只看方向）

```powershell
python run_matrix.py --datasets ds2 --seeds 0 1 2 --exps yolov8s_pre yolov8s_pre_band yolov8s_scratch yolov8s_scratch_band
python ../../shared/evaluate.py compare runs ds2 yolov8s_pre yolov8s_pre_band --dataset ds2
python ../../shared/evaluate.py compare runs ds2 yolov8s_scratch yolov8s_scratch_band --dataset ds2
```

看配对 ΔF1 的符号是否一致（3 个 seed 全正才算有方向）。有方向再进第 4 步；没有，按 revision_map 的退路走。

### 第 4 步：完整矩阵（10 个 seed）

```powershell
python run_matrix.py --datasets ds2 ds1 --seeds 0-9
python ../../shared/evaluate.py summary runs
python model_stats.py runs
```

默认实验集合：`yolov8s_pre, yolov8s_pre_band, yolov8s_scratch, yolov8s_scratch_band, yolov8s_pre_boxmask, frcnn_r50_pre, rtdetr_l_pre`。
λ 扫描：`--exps yolov8s_pre_band_g0.5 yolov8s_pre_band_g2 yolov8s_pre_band_g5`。

粗估单个 run 的时间（RTX 5060 Ti 8 GB，100 epoch）：YOLOv8s 约 3–6 分钟，RT-DETR-L 约 15–25 分钟，Faster R-CNN 约 10 分钟。全矩阵两个数据集 10 seed 约 1–2 天。

### 第 5 步：把本文方法放进同一张表

本文方法的训练脚本只需多写两个文件：`preds_val.json`、`preds_test.json`，格式见下；然后

```powershell
python ../../shared/evaluate.py run <run_dir> --data ../../shared/datasets/ds2_f0_s100
```

就能得到和其他检测器完全同口径的数字。

## 3. 评估口径（所有模型一致）

- 预测格式：`{图像stem: [[x1, y1, x2, y2, score], ...]}`，原图像素坐标，置信度阈值放到 0.001 全部导出。
- **预测框先裁到图内**（2026-11-21 起）：每个预测框裁到 [0, width] × [0, height] 再匹配，只比较图片范围内的重叠；GT 框本来就裁过（`hbat.hyperbola_to_bbox`）。完全在图外的预测面积为 0，只能算 FP。ultralytics 自己已经裁过，YOLO 的数字不受影响；exp1_basic 的旧 run 已在新口径下重算（约 15% 的预测框越界，配对结论的变化都在 0.01 以内，见 exp1_basic Round 17–18）。
- 匹配规则：按置信度从高到低，一对一贪心匹配，IoU ≥ 0.5（和论文原 `evaluate()` 一致）。
- **主指标：测试集 F1，置信度阈值取验证集上 F1 最高的那个**（"val-selected"）。这样 YOLO、RT-DETR、Faster R-CNN、本文方法都在各自最合适的阈值上，没有谁被不公平的阈值拖累。
- 同时报告：AP50（全点插值）、固定阈值 0.25 / 0.5 的 F1、测试集上最优阈值的 F1（"oracle"，只作参考，不写进正文）。
- 数据集 1 另按图像类别（pipe / rebar）和图像风格分别给出 F1。
- 配对统计（`evaluate.py compare`）：按 seed 配对的 ΔF1 均值、bootstrap 95% 置信区间、Wilcoxon 符号秩检验 p 值、胜/平/负次数（R2-8）。
- 模型选择：ultralytics 用其默认 fitness（验证集 mAP），Faster R-CNN 用验证集 AP50。都只看验证集，不看测试集。
- 导出预测时的 NMS：ultralytics 默认 IoU 0.7，torchvision 默认 0.5。写进论文的实验设置。

## 4. 插件（`band_yolo.py`）在做什么

用大白话：YOLO 一个零件不改，训练时多布置一道作业——在颈部 P3 那层特征（640 输入下 80×80）旁边接一个很小的头，让它画"双曲线带在哪"，用 HBAT 导出的掩码批改。这道作业的答案**不传回**检测分支（接法 A），所以推理时可以直接扔掉这个头，速度和参数量与原版 YOLO 一样。

技术细节：

| 项 | 做法 | 为什么 |
|---|---|---|
| 掩码跟随增广 | 掩码作为图像的第 4 通道一起进 ultralytics 的 Mosaic / RandomPerspective / Flip / LetterBox | 几何变换对 4 个通道完全相同，不可能对不齐 |
| 颜色抖动、BGR→RGB | 只对前 3 个通道做（`BandRandomHSV`、`BandFormat`） | 检测分支看到的输入和原版 YOLO 完全一样 |
| 填充值 | mosaic / letterbox 用 114 填充，掩码通道按 >0.5（即 >127）二值化 | 填充区域正确地算背景 |
| 接哪一层 | 默认 `--band-layer 15`（yolov8 颈部 P3/8 输出，检测小目标那一路）；可选 4（骨干 P3/8） | 离检测头近、通道够 |
| 损失分辨率 | 头的输出双线性放大、掩码平均池化到 imgsz/4（160×160），用"每格里带占几成"的软目标 | 带在 80×80 上不到两格宽，直接比会碎成点 |
| 损失 | BCE + Dice，权重 `--band-gain`（默认 1.0；ultralytics 的 box/cls/dfl 增益是 7.5/0.5/1.5） | 与论文注意力分支相同 |
| 热身 | 前 `--band-warmup 3` 个 epoch 线性加大带损失 | 新头随机初始化，开头别冲击预训练骨干 |
| 对照 | `--mask-dir masks_box`：同一个头，答案换成填实的官方框 | 分离"带的形状"和"多了一个任务"（R1-1） |

接法 B（把注意力图拼回特征）暂未实现——按计划，A 有效再做 B。

## 5. 已知限制 / 待办

- 数据集 1 目前只有 106 组有 HBAT 带（旧的 selected_318），其余 64 组（含钢筋排）要等补标注；冒烟测试用 `--missing-mask skip` 跳过了它们。正式实验前重新 `prepare_data.py`。
- 数据集 2 的 HBAT 文件 `data/additional_utilities/hbat/annotations.json` 目前是从旧标注自动换算的初值（`source: legacy_init`），还没在 HBAT 里逐个核对。
- 数据集 2 原图 224×224，这里统一放大到 640 训练（`--imgsz`），和 YOLO 预训练的尺度匹配；论文里要写明。
- GPR-Former 等专用方法没有公开代码，没有纳入；回复信里说明，用 RT-DETR 代表 Transformer 一路。
- `runs/` 下 ultralytics 的 best.pt 里 pickle 了 `band_yolo.BandAuxDetectionModel`，加载时必须先 `import band_yolo`（脚本里已处理）。
- **RT-DETR 的 ultralytics 8.4.70 bug**：开 mosaic 时它的数据管线不把 2×imgsz 的拼接画布裁回 imgsz，等于在 1280 px 上训练，显存翻 4 倍（8 GB 卡直接溢出）。`train_ultralytics.py` 里的 `RTDETRTrainerFixed` 把仿射输出尺寸固定为 imgsz，使其与 YOLO 的管线一致。升级 ultralytics 后要重新确认这一点。
- 参数量表里 YOLO+带监督的 `params_M_inference` 已去掉辅助头（训练专用）；FLOPs 本来就只算推理前向。
