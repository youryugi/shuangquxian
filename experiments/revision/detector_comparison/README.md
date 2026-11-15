# 完整检测器对比 + 带监督插件实验（R1-3 / R2-3）

对应 [revision_map.md](../../../paper/revision/revision_map.md) 实验 3 和 R1-3 下的"插件实验设计思路"。
两件事共用一套数据、一套评估脚本：

1. **对比表**：YOLOv8 / RT-DETR / Faster R-CNN（预训练与从零各一组），和本文方法放进同一张表，附参数量、FLOPs、速度。
2. **插件实验**：YOLOv8 vs YOLOv8 + 双曲线带辅助监督（接法 A：只加辅助损失，YOLO 结构不动，推理零开销）。

运行环境：`C:/Users/79152/.conda/envs/gpr/python.exe`（torch 2.10 + cu128，ultralytics 8.4.70，torchvision 0.25）。
下文命令都在本目录执行。

---

## 1. 文件

| 文件 | 作用 |
|---|---|
| `common.py` | 读划分文件、官方框、HBAT 标注；用 `tools/annotation/hbat.py` 的画图代码生成带掩码（和标注界面逐像素一致） |
| `prepare_data.py` | 把一个划分（数据集 × fold × 训练子集）导出成 YOLO 格式目录：图像、官方框标签、带掩码 PNG、对照掩码、`gt_boxes.json` |
| `evaluate.py` | **统一评估**。所有模型（包括本文方法）都只需输出同一格式的预测 JSON；给出 F1@IoU0.5、AP50、配对统计 |
| `band_yolo.py` | **插件本体**：掩码作为第 4 通道跟着 ultralytics 自己的增广走；辅助头接在颈部 P3；BCE+Dice 损失 |
| `train_ultralytics.py` | 训练 YOLOv8 / YOLO11 / RT-DETR（加 `--band` 启用插件），导出预测，写指标 |
| `train_frcnn.py` | 训练 torchvision Faster R-CNN（R50-FPN v2），同样的输出 |
| `model_stats.py` | 参数量、FLOPs、单张推理时延 |
| `run_matrix.py` | 按实验名 × 数据集 × seed 顺序跑完整矩阵，自动跳过已完成的 |

输出目录：`datasets/`（准备好的数据）、`runs/`（正式结果）、`runs_smoke/`（冒烟测试）。这些目录不要提交到 git。

## 2. 步骤

### 第 1 步：准备数据

```powershell
python prepare_data.py --dataset ds2 --fold 0 --subset 100
python prepare_data.py --dataset ds1 --fold 0 --subset 100
```

- 训练集用每个来源组的全部图（数据集 1 的 3 个光度副本都用），验证集和测试集只用 `eval_file`。
- 检测框一律用**官方框**（数据集 1：Roboflow labelTxt；数据集 2：`gt_boxes_official_aug6`），HBAT 只提供掩码。这和 2026-11-14 的决定一致。
- 掩码：`masks_band/`（HBAT 带并集）和 `masks_box/`（官方框填实，R1-1 的对照）。以后 thickness / span 敏感性掩码也放成同级文件夹，训练时 `--mask-dir` 换目录即可。
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
python evaluate.py compare runs ds2 yolov8s_pre yolov8s_pre_band --dataset ds2
python evaluate.py compare runs ds2 yolov8s_scratch yolov8s_scratch_band --dataset ds2
```

看配对 ΔF1 的符号是否一致（3 个 seed 全正才算有方向）。有方向再进第 4 步；没有，按 revision_map 的退路走。

### 第 4 步：完整矩阵（10 个 seed）

```powershell
python run_matrix.py --datasets ds2 ds1 --seeds 0-9
python evaluate.py summary runs
python model_stats.py runs
```

默认实验集合：`yolov8s_pre, yolov8s_pre_band, yolov8s_scratch, yolov8s_scratch_band, yolov8s_pre_boxmask, frcnn_r50_pre, rtdetr_l_pre`。
λ 扫描：`--exps yolov8s_pre_band_g0.5 yolov8s_pre_band_g2 yolov8s_pre_band_g5`。

粗估单个 run 的时间（RTX 5060 Ti 8 GB，100 epoch）：YOLOv8s 约 3–6 分钟，RT-DETR-L 约 15–25 分钟，Faster R-CNN 约 10 分钟。全矩阵两个数据集 10 seed 约 1–2 天。

### 第 5 步：把本文方法放进同一张表

本文方法的训练脚本只需多写两个文件：`preds_val.json`、`preds_test.json`，格式见下；然后

```powershell
python evaluate.py run <run_dir> --data datasets/ds2_f0_s100
```

就能得到和其他检测器完全同口径的数字。

## 3. 评估口径（所有模型一致）

- 预测格式：`{图像stem: [[x1, y1, x2, y2, score], ...]}`，原图像素坐标，置信度阈值放到 0.001 全部导出。
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
