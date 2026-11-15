# 数据集来源说明

本目录下的两个数据集都来自公开数据集。本文只记录**公开来源本身的信息**：发布页面、官方说明、官方文件。
信息整理日期：2026-11-10；2026-11-11 补充官方原始下载包的核验结果。

| 本地目录 | 公开来源 | 来源确认程度 |
|---|---|---|
| `additional_utilities/` | Mendeley Data：GPR 地下管线与空洞数据集（DOI 10.17632/ww7fd9t325.1） | **已确认**：下载官方文件逐字节比对 |
| `primary_roboflow/` | Roboflow Universe：`gpr-eywbm/gpr-hhalx` v1（论文所引） | **已确认**：2026-11-10 从论文所引项目下载 v1 官方导出包，与本地图像逐张比对像素一致（见 2.4） |

各子目录一览：

| 目录 | 内容 | 来源 |
|---|---|---|
| `additional_utilities/original/original_Utilities/` | 131 张管线原始剖面，无标注 | 官方 `GPR_data.rar` 中的 `Utilities/`，原样保存（2026-11-14 由 `additional_utilities/original_Utilities/` 移入 `original/`） |
| `additional_utilities/original/augmented_utilities/` | 786 张增强图 + `annotations/`（YOLO、VOC 各 786 个） | 官方 `augmented_utilities/`，原样保存。六个增强版本的框是分别手工标注的：总数 164–172 不等；几何不变的 `aug_1` 与 `aug_6` 之间，同一目标框中心偏移中位数 6.3 px、IoU 中位数 0.72（224 px 图） |
| `additional_utilities/legacy/raw_131/` | 同上 131 张 + 自行标注的 `annotations.json` | 图像与 `original_Utilities/` 逐字节相同 |
| `additional_utilities/legacy/augmented_247_hbat/` | 247 张官方增强图 + 自行标注 | 官方 `augmented_utilities/` 的子集 |
| `additional_utilities/gt_boxes_official_aug6/` | 131 个 YOLO 格式 `.txt`，共 169 个框，作为原图的官方框 | `original/augmented_utilities/annotations/YOLO_format/<id>_aug_6.txt` 原样复制并改名为 `<id>.txt`（2026-11-14）。`aug_6` 与原图几乎相同（1.7 节），叠图抽查 16 张，框与原图回波对齐。每图框数：1 个 99 张、2 个 28 张、3 个 2 张、4 个 2 张；不少单个框内含 2–4 条双曲线 |
| `primary_roboflow/original_GPR.v1i.yolov5-obb/` | 442 张，train/valid/test 三个划分，YOLOv5-OBB 标注 | `gpr-hhalx` v1 官方导出包，原样保存 |
| `primary_roboflow/all_405/`、`labels_dota_405/` | 405 张及其 DOTA 标注 | 早先的 `gpr-hhalx` 导出，只有 train 划分且缺 1 组 |
| `primary_roboflow/selected_318_*` | 从 405 张中选出的 106 组 × 3 = 318 张 + 自行标注 | 本研究筛选 |

---

## 1. Mendeley GPR 数据集（`additional_utilities/`）

### 1.1 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Intelligent recognition of subsurface utilities and voids: A Ground Penetrating Radar dataset for Deep Learning applications |
| 作者 | Abdelaziz Mojahid, Driss El Ouai, Khalid El Amraoui, Khalil El Hami, Hamou Aitbenamer |
| 单位 | Université Mohammed V de Rabat（Institut Scientifique / Faculté des Sciences），摩洛哥 |
| 平台 | Mendeley Data，<https://data.mendeley.com/datasets/ww7fd9t325/1> |
| DOI | 10.17632/ww7fd9t325.1 |
| 版本 / 发布日期 | Version 1，2024-11-28 |
| 许可 | CC BY 4.0（可共享、复制、修改，需署名、附许可链接并注明改动） |
| 学科分类 | Use of Computers in Earth Sciences; Geophysics; Civil Engineering; Deep Learning |
| 配套论文 | *Data in Brief*（Elsevier），Vol. 59，Article 111338，2025 年 4 月（ScienceDirect PII: S2352340925000708） |
| 代码仓库 | <https://github.com/LCSkhalid/GPR_Data>（含 `crop.py`、`augmentation.py`） |

### 1.2 官方描述（Mendeley 原文）

> The dataset consists of high-resolution Ground Penetrating Radar (GPR) data, specifically A-scan and B-scan profiles, obtained using antennas operating at 400 MHz and 200 MHz. This dataset contains 2239 images of buried utilities, voids, and intact zones, acquired across various infrastructure projects in Morocco between 2019 and 2024. These projects are located in different cities and towns across the country.

### 1.3 采集参数（GitHub README）

- 设备：两台 GSSI 天线，人工推行，测距轮（survey wheel）模式。
- 400 MHz 天线：时窗 50 ns，增益点 5，垂直滤波低通 800 MHz、高通 100 MHz，每道 512 采样点。
- 200 MHz 天线：时窗 150 ns，增益点 5，垂直滤波低通 400 MHz、高通 30 MHz，每道 512 采样点。
- 介电常数按所测介质分别设置。
- 剖面用 `crop.py` 裁剪为 **224×224** 像素（为适配 VGG19 等预训练骨干网络）。原始剖面尺寸未说明。

### 1.4 官方文件

Mendeley 只提供一个文件：

| 文件 | 大小 | SHA-256 |
|---|---|---|
| `GPR_data.rar` | 50,429,050 字节 | `223868f1b9432a674a129e4fc852eafcdde95d76241993825085b4435d3974f7` |

解压后的结构（数量为实际清点，与 README 一致）：

```
GPR_data/
├── Utilities/            131  管线原始剖面
├── augmented_utilities/  786  管线增强图像（131 × 6）
│   └── annotations/
│       ├── VOC_XML_format/  786 .xml
│       └── YOLO_format/     786 .txt
├── cavities/              79  空洞原始剖面
├── augmented_cavities/   553  空洞增强图像
│   └── annotations/ (VOC_XML_format 553 .xml, Yolo_format 553 .txt)
├── intact/                75  完好区原始剖面（无目标）
└── augmented_intact/     900  完好区增强图像（无标注）
```

- 2,239 张 = 增强图像总数（786 + 553 + 900）。
- 标注工具 makesense.ai，导出 YOLO（.txt）和 VOC XML 两种格式；类别为 Utility、Cavity。
- 只有增强图像带官方标注，原始剖面目录（`Utilities/` 等）没有标注文件。
- 增强图像命名为 `<原图编号>_aug_<k>.jpg`。管线类每张原图对应 `aug_1` 到 `aug_6` 共 6 张。

### 1.5 增强方法

`augmentation.py` 中列出的方法（GitHub README）：

- 几何变换：随机旋转、缩放、翻转
- 随机噪声
- 频谱平移（spectral shifting）
- 弹性形变（elastic deformation）
- 时间平移（time-shifting）

官方没有说明 `aug_1`–`aug_6` 各对应哪种方法。下表是把官方原图与官方增强图逐张比较得出的观察结果（管线类，前 40 张原图）：

| 编号 | 观察到的变换 | 依据 |
|---|---|---|
| aug_1 | 强随机噪声 | 位置不变，高频能量约为原图的 245 倍 |
| aug_2 | 小幅平移 | 亮度、对比度不变，图像整体错位 |
| aug_3 | 旋转（约 10°） | 图中竖直标记线倾斜 |
| aug_4 | 水平翻转 | 与原图左右翻转后的相关系数为 1.000 |
| aug_5 | 弹性形变 | 竖直标记线弯曲，高频能量降为约 0.66 |
| aug_6 | 与原图几乎相同 | 相关系数 1.000，像素残差约 0.5 灰度 |

### 1.6 引用

```bibtex
@article{mojahid2025gpr,
  author  = {Mojahid, Abdelaziz and El Ouai, Driss and El Amraoui, Khalid and El Hami, Khalil and Aitbenamer, Hamou},
  title   = {Intelligent Recognition of Subsurface Utilities and Voids: A Ground Penetrating Radar Dataset for Deep Learning Applications},
  journal = {Data in Brief},
  volume  = {59},
  pages   = {111338},
  year    = {2025},
  publisher = {Elsevier}
}

@misc{mojahid2024gprdata,
  author = {Mojahid, Abdelaziz and El Ouai, Driss and El Amraoui, Khalid and El Hami, Khalil and Aitbenamer, Hamou},
  title  = {Intelligent recognition of subsurface utilities and voids: A Ground Penetrating Radar dataset for Deep Learning applications},
  year   = {2024},
  publisher = {Mendeley Data},
  version = {V1},
  doi    = {10.17632/ww7fd9t325.1}
}
```

### 1.7 与本地目录的对应（核验结果）

- `original/original_Utilities/` 与 `legacy/raw_131/` 的 131 张图片都与官方 `Utilities/` 同名且逐字节相同。
- `legacy/augmented_247_hbat/` 中的全部图片都能在官方 `augmented_utilities/` 中找到同名且逐字节相同的文件。
- 本地标注（`annotations.json` 等）是自行重新标注的，不是官方 YOLO/VOC 标注。
- `legacy/augmented_247_hbat/` 的 247 张来自 71 张原图，只用了 `aug_1`（46 张）、`aug_2`（62）、`aug_4`（69）、`aug_6`（70）四种；每张原图对应 1–4 张增强图。
- `legacy/raw_131/annotations.json` 覆盖全部 131 张原图，其中 126 张有双曲线标注；有标注但未进入 `legacy/augmented_247_hbat/` 的原图有 56 张。

---

## 2. Roboflow GPR 数据集（`primary_roboflow/`）

### 2.1 论文所引项目：`gpr-eywbm/gpr-hhalx`

版本、图像数、划分、预处理、增强和许可取自官方导出包中的 `README.dataset.txt`、`README.roboflow.txt`、`data.yaml`（见 2.2）。项目描述和 workspace 信息来自搜索引擎索引（页面拒绝自动访问，HTTP 403）。

| 项目 | 内容 |
|---|---|
| 项目 | GPR（Object Detection），workspace `gpr-eywbm` |
| 地址 | <https://universe.roboflow.com/gpr-eywbm/gpr-hhalx> |
| 版本 | **v1**，“GPR - v1 2024-08-03 6:34am” |
| 图像数 | 442 张；划分为训练 408 / 验证 17 / 测试 17 |
| 类别 | 1 类，`hyperbola`；标注为旋转框（导出格式 YOLOv5 Oriented Object Detection） |
| 预处理（官方说明） | Auto-orientation（去除 EXIF 方向）；缩放到 640×640（Stretch） |
| 增强（官方说明） | “No image augmentation techniques were applied.”（但训练集实际含 3 倍副本，见 2.2） |
| 项目描述 | 未发布（页面显示 “A description for this project has not been published yet”）；同 workspace 另有空项目 “GPR 010824m” |
| 许可 | CC BY 4.0 |
| 引用 | 页面提供 BibTeX，年份 2024 |

```bibtex
@misc{gpr-hhalx_dataset,
  title        = {{GPR} Dataset},
  author       = {{GPR}},
  howpublished = {\url{https://universe.roboflow.com/gpr-eywbm/gpr-hhalx}},
  publisher    = {Roboflow},
  year         = {2024},
  month        = aug,
  note         = {Roboflow Universe, open source dataset, visited on 2026-07-31}
}
```

论文引用时应注明版本 v1。

### 2.2 官方导出包（`original_GPR.v1i.yolov5-obb/`）

2026-11-10 从 `gpr-hhalx` 下载 v1 的 YOLOv5-OBB 格式导出包，原样保存，未作任何改动（`README.roboflow.txt` 记录导出时间为 2026-11-10 14:36 GMT）。

```
original_GPR.v1i.yolov5-obb/
├── data.yaml            nc: 1, names: ['hyperbola']
├── README.dataset.txt
├── README.roboflow.txt
├── train/   images 408 + labelTxt 408   旋转框 1365 个
├── valid/   images  17 + labelTxt  17   旋转框   67 个
└── test/    images  17 + labelTxt  17   旋转框   72 个
```

- 全部 442 张都是 640×640。文件名为 `<原名>_jpg.rf.<哈希>.jpg`；标签每行 8 个角点坐标 + 类别名 + difficult 标志（DOTA 风格）。
- **训练集 408 张 = 136 张原图 × 3 份副本。** 同组 3 份几何上完全一致（与恒等、翻转、旋转等变换比较，272 对副本全部与恒等变换最相关），标签文件逐字相同，只有亮度、曝光和噪声不同（副本间相关系数中位数 0.97，亮度差中位数 27.5 灰度，可见椒盐噪点）。
- **验证、测试各 17 张，每张只有 1 份**，原名与训练集不重复。
- 官方说明写的是没有做增强，但训练集已经带了 3 倍副本。可见这些副本在 v1 生成之前就已经存在于上传的图像中。因此从 `gpr-hhalx` **拿不到这 136 张训练图的未增强原图**，也无法判断 3 份中哪一份最接近原图。
- 去掉副本后共 170 张不同原名的图像。按文件名前缀分为三类，画面风格不同：

| 前缀 | 画面 | 训练 | 验证 | 测试 | 合计 |
|---|---|---|---|---|---|
| 纯数字（如 `3198`） | 带坐标轴的蓝色剖面 | 34 | 8 | 3 | 45 |
| `hyperbolaN` | 带时间刻度的蓝色剖面 | 45 | 2 | 9 | 56 |
| `lineXXXX` | 灰度裁剪图 | 57 | 7 | 5 | 69 |
| 合计 | | 136 | 17 | 17 | 170 |

Roboflow 没有说明原图的采集设备和地点。

### 2.3 同为 442 张的相关项目：`annotation-v5zxq/gpr-9pdij`

以下来自搜索引擎索引摘要，**需在浏览器中核对**。它与 `gpr-hhalx` 的关系（复制 / 派生 / 同一上传者）未确认。

| 项目 | 内容 |
|---|---|
| 项目 | GPR，workspace “Annotation” |
| 地址 | <https://universe.roboflow.com/annotation-v5zxq/gpr-9pdij/dataset/1> |
| 版本 | v1，2024-08-03 6:25am（另有 v2，同为 442 张） |
| 图像数 | 442 张；划分为训练 408 / 验证 17 / 测试 17（92% / 4% / 4%） |
| 项目描述 | 170 张开源双曲线（hyperbola）图像 |
| 预处理 | Auto-Orient；缩放到 640×640（Stretch）；对比度调整为直方图均衡化 |
| 增强 | 每张训练图输出 3 份；亮度 −20% ~ +20%；曝光 −12% ~ +12%；噪声最多 1.29% 像素 |
| 许可 | CC BY 4.0 |

注：只有训练集做了增强。136 张训练原图 × 3 = 408；加上验证、测试各 17 张，共 170 张原图，与项目描述一致。

与 `gpr-hhalx` v1 对照：`gpr-9pdij` v1 生成于同一天的 6:25am，比 `gpr-hhalx` v1（6:34am）早 9 分钟；两者划分完全相同（408 / 17 / 17）；`gpr-9pdij` 的增强设置（训练图 ×3，亮度、曝光、噪声，无几何变换）正好对应 `gpr-hhalx` 训练集中副本的特征。**推测** `gpr-hhalx` 是导入 `gpr-9pdij` v1 的导出结果后建成的，所以增强副本被当作原图带了进来；未下载 `gpr-9pdij` 逐图核对，尚不能确认。

另一相关项目 `gpr-uurq0/gpr-liosq`（<https://universe.roboflow.com/gpr-uurq0/gpr-liosq>）：上传者 “Gpr”，2023 年，CC BY 4.0，单类 “Hyperbola”，数据版本 `gpr-liosq/2` 共 408 张，描述同样为 170 张开源双曲线图像。三个项目可能是同一批数据的不同副本，尚未确认。

### 2.4 与本地目录的对应（核验结果）

本地其他目录的文件名哈希与官方导出包不同（Roboflow 每次导出都会重新命名并重新压缩 JPEG），所以按原名分组后逐张比较像素。

- **`all_405/` = 官方 train 去掉 `line2888_0-95` 一组（3 张）。** 405 张与官方 train 一一对应：逐字节相同的只有 4 张，但像素平均绝对差中位数 0.13 灰度、最大 1.07 灰度，属于重新压缩造成的差异，图像内容相同。
- **`labels_dota_405/` 与官方 `labelTxt/` 一致。** 每张图的框数和顺序相同，坐标最多差 1 像素（141 张完全相同，其余为导出时的取整差异）。
- `selected_318_*` 的 106 组全部在官方 train 中。
- **官方 valid / test 的 34 张此前不在任何本地目录中，至今没有参与过任何实验。** 但它们与训练集不独立：
  - 用 `tools/data_audit/source_clusters.py` 的方法（缩放到 128×128，半幅模板滑窗匹配，取最大归一化相关系数）与 136 组训练图比较：相关系数 > 0.85 的有 10 张，> 0.80 的有 13 张。例如 `3289`、`3291`、`3292` 与训练集的 `3290` 相关系数均为 0.94，`line2888_0-0` 与 `line2888_0-95` 为 0.94，`line17836_0` 与 `line2888_0-50` 为 0.94。
  - 按文件名，34 张中有 5 张是剖面 `line2888` 的裁剪（训练集中有 16 组来自同一剖面），其中 3 张（`0-21`、`0-54`、`0-78`）相似度方法没有检出。灰度裁剪图之间相关系数普遍偏低，该方法对 `line*` 类可能漏检。
  - 合计：阈值 0.80 时至少 16 / 34 张在训练集中有同源图像。
  - 纯数字编号的 11 张（3170–3292）与训练集编号（3172–3296）交错，像是同一序列中的相邻帧。
- 官方 valid / test 只有 Roboflow 的旋转框标注，没有本研究的双曲线带标注。

### 2.5 尚未确认的事项

- 136 张训练原图的未增强版本：`gpr-hhalx` 中没有。`gpr-9pdij` 的 v1、v2 也都是 442 张，很可能同样带增强副本；需下载核对。
- 170 张原图的最初来源（采集设备、地点、是否含仿真或软件截图）在三个 Roboflow 项目中都没有说明。
- `gpr-hhalx` 与 `gpr-9pdij` 的关系（见 2.3 的推测）。

---

## 参考链接

- Mendeley Data：<https://data.mendeley.com/datasets/ww7fd9t325/1>
- GitHub（Mendeley 数据集代码与 README）：<https://github.com/LCSkhalid/GPR_Data>
- Data in Brief 论文记录：<https://library.kab.ac.ug/Record/doaj-art-7c4c1b5e386a490689810e36f7dcc759>
- Roboflow `gpr-hhalx`（论文所引）：<https://universe.roboflow.com/gpr-eywbm/gpr-hhalx>
- Roboflow `gpr-9pdij` v1：<https://universe.roboflow.com/annotation-v5zxq/gpr-9pdij/dataset/1>
- Roboflow `gpr-liosq`：<https://universe.roboflow.com/gpr-uurq0/gpr-liosq>
