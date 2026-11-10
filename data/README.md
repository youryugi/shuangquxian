# 数据集来源说明

本目录下的两个数据集都来自公开数据集。本文只记录**公开来源本身的信息**：发布页面、官方说明、官方文件。
信息整理日期：2026-11-10。

| 本地目录 | 公开来源 | 来源确认程度 |
|---|---|---|
| `additional_utilities/` | Mendeley Data：GPR 地下管线与空洞数据集（DOI 10.17632/ww7fd9t325.1） | **已确认**：下载官方文件逐字节比对 |
| `primary_roboflow/` | Roboflow Universe：`gpr-eywbm/gpr-hhalx`（论文所引） | **未逐文件确认**：页面拒绝自动访问，信息来自搜索引擎索引；同为 442 张的 `annotation-v5zxq/gpr-9pdij` 特征与本地数据吻合 |

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

- `raw_131/` 的 131 张图片与官方 `Utilities/` 同名且逐字节相同。
- `augmented_247_hbat/` 中的全部图片都能在官方 `augmented_utilities/` 中找到同名且逐字节相同的文件。
- 本地标注（`annotations.json` 等）是自行重新标注的，不是官方 YOLO/VOC 标注。

---

## 2. Roboflow GPR 数据集（`primary_roboflow/`）

### 2.1 论文所引项目：`gpr-eywbm/gpr-hhalx`

Roboflow Universe 页面拒绝自动访问（HTTP 403）。以下来自搜索引擎索引摘要，**需在浏览器中核对**。

| 项目 | 内容 |
|---|---|
| 项目 | GPR（Object Detection），workspace `gpr-eywbm` |
| 地址 | <https://universe.roboflow.com/gpr-eywbm/gpr-hhalx> |
| 图像数 | 442 张（workspace 页面显示；同 workspace 另有空项目 “GPR 010824m”） |
| 项目描述 | 未发布（页面显示 “A description for this project has not been published yet”） |
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

版本号、划分、预处理与增强设置未能从公开索引中获取。

### 2.2 同为 442 张的相关项目：`annotation-v5zxq/gpr-9pdij`

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

另一相关项目 `gpr-uurq0/gpr-liosq`（<https://universe.roboflow.com/gpr-uurq0/gpr-liosq>）：上传者 “Gpr”，2023 年，CC BY 4.0，单类 “Hyperbola”，数据版本 `gpr-liosq/2` 共 408 张，描述同样为 170 张开源双曲线图像。三个项目可能是同一批数据的不同副本，尚未确认。

### 2.3 判断依据（本地数据的特征与 `gpr-9pdij` 公开信息对照）

| 公开信息 | 本地数据 |
|---|---|
| 缩放到 640×640 | 405 张全部为 640×640 |
| 每张训练图输出 3 份 | 135 组，每组正好 3 份 |
| 增强只有亮度、曝光、噪声，没有翻转 | 405 对副本中没有翻转关系；副本间相关系数中位数 0.97，亮度差中位数 27.5 灰度，可见椒盐噪点 |
| 训练集 408 张 | 405 张，少 1 组（3 张） |
| 验证 / 测试各 17 张未增强原图 | 本地没有 |
| — | 文件名为 Roboflow 导出格式 `<原名>_jpg.rf.<哈希>.jpg`；`labels_dota_405/` 为 Roboflow DOTA 导出格式，类别 `hyperbola` |

从文件名前缀看，原图来自三类来源，画面风格不同：纯数字编号（如 `3198`，带坐标轴的蓝色剖面）、`hyperbolaN`（带时间刻度的蓝色剖面）、`lineXXXX`（灰度裁剪图）。Roboflow 页面没有说明原图的采集设备和地点。

### 2.4 尚未确认的事项

- 需登录 Roboflow，下载 `gpr-hhalx`（论文所引）的 DOTA 格式导出包（训练、验证、测试三个划分都要），与 `all_405/` 逐文件比对哈希；如不一致，再比对 `gpr-9pdij` v1。
- `gpr-hhalx` 的版本号、划分与增强设置需在页面上核对，并在论文引用中注明版本。

---

## 参考链接

- Mendeley Data：<https://data.mendeley.com/datasets/ww7fd9t325/1>
- GitHub（Mendeley 数据集代码与 README）：<https://github.com/LCSkhalid/GPR_Data>
- Data in Brief 论文记录：<https://library.kab.ac.ug/Record/doaj-art-7c4c1b5e386a490689810e36f7dcc759>
- Roboflow `gpr-hhalx`（论文所引）：<https://universe.roboflow.com/gpr-eywbm/gpr-hhalx>
- Roboflow `gpr-9pdij` v1：<https://universe.roboflow.com/annotation-v5zxq/gpr-9pdij/dataset/1>
- Roboflow `gpr-liosq`：<https://universe.roboflow.com/gpr-uurq0/gpr-liosq>
