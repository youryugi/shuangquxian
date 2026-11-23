# Hyperbola-Band Attention Supervision (JSTARS-2026-06074)

GPR B-scan 双曲线检测：HBAT 参数化标注 + 双曲线带掩码显式监督注意力分支。当前处于大修（major revision）阶段。

**本页是唯一入口**：先在下表找到要看的文档，不用逐个翻 README。

## 找东西：看哪个文件

| 想知道 | 看这里 |
|---|---|
| **现在的结论、下一步做什么** | [experiments/revision/exp1_basic/README.md](experiments/revision/exp1_basic/README.md) 开头"答案"和"下一步" |
| 大修各实验的状态（审稿意见 → 实验 → 做到哪了） | [experiments/revision/README.md](experiments/revision/README.md) |
| 实验 2：标注敏感性（厚度、跨度、顶点、slope、标注误差） | [experiments/revision/exp2_annotation_sensitivity/README.md](experiments/revision/exp2_annotation_sensitivity/README.md) |
| 审稿意见逐条分析、修改计划、自查发现 S1–S5 | [paper/revision/revision_map.md](paper/revision/revision_map.md) |
| 实验 1（exp1_basic）某一轮（Round 1–22）具体怎么做的 | [experiments/revision/exp1_basic/rounds.md](experiments/revision/exp1_basic/rounds.md) |
| exp1_basic 的代码、结果目录、生成的汇总文件对应哪轮 | [exp1_basic/README.md 附录 B](experiments/revision/exp1_basic/README.md#附录-b文件与数据位置) |
| 在新机器上复现 / 续跑 | [exp1_basic/README.md 附录 D](experiments/revision/exp1_basic/README.md#附录-d在新机器上复现--续跑) |
| 数据导出、统一评估口径（实验 1–3 共用） | [experiments/revision/shared/README.md](experiments/revision/shared/README.md) |
| 实验 3：和成熟检测器对比（YOLOv8s / Faster R-CNN 等） | [experiments/revision/exp3_comparison/detectors/README.md](experiments/revision/exp3_comparison/detectors/README.md) |
| 实验 3：YOLOv8 + band 辅助监督（插件实验） | [experiments/revision/exp3_comparison/yolo_band/README.md](experiments/revision/exp3_comparison/yolo_band/README.md) |
| 投稿版论文每张表 / 图来自哪个脚本、哪次运行 | [experiments/submission/README.md](experiments/submission/README.md) |
| 数据集来源、目录、标注格式 | [data/README.md](data/README.md) |
| 分组划分（5 折、固定测试集） | [data/splits/README.md](data/splits/README.md)（生成文件） |
| 标注工具 HBAT 用法 | 下方目录结构 `tools/annotation/hbat.py` 一行 |

`runs*/` 下面的 `summary.md`、`sweep.md`、`cv_summary.md` 等都是脚本自动生成的结果表，不是文档，不要手改；它们由哪个脚本、哪一轮生成，见各实验 README 的"文件位置"附录。

## 写文档的约定

- 每个实验目录只放**一个** `README.md`：开头几行给答案 → 术语表 → 一页总览 → 下一步 → 附录（代码、结果、复现）。按时间线的逐轮记录超过几百行时拆到同目录的 `rounds.md`。
- 当前结论**只写在 exp1_basic README 开头一处**；其他文档只放链接，不复制数字（避免改一处漏一处）。
- 正式数字只认标为"正式结果"的那一节；标了"已取代"的不要引用。
- 术语用英文（baseline、arm、fold、recall、box refinement 等），README 开头放术语表。
- 结果目录里的 `*.pth / *.pt / *.png / *.jpg / *.npz` 不入库（.gitignore）；重跑脚本可以再生成。
- 运行环境：本机 `C:/Users/79152/.conda/envs/gpr/python.exe`（torch 2.10 + cu128，ultralytics 8.4.70）；IWAC1 / IWAC2（RTX 5090，conda `gpr`，torch 2.11 + cu128）。

## 目录结构

```
data/                        数据集（脚本按 <repo>/data/... 相对路径读取）；公开来源见 data/README.md
  primary_roboflow/            数据集 1（Roboflow Universe gpr-eywbm/gpr-hhalx）
    original_GPR.v1i.yolov5-obb/ 【修改稿用】v1 官方导出包，原样保存：train 408 / valid 17 / test 17，YOLOv5-OBB 官方标注
    hbat/                        【修改稿用】HBAT 输出，三个 split 共用一个 annotations.json；初始内容由 selected_318_hbat 按组名映射而来（source=legacy_init）
    all_405/                     旧：早先的导出，405 张（= 官方 train 去掉 line2888_0-95 一组）
    labels_dota_405/             旧：all_405 对应的 DOTA 格式标签
    selected_318_bbox/           旧：原论文筛选的 318 张（106 张原图 × 3），旧格式自标注 width/height/thickness，468 个目标
    selected_318_hbat/           旧：同一批 318 张图，HBAT 格式自标注 slope/span/thickness，486 个目标
  additional_utilities/        数据集 2（Mendeley, Mojahid et al.）
    original/                    官方下载包 GPR_data.rar 的管线部分，原样保存（2026-11-14 重新整理）
      original_Utilities/          【修改稿用】官方原图 131 张，无标注
      augmented_utilities/         官方增强图 786 张（131 × aug_1–6）+ annotations/（YOLO、VOC 各 786 个；官方标注只在这里，且每张增强图是分别手工标的）
    gt_boxes_official_aug6/      【修改稿用】原图的官方框：original/augmented_utilities/annotations/YOLO_format/<id>_aug_6.txt 改名为 <id>.txt，共 169 个框；HBAT 自动读取
    hbat/                        【修改稿用】HBAT 输出（annotations[_标注员].json、timelog、导出框），以后掩码也放这一层；初始内容由 legacy/raw_131 旧标注转换而来（source=legacy_init 表示尚未复核）
    legacy/                      旧论文留下的自标注（2026-11-14 移入；图片都与 original/ 逐字节相同，保留只为可追溯）
      raw_131/                     131 张原图 + 自行标注的 annotations.json（旧 width/height 格式，218 个目标）
      augmented_247_hbat/          原论文用的 247 张官方增强图（来自 71 张原图）+ HBAT 格式自标注 + 旧派生框
  splits/                      修改稿用的分组划分（全部数据、按来源组划分、5 折 + 固定测试集 + 嵌套训练子集），由 tools/splits/make_splits.py 生成

tools/
  annotation/hbat.py           HBAT 标注工具（论文 Fig. 1）。2026-11-14 改版：显示官方 GT 框并记录每条带所属的框（gt_box_id）、
                               四个目标标签（可见度/完整度/畸变/参数化状态）、顶点可放在图外、导出框裁到图内、每个目标的耗时日志（窗口右上角实时显示）、
                               按标注员分文件。用法：python hbat.py --folder <图像目录> --out <输出目录> [--annotator B] [--gt <框目录或json>] [--hide-gt] [--extend]
                               官方数据目录只读，输出一律用 --out 指到 data/<数据集>/hbat/
  annotation/seed_from_legacy.py 把旧自标注转成 HBAT 初值写入 data/<数据集>/hbat/annotations.json（已存在则不覆盖）
                               画带的函数 hyperbola_band_polygon / band_mask 在模块顶层，掩码导出和厚度扫描都要调用它们
  annotation/rect_annotator.py 旧版抛物线(width/height)标注工具加矩形视图，不是画矩形框的工具；与 legacy_bbox_annotator.py 基本相同
  annotation/legacy_bbox_annotator.py  旧版 width/height 标注工具（产生 selected_318_bbox 的标注）
  data_audit/                  修改稿用的数据与结果核查脚本
    revision_data_check.py       按审稿意见逐项检查数据能否满足（来源重叠、排除样本、可见度、分组划分、标注一致性）
    source_clusters.py           按图像相似度聚类“独立来源”，供分组划分使用
    paper_table_check.py         在全部结果 CSV 中查找与论文表格数字一致的运行
  splits/make_splits.py        生成 data/splits/ 分组划分文件（--report 查看不同相似度阈值下的分组）
  visualize_yolo_labels.py     标签可视化
  git_batch_upload.py          分批 git 提交上传

experiments/                 实验代码与结果。每个实验目录：README.md = 文档，*.py = 代码，runs*/ = 结果与日志
  submission/                投稿版实验（代码冻结，只读）→ experiments/submission/README.md
    primary_yolohead/          主数据集，YOLO 式检测头，640 输入 → Table II / III / Fig. 4
    additional_centernet/      第二数据集，CenterNet 热图头，224 输入 → Table IV
    cross_dataset/             跨数据集 → Table V
    explore/                   探索分支，未进论文（已删模型权重和可视化图）
  revision/                  大修阶段实验 → experiments/revision/README.md
    shared/                    共享：数据导出（prepare_data.py → datasets/）、统一评估（evaluate.py）、common.py
    exp1_basic/                实验 1 基础实验：有无 band attention、band + hyperbola fit；目前做了数据集 2（README = 结论；rounds.md = 逐轮记录）
    exp2_annotation_sensitivity/  实验 2 标注敏感性：扰动 HBAT 参数重画监督掩码（数据集 2）
    exp3_comparison/           实验 3 对比：和成熟检测器比
      detectors/                 YOLOv8s / Faster R-CNN 等检测器对比、band 插件代码（band_yolo.py）
      yolo_band/                 YOLOv8s + band 辅助监督（插件实验）

paper/
  manuscript/                  投稿版（JSTARS-2026-06074）：2main.tex + 1references.bib + 图；修改稿在此基础上改
  revision/                    审稿意见 reviewer_comments.txt；revision_map.md（逐条意见分析、环节总览、数据集能撑什么）
  figure_scripts/lambda_boxplot/  λ 箱线图（Fig. 6）作图脚本及数据 CSV

_待确认/                      不确定是否还需要的文件，见下文
```

## _待确认

| 文件 | 说明 |
|---|---|
| `utilities_annotation_versions/` | `raw_131` 的另外两个标注版本（164 / 219 个目标）；不确定 `augmented_247_hbat` 由哪个版本生成 |
| `selected_318_old_annotations_rect.json` | 318 张主数据集的旧版矩形标注 |
| `hbat_simple_0701.py` | HBAT 的早期简化版（比 `hbat.py` 早 3 天） |
| `manuscript_unused_figures/` | 投稿版目录中未被 2main.tex 引用的图：`bscan.png`（引用已注释）、`example-011/030`（第二数据集样例）、`11.png`、`unet.pdf` |

`augmented_247_hbat` 的增强脚本不在仓库中；302 条标注里有 55 张图片已被删除，只剩 247 张。

旧草稿（原 `paper/submission_v1`，即 git 中的 `writepaper/IEEE-final`）已删除，可从 git 历史恢复；它的 bib 只比投稿版多一条 `rw-gprsurvey2`（GPR 深度学习综述，Construction and Building Materials 2020）。
