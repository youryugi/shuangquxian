# 共享：数据导出与统一评估

实验 1–3 都用这里的数据目录和评估代码，不要另写口径。

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 把一个划分（数据集 × fold × 训练子集）导出成 YOLO 格式目录：图像、标签、`masks_band/`（HBAT 带掩码）、`masks_box/`（填实的框，box-mask 对照）、`gt_boxes*.json`、`data.yaml`、`manifest.json` |
| `common.py` | 读划分文件、官方框、HBAT 标注；带掩码直接调用 `tools/annotation/hbat.py` 的画图代码（和标注界面逐像素一致） |
| `evaluate.py` | 统一评估：预测框先裁到图内，IoU 0.5 一对一贪心匹配，验证集选阈值，测试集 F1 为主，另报 AP50；`compare` 子命令做按 seed 配对的统计 |
| `datasets/` | 导出的数据目录（不入库，用 `prepare_data.py` 重建）。正式实验用 `ds2_f{0-4}_s100_hbatbox`（目标 = HBAT 框）。实验 2 的扰动掩码 `masks_ann_<档位>/` 也放在这些目录里，由 [make_annot_masks.py](../exp2_annotation_sensitivity/make_annot_masks.py) 生成 |

```bash
python prepare_data.py --dataset ds2 --fold 0 --subset 100 --box-source hbat   # → datasets/ds2_f0_s100_hbatbox/
```

`data.yaml` 里写的是绝对路径（ultralytics 需要）；整个目录换位置后，要重新导出或改 `path:` 一行。

参数和评估口径的详细说明见 [实验 3 README](../exp3_comparison/detectors/README.md) 第 2 节第 1 步和第 3 节。
