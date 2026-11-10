# 基于现有标注格式的训练与推理（YOLO）

本目录提供一套最小可用流程：
1. 读取 `Utilities/annotations.json`（你的超曲线参数标注）
2. 自动转换为 YOLO 检测数据集
3. 训练检测模型
4. 推理并可导出回“近似同结构”的 JSON

> 说明：YOLO 检测本质输出的是矩形框，脚本会把框近似还原为 `x_vertex/y_vertex/width/height/thickness`，便于与你现有工具对接。

---

## 1) 安装依赖

在 `biaozhu/Utilities/ml` 下执行：

```bash
pip install -r requirements.txt
```

如果你有 GPU，建议先确认 PyTorch CUDA 版本与驱动匹配。

---

## 2) 生成训练数据

在 `biaozhu/Utilities/ml/scripts` 下执行：

```bash
python prepare_dataset.py --data-dir ../../ --output-dir ../yolo_dataset --train-ratio 0.8
```

生成后目录大致如下：

- `yolo_dataset/images/train`
- `yolo_dataset/images/val`
- `yolo_dataset/labels/train`
- `yolo_dataset/labels/val`
- `yolo_dataset/dataset.yaml`

---

## 3) 开始训练

在 `biaozhu/Utilities/ml/scripts` 下执行：

```bash
python train_yolo.py --data ../yolo_dataset/dataset.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

如果没有 GPU，把 `--device 0` 改成 `--device cpu`。

训练结果默认输出到：
- `../runs/hyperbola_yolo/weights/best.pt`

---

## 4) 推理

```bash
python infer_yolo.py --weights ../runs/hyperbola_yolo/weights/best.pt --source ../../ --conf 0.25 --device 0 --save-json --save-hyperbola-vis
```

常见 `--source`：
- 单张图片：`../../001.jpg`
- 图片文件夹：`../../`

输出：
- 可视化与 txt：`../runs/predict_hyperbola/`
- 近似标注 JSON：`../runs/predict_hyperbola/predictions_annotations.json`
- 双曲线形状可视化图：`../runs/predict_hyperbola/hyperbola_vis/`

---

## 5) 方案A：直接学习 5 个参数（回归）

你需要的参数为：
- `x_vertex`
- `y_vertex`
- `width`
- `height`
- `thickness`

这里采用“两阶段但参数直接回归”的做法：
1. 检测器定位目标区域
2. 参数回归网络直接输出 5 参数（不是固定比例反算）

### 5.1 准备回归数据

在 `biaozhu/Utilities/ml/scripts` 下执行：

```bash
python prepare_regression_dataset.py --data-dir ../../ --output-dir ../reg_dataset --train-ratio 0.8
```

输出：
- `../reg_dataset/crops/`（每个标注对象的裁剪图）
- `../reg_dataset/train.jsonl`
- `../reg_dataset/val.jsonl`

### 5.2 训练参数回归模型

```bash
python train_param_regressor.py --dataset-dir ../reg_dataset --epochs 80 --batch-size 32 --lr 1e-3 --device cuda
```

输出最佳权重：
- `../runs_reg/param_regressor/best.pt`

### 5.3 推理（检测 + 参数直回归）

```bash
python infer_param_regressor.py --det-weights ../runs/hyperbola_yolo/weights/best.pt --reg-weights ../runs_reg/param_regressor/best.pt --source ../../ --conf 0.25 --device cuda --save-json --save-hyperbola-vis
```

输出：
- 参数 JSON：`../runs_reg/predict_param_reg/predictions_annotations.json`
- 双曲线可视化：`../runs_reg/predict_param_reg/hyperbola_vis/`

说明：`det-weights` 仍然需要一个检测模型来找多个目标；但 `x_vertex/y_vertex/width/height/thickness` 是回归网络直接预测，不是固定规则反算。

---

## 6) 端到端检测头（单模型直接输出框+5参数）

该方案不再使用 “YOLO 检测 + 回归网络” 两阶段链路，而是单模型直接从整图输出：
- 目标存在置信度
- 目标框
- `x_vertex/y_vertex/width/height/thickness`

### 6.1 训练

在 `biaozhu/Utilities/ml/scripts` 下执行：

```bash
python train_e2e_detector.py --data-dir ../../ --img-size 640 --batch-size 8 --epochs 120 --lr 1e-3 --device cuda
```

输出权重：
- `../runs_e2e/e2e_detector/best.pt`

### 6.2 推理

```bash
python infer_e2e_detector.py --weights ../runs_e2e/e2e_detector/best.pt --source ../../ --conf 0.35 --nms-iou 0.5 --device cuda --save-json --save-hyperbola-vis
```

输出：
- 参数 JSON：`../runs_e2e/predict_e2e/predictions_annotations.json`
- 双曲线可视化：`../runs_e2e/predict_e2e/hyperbola_vis/`

脚本文件：
- `scripts/e2e_detector_model.py`
- `scripts/train_e2e_detector.py`
- `scripts/infer_e2e_detector.py`

---

## 参数映射规则（你的格式 -> 检测框）

由你的标注参数生成框：
- `x_min = x_vertex - width/2`
- `x_max = x_vertex + width/2`
- `y_min = y_vertex - thickness/2`
- `y_max = y_vertex + height + thickness/2`

推理时从检测框反算回参数（近似）：
- `x_vertex = x_center`
- `width = bbox_width`
- `thickness = bbox_height * thickness_ratio`（默认 0.25）
- `height = bbox_height - thickness`
- `y_vertex = y_center - height/2`

可通过 `infer_yolo.py` 的 `--thickness-ratio` 调整反算风格。
