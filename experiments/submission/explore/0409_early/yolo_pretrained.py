"""
YOLO baseline —— 带 backbone（加载 COCO 预训练权重）。

工业界常用配置，展示"即使 YOLO 用了 COCO 预训练，你的方法是否仍有竞争力"。
yolov8n.pt 会加载在 COCO 上预训练好的权重（首次运行自动下载，需联网）。

运行：
  pip install ultralytics
  python yolo_pretrained.py
"""
from yolo_common import run_experiment

if __name__ == "__main__":
    run_experiment("yolov8n.pt", "yolo_pretrained")
