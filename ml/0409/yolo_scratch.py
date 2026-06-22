"""
YOLO baseline —— 不带 backbone（from scratch，随机初始化，无预训练权重）。

与你的方法 / 朴素 CNN baseline 同等条件（无外部数据先验）的公平对比。
yolov8n.yaml 只加载网络结构，不加载任何预训练权重。

运行：
  pip install ultralytics
  python yolo_scratch.py
"""
from yolo_common import run_experiment

if __name__ == "__main__":
    run_experiment("yolov8n.yaml", "yolo_scratch")
