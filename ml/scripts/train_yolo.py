import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on converted hyperbola dataset.")
    parser.add_argument("--data", type=str, default="../yolo_dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device: 0/cpu")
    parser.add_argument("--project", type=str, default="../runs", help="Training output folder")
    parser.add_argument("--name", type=str, default="hyperbola_yolo", help="Experiment name")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=args.name,
    )

    print("Training finished.")
    print("Best weight usually saved under: <project>/<name>/weights/best.pt")


if __name__ == "__main__":
    main()
